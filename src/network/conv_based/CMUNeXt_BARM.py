import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_sigmoid(value, eps=1e-6):
    value = min(max(float(value), eps), 1.0 - eps)
    return torch.logit(torch.tensor(value)).item()


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                ch_in,
                                ch_in,
                                kernel_size=(k, k),
                                groups=ch_in,
                                padding=(k // 2, k // 2),
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(ch_in),
                        )
                    ),
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in),
                )
                for _ in range(depth)
            ]
        )
        self.up = ConvBlock(ch_in, ch_out)

    def forward(self, x):
        return self.up(self.block(x))


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class FusionConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):
        return self.conv(x)


class FixedHaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ).mul_(0.5)
        self.register_buffer("filters", filters.unsqueeze(1))

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("FixedHaarDWT expects input with shape (B, C, H, W).")

        _, channels, height, width = x.shape
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        filters = self.filters.to(dtype=x.dtype, device=x.device).repeat(channels, 1, 1, 1)
        coeffs = F.conv2d(x, filters, stride=2, groups=channels)
        batch, _, coeff_h, coeff_w = coeffs.shape
        coeffs = coeffs.view(batch, channels, 4, coeff_h, coeff_w)
        return (
            coeffs[:, :, 0],
            coeffs[:, :, 1],
            coeffs[:, :, 2],
            coeffs[:, :, 3],
        )


class FixedSobelEdge(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("FixedSobelEdge expects input with shape (B, C, H, W).")

        feature = x.mean(dim=1, keepdim=True)
        feature = F.pad(feature, (1, 1, 1, 1), mode="reflect")
        sobel_x = self.sobel_x.to(dtype=x.dtype, device=x.device)
        sobel_y = self.sobel_y.to(dtype=x.dtype, device=x.device)
        grad_x = F.conv2d(feature, sobel_x)
        grad_y = F.conv2d(feature, sobel_y)
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)
        edge_max = edge.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return edge / edge_max


class BoundaryAwareRefinement(nn.Module):
    def __init__(
        self,
        channels,
        edge_channels=1,
        gate_init=0.02,
        gate_max=0.2,
        hf_keep_init=0.3,
        eps=1e-6,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if edge_channels <= 0:
            raise ValueError("edge_channels must be positive.")
        if gate_max <= 0:
            raise ValueError("gate_max must be positive.")
        if not 0.0 < gate_init < gate_max:
            raise ValueError("gate_init must be in (0, gate_max).")
        if not 0.0 < hf_keep_init < 1.0:
            raise ValueError("hf_keep_init must be in (0, 1).")

        self.eps = float(eps)
        self.gate_max = float(gate_max)
        self.dwt = FixedHaarDWT()
        self.sobel = FixedSobelEdge(eps=eps)
        self.hf_atten = nn.Parameter(torch.full((3,), _inverse_sigmoid(hf_keep_init)))
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.delta_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(channels, edge_channels, kernel_size=1)
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_head.bias)
        gate_ratio = float(gate_init) / self.gate_max
        self.gamma = nn.Parameter(torch.tensor(_inverse_sigmoid(gate_ratio), dtype=torch.float32))
        self.register_buffer("residual_scale", torch.tensor(1.0, dtype=torch.float32))
        self.last_diagnostics = None

    def set_residual_scale(self, scale):
        self.residual_scale.fill_(min(max(float(scale), 0.0), 1.0))

    def effective_gamma(self):
        return self.gate_max * torch.sigmoid(self.gamma) * self.residual_scale

    def denoised_hf_energy(self, x):
        _, lh, hl, hh = self.dwt(x)
        keep = torch.sigmoid(self.hf_atten)
        high_frequency = torch.cat((lh * keep[0], hl * keep[1], hh * keep[2]), dim=1)
        energy = high_frequency.pow(2).mean(dim=1, keepdim=True).sqrt()
        energy = F.interpolate(
            energy,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return energy / energy.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)

    def forward(self, feature, seg_logits, uncertainty=None):
        seg_prob = torch.sigmoid(seg_logits.detach())
        band = self.sobel(seg_prob)
        if uncertainty is not None:
            uncertainty = F.interpolate(
                uncertainty.detach(),
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            band = (0.5 * band + 0.5 * uncertainty).clamp(0.0, 1.0)

        high_frequency = self.denoised_hf_energy(feature)
        boundary_feature = self.boundary_conv(torch.cat((feature, high_frequency, band), dim=1))
        raw_delta = self.delta_head(boundary_feature)
        logit_correction = self.effective_gamma() * band.detach() * torch.tanh(raw_delta)
        refined_logits = seg_logits + logit_correction
        edge_logits = self.edge_head(boundary_feature)

        with torch.no_grad():
            coarse_binary = torch.sigmoid(seg_logits) > 0.5
            refined_binary = torch.sigmoid(refined_logits) > 0.5
            self.last_diagnostics = {
                "effective_gamma": self.effective_gamma().detach(),
                "band_mean": band.detach().mean(),
                "hf_energy_mean": high_frequency.detach().mean(),
                "logit_correction_abs_mean": logit_correction.detach().abs().mean(),
                "logit_correction_abs_max": logit_correction.detach().abs().max(),
                "prediction_flip_ratio": (coarse_binary != refined_binary).to(seg_logits.dtype).mean(),
            }

        return refined_logits, edge_logits, band, logit_correction


class CMUNeXt_BARM(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        barm_gate_init=0.02,
        barm_gate_max=0.2,
        barm_hf_keep_init=0.3,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("CMUNeXt_BARM supports binary segmentation only.")
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = ConvBlock(input_channel, dims[0])
        self.encoder1 = CMUNeXtBlock(dims[0], dims[0], depths[0], kernels[0])
        self.encoder2 = CMUNeXtBlock(dims[0], dims[1], depths[1], kernels[1])
        self.encoder3 = CMUNeXtBlock(dims[1], dims[2], depths[2], kernels[2])
        self.encoder4 = CMUNeXtBlock(dims[2], dims[3], depths[3], kernels[3])
        self.encoder5 = CMUNeXtBlock(dims[3], dims[4], depths[4], kernels[4])

        self.Up5 = UpConv(dims[4], dims[3])
        self.Up_conv5 = FusionConv(dims[3] * 2, dims[3])
        self.Up4 = UpConv(dims[3], dims[2])
        self.Up_conv4 = FusionConv(dims[2] * 2, dims[2])
        self.Up3 = UpConv(dims[2], dims[1])
        self.Up_conv3 = FusionConv(dims[1] * 2, dims[1])
        self.Up2 = UpConv(dims[1], dims[0])
        self.Up_conv2 = FusionConv(dims[0] * 2, dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1)
        self.barm = BoundaryAwareRefinement(
            channels=dims[0],
            edge_channels=num_classes,
            gate_init=barm_gate_init,
            gate_max=barm_gate_max,
            hf_keep_init=barm_hf_keep_init,
        )

    def set_warmup_scale(self, residual_scale=1.0):
        self.barm.set_residual_scale(residual_scale)

    def get_barm_diagnostics(self):
        return self.barm.last_diagnostics

    def forward(self, x, uncertainty=None):
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))
        x5 = self.encoder5(self.Maxpool(x4))

        d5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))
        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))
        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        seg_coarse = self.Conv_1x1(d2)
        seg, edge, band, logit_correction = self.barm(
            d2,
            seg_coarse,
            uncertainty=uncertainty,
        )
        return {
            "seg": seg,
            "seg_coarse": seg_coarse,
            "edge": edge,
            "band": band,
            "logit_correction": logit_correction,
        }


def cmunext_barm(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    barm_gate_init=0.02,
    barm_gate_max=0.2,
    barm_hf_keep_init=0.3,
):
    return CMUNeXt_BARM(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        barm_gate_init=barm_gate_init,
        barm_gate_max=barm_gate_max,
        barm_hf_keep_init=barm_hf_keep_init,
    )
