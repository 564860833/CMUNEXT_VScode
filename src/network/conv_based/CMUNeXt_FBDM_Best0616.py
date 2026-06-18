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
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        return self.up(self.block(x))


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
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


class fusion_conv(nn.Module):
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


class FBDM(nn.Module):
    def __init__(
        self,
        channels,
        edge_channels=1,
        gate_init=0.01,
        gate_max=0.06,
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

        self.gate_max = float(gate_max)
        self.use_hspm_prior = False
        self.detach_hspm_prior = True
        self.semantic_uncertainty_weight = 0.7
        self.semantic_coarse_weight = 0.3
        self.semantic_gate_base = 0.7
        self.edge_aux_only = True
        self.eps = float(eps)
        self.dwt = FixedHaarDWT()
        self.sobel_edge = FixedSobelEdge(eps=eps)

        self.low_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels * 3,
                kernel_size=3,
                padding=1,
                groups=channels * 3,
                bias=False,
            ),
            nn.BatchNorm2d(channels * 3),
            nn.GELU(),
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.edge_branch = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.gate_proj = nn.Conv2d(3, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(channels, edge_channels, kernel_size=1)

        gate_ratio = float(gate_init) / self.gate_max
        self.gamma_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(gate_ratio), dtype=torch.float32))
        self.residual_scale = 1.0
        self.last_boundary_gate = None
        self.last_edge_logits = None

    def effective_gamma(self):
        return self.gate_max * torch.sigmoid(self.gamma_raw) * self.residual_scale

    def set_residual_scale(self, scale):
        self.residual_scale = min(max(float(scale), 0.0), 1.0)

    def build_boundary_prior(self, x):
        edge_prior = self.sobel_edge(x)
        semantic_prior = torch.ones_like(edge_prior)
        return edge_prior, semantic_prior, edge_prior

    def forward(self, x):
        low_feature = self.low_branch(x)
        _, lh, hl, hh = self.dwt(x)
        high_feature = self.high_branch(torch.cat((lh, hl, hh), dim=1))
        high_feature = F.interpolate(
            high_feature,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        _, semantic_prior, boundary_gate = self.build_boundary_prior(x)
        edge_feature = self.edge_branch(boundary_gate)

        residual = self.fusion(torch.cat((low_feature, high_feature, edge_feature), dim=1))
        residual_mean = residual.mean(dim=1, keepdim=True)
        gate_input = torch.cat((residual_mean, boundary_gate, semantic_prior), dim=1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        out = x + self.effective_gamma() * gate * residual
        edge_logits = self.edge_head(out)
        self.last_boundary_gate = boundary_gate
        self.last_edge_logits = edge_logits
        return x, edge_logits


class LightweightEdgeHead(nn.Module):
    def __init__(self, channels, edge_channels=1):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if edge_channels <= 0:
            raise ValueError("edge_channels must be positive.")

        self.head = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, edge_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


def _normalize_fbdm_stages(stages):
    try:
        normalized = tuple(sorted(set(int(stage) for stage in stages)))
    except (TypeError, ValueError) as exc:
        raise ValueError("fbdm_stages must contain only stages 0 and/or 1.") from exc
    if normalized not in {(0,), (1,), (0, 1)}:
        raise ValueError("fbdm_stages must be one of: (0,), (1,), or (0, 1).")
    return normalized


class CMUNeXt_FBDM_Best0616(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        fbdm_stages=(0,),
    ):
        super().__init__()
        self.fbdm_stages = _normalize_fbdm_stages(fbdm_stages)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.fbdm1 = (
            FBDM(channels=dims[0], edge_channels=num_classes)
            if 0 in self.fbdm_stages
            else None
        )
        self.x2_edge_head = (
            LightweightEdgeHead(channels=dims[1], edge_channels=num_classes)
            if 1 in self.fbdm_stages
            else None
        )

    def forward(self, x):
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

        edge_outputs = {}
        x1_skip = x1
        if self.fbdm1 is not None:
            x1_skip, edge_logits = self.fbdm1(x1)
            edge_outputs["edge"] = edge_logits
        if self.x2_edge_head is not None:
            x2_edge_logits = self.x2_edge_head(x2)
            if self.fbdm1 is None:
                edge_outputs["edge"] = x2_edge_logits
            else:
                edge_outputs["edge_x2"] = x2_edge_logits

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1_skip, d2), dim=1))

        return {
            "seg": self.Conv_1x1(d2),
            **edge_outputs,
        }


def cmunext_fbdm_best0616(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    fbdm_stages=(0,),
):
    return CMUNeXt_FBDM_Best0616(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        fbdm_stages=fbdm_stages,
    )
