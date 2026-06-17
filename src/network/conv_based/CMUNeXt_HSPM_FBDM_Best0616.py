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


class HighResolutionContext(nn.Module):
    def __init__(self, ch_in, ch_out, dilations=(1, 2, 3)):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        ch_out,
                        ch_out,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=ch_out,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ch_out),
                    nn.GELU(),
                )
                for dilation in dilations
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
        )

    def forward(self, x):
        residual = self.input_proj(x)
        context = torch.stack([branch(residual) for branch in self.branches], dim=0).sum(dim=0)
        return residual + self.fuse(context)


class ConfidenceAwarePrototypeMixer(nn.Module):
    def __init__(
        self,
        channels,
        temperature=0.1,
        gamma_init=0.1,
        gamma_max=0.3,
        eps=1e-6,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.temperature = float(temperature)
        self.gamma_max = float(gamma_max)
        self.mixer_mode = "legacy"
        self.eps = float(eps)
        self.prototype_scale = 1.0
        self.coarse_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.context_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _weighted_prototype(self, feature, weight):
        denominator = weight.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return (feature * weight).sum(dim=(2, 3), keepdim=True) / denominator

    def set_prototype_scale(self, scale):
        self.prototype_scale = min(max(float(scale), 0.0), 1.0)

    def effective_gamma(self):
        return self.gamma

    def forward(self, feature, use_prototype=True):
        coarse_logits = self.coarse_head(feature)
        coarse_prob = torch.sigmoid(coarse_logits)
        uncertainty = 4.0 * coarse_prob * (1.0 - coarse_prob)

        if not use_prototype:
            return feature, coarse_logits, uncertainty

        confidence = 1.0 - uncertainty
        foreground_weight = confidence * coarse_prob
        background_weight = confidence * (1.0 - coarse_prob)

        foreground_proto = self._weighted_prototype(feature, foreground_weight)
        background_proto = self._weighted_prototype(feature, background_weight)

        normalized_feature = F.normalize(feature, dim=1, eps=self.eps)
        normalized_foreground = F.normalize(foreground_proto, dim=1, eps=self.eps)
        normalized_background = F.normalize(background_proto, dim=1, eps=self.eps)

        foreground_similarity = (normalized_feature * normalized_foreground).sum(dim=1, keepdim=True)
        background_similarity = (normalized_feature * normalized_background).sum(dim=1, keepdim=True)
        assignments = torch.softmax(
            torch.cat([foreground_similarity, background_similarity], dim=1) / self.temperature,
            dim=1,
        )

        prototype_context = (
            assignments[:, 0:1] * foreground_proto
            + assignments[:, 1:2] * background_proto
        )
        enhanced_feature = feature + self.effective_gamma() * self.context_proj(prototype_context)
        return enhanced_feature, coarse_logits, uncertainty


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


class CMUNeXt_HSPM_FBDM_Best0616(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("CMUNeXt_HSPM_FBDM_Best0616 supports binary segmentation only.")

        self.hspm_mode = "full"
        self.hspm_backbone_mode = "dual_path"
        self.hspm_fusion_gate_max = 0.3
        self.hspm_fusion_mode = "global"
        self.hspm_small_area_threshold = 0.05
        self.hspm_small_area_temperature = 0.02
        self.last_fusion_diagnostics = None
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])

        self.high_resolution_context = HighResolutionContext(ch_in=dims[3], ch_out=dims[4])
        self.prototype_mixer = ConfidenceAwarePrototypeMixer(
            channels=dims[4],
            temperature=0.1,
            gamma_init=0.1,
            gamma_max=0.3,
        )

        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.hspm_projection = nn.Sequential(
            nn.Conv2d(dims[4], dims[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[3]),
        )
        gate_ratio = 0.05 / self.hspm_fusion_gate_max
        self.fusion_gate_raw = nn.Parameter(
            torch.tensor(_inverse_sigmoid(gate_ratio), dtype=torch.float32)
        )

        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.fbdm1 = FBDM(channels=dims[0], edge_channels=num_classes)

    def effective_fusion_gate(self):
        return self.hspm_fusion_gate_max * torch.sigmoid(self.fusion_gate_raw)

    def _record_fusion_diagnostics(self, effective_gate, deep_feature, injection):
        eps = torch.finfo(deep_feature.dtype).eps
        injection_rms = injection.detach().square().mean(dim=(1, 2, 3)).sqrt()
        deep_feature_rms = deep_feature.detach().square().mean(dim=(1, 2, 3)).sqrt().clamp_min(eps)
        injection_deep_rms_ratio = injection_rms / deep_feature_rms
        self.last_fusion_diagnostics = {
            "effective_fusion_gate": effective_gate.detach().mean(),
            "injection_deep_rms_ratio": injection_deep_rms_ratio.mean(),
        }

    def forward(self, x):
        self.last_fusion_diagnostics = None
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))

        hspm_feature = self.high_resolution_context(x4)
        hspm_feature, coarse_logits, uncertainty = self.prototype_mixer(hspm_feature)

        x5 = self.encoder5(self.Maxpool(x4))
        deep_feature = self.Up5(x5)
        deep_feature = self.Up_conv5(torch.cat((x4, deep_feature), dim=1))
        hspm_residual = self.hspm_projection(hspm_feature)
        effective_gate = self.effective_fusion_gate()
        injection = effective_gate * hspm_residual
        self._record_fusion_diagnostics(effective_gate, deep_feature, injection)
        decoder_input = deep_feature + injection

        d4 = self.Up4(decoder_input)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        x1_fbdm, edge_logits = self.fbdm1(x1)
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1_fbdm, d2), dim=1))

        return {
            "seg": self.Conv_1x1(d2),
            "coarse": coarse_logits,
            "uncertainty": uncertainty,
            "edge": edge_logits,
        }


def cmunext_hspm_fbdm_best0616(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
):
    return CMUNeXt_HSPM_FBDM_Best0616(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
    )
