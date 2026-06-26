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
        gamma_max=0.35,
        dropout=0.0,
        mixer_mode="bounded",
        eps=1e-6,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if gamma_max <= 0:
            raise ValueError("gamma_max must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if mixer_mode not in {"legacy", "bounded", "stable"}:
            raise ValueError("mixer_mode must be 'legacy', 'bounded', or 'stable'.")
        if mixer_mode != "legacy" and not 0.0 < gamma_init < gamma_max:
            raise ValueError("bounded gamma_init must be in (0, gamma_max).")

        self.temperature = float(temperature)
        self.gamma_max = float(gamma_max)
        self.mixer_mode = mixer_mode
        self.eps = float(eps)
        self.prototype_scale = 1.0
        self.coarse_head = nn.Conv2d(channels, 1, kernel_size=1)
        context_layers = [
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        ]
        if mixer_mode != "stable":
            context_layers.append(nn.GELU())
        if mixer_mode == "stable" and dropout > 0:
            context_layers.append(nn.Dropout2d(p=float(dropout)))
        self.context_proj = nn.Sequential(*context_layers)

        gamma_value = float(gamma_init)
        if mixer_mode != "legacy":
            gamma_value = _inverse_sigmoid(gamma_value / self.gamma_max, eps=self.eps)
        self.gamma = nn.Parameter(torch.tensor(gamma_value))

    def _weighted_prototype(self, feature, weight):
        denominator = weight.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return (feature * weight).sum(dim=(2, 3), keepdim=True) / denominator

    def set_prototype_scale(self, scale):
        self.prototype_scale = min(max(float(scale), 0.0), 1.0)

    def effective_gamma(self):
        if self.mixer_mode == "legacy":
            return self.gamma
        return self.gamma_max * torch.sigmoid(self.gamma)

    def forward(self, feature, use_prototype=True):
        coarse_logits = self.coarse_head(feature)
        coarse_prob = torch.sigmoid(coarse_logits)
        uncertainty = 4.0 * coarse_prob * (1.0 - coarse_prob)

        if not use_prototype:
            return feature, coarse_logits, uncertainty

        routing_prob = coarse_prob.detach() if self.mixer_mode == "stable" else coarse_prob
        routing_uncertainty = 4.0 * routing_prob * (1.0 - routing_prob)
        confidence = 1.0 - routing_uncertainty
        foreground_weight = confidence * routing_prob
        background_weight = confidence * (1.0 - routing_prob)

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
        gamma = self.effective_gamma()
        if self.mixer_mode == "stable":
            gamma = gamma * self.prototype_scale
        enhanced_feature = feature + gamma * self.context_proj(prototype_context)
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


class BoundaryAwareRefinement(nn.Module):
    def __init__(
        self,
        channels,
        edge_channels=1,
        gate_init=0.05,
        gate_max=0.5,
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
        energy = torch.sqrt(high_frequency.pow(2).mean(dim=1, keepdim=True) + self.eps)
        energy = torch.clamp_min(energy - self.eps ** 0.5, 0.0)
        energy = F.interpolate(
            energy,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        energy = torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
        return energy / energy.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)

    def forward(self, feature, seg_logits, uncertainty=None, hf_feature=None):
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

        hf_source = feature if hf_feature is None else hf_feature.detach()
        if hf_source.shape[-2:] != feature.shape[-2:]:
            hf_source = F.interpolate(
                hf_source,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        high_frequency = self.denoised_hf_energy(hf_source)
        if high_frequency.shape[-2:] != feature.shape[-2:]:
            high_frequency = F.interpolate(
                high_frequency,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
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


class CMUNeXt_HSPM_BARM(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        hspm_mode="full",
        hspm_mixer_mode="bounded",
        hspm_gamma_init=0.1,
        hspm_gamma_max=0.35,
        hspm_temperature=0.1,
        hspm_prototype_dropout=0.0,
        hspm_backbone_mode="dual_path",
        hspm_fusion_gate_init=0.05,
        hspm_fusion_gate_max=0.3,
        hspm_fusion_mode="global",
        hspm_small_area_threshold=0.05,
        hspm_small_area_temperature=0.02,
        barm_gate_init=0.05,
        barm_gate_max=0.5,
        barm_hf_keep_init=0.3,
        barm_hf_bypass=False,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("CMUNeXt_HSPM_BARM supports binary segmentation only.")
        if hspm_mode not in {"full", "context_only"}:
            raise ValueError("hspm_mode must be either 'full' or 'context_only'.")
        if hspm_backbone_mode not in {"highres_only", "dual_path"}:
            raise ValueError("hspm_backbone_mode must be either 'highres_only' or 'dual_path'.")
        if hspm_fusion_mode not in {"global", "size_aware"}:
            raise ValueError("hspm_fusion_mode must be either 'global' or 'size_aware'.")
        if not 0.0 < hspm_fusion_gate_init < hspm_fusion_gate_max:
            raise ValueError("hspm_fusion_gate_init must be in (0, hspm_fusion_gate_max).")
        if not 0.0 < hspm_small_area_threshold < 1.0:
            raise ValueError("hspm_small_area_threshold must be in (0, 1).")
        if hspm_small_area_temperature <= 0.0:
            raise ValueError("hspm_small_area_temperature must be positive.")

        self.hspm_mode = hspm_mode
        self.hspm_backbone_mode = hspm_backbone_mode
        self.hspm_fusion_gate_max = float(hspm_fusion_gate_max)
        self.hspm_fusion_mode = hspm_fusion_mode
        self.hspm_small_area_threshold = float(hspm_small_area_threshold)
        self.hspm_small_area_temperature = float(hspm_small_area_temperature)
        self.barm_hf_bypass = bool(barm_hf_bypass)
        self.last_fusion_diagnostics = None
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem = ConvBlock(input_channel, dims[0])
        self.encoder1 = CMUNeXtBlock(dims[0], dims[0], depths[0], kernels[0])
        self.encoder2 = CMUNeXtBlock(dims[0], dims[1], depths[1], kernels[1])
        self.encoder3 = CMUNeXtBlock(dims[1], dims[2], depths[2], kernels[2])
        self.encoder4 = CMUNeXtBlock(dims[2], dims[3], depths[3], kernels[3])

        self.high_resolution_context = HighResolutionContext(dims[3], dims[4])
        self.prototype_mixer = ConfidenceAwarePrototypeMixer(
            channels=dims[4],
            temperature=hspm_temperature,
            gamma_init=hspm_gamma_init,
            gamma_max=hspm_gamma_max,
            dropout=hspm_prototype_dropout,
            mixer_mode=hspm_mixer_mode,
        )

        if self.hspm_backbone_mode == "dual_path":
            self.encoder5 = CMUNeXtBlock(dims[3], dims[4], depths[4], kernels[4])
            self.Up5 = UpConv(dims[4], dims[3])
            self.Up_conv5 = FusionConv(dims[3] * 2, dims[3])
            self.hspm_projection = nn.Sequential(
                nn.Conv2d(dims[4], dims[3], kernel_size=1, bias=False),
                nn.BatchNorm2d(dims[3]),
            )
            gate_ratio = float(hspm_fusion_gate_init) / self.hspm_fusion_gate_max
            self.fusion_gate_raw = nn.Parameter(
                torch.tensor(_inverse_sigmoid(gate_ratio), dtype=torch.float32)
            )
            up4_input_channels = dims[3]
        else:
            up4_input_channels = dims[4]

        self.Up4 = UpConv(up4_input_channels, dims[2])
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

    def effective_fusion_gate(self):
        if self.hspm_backbone_mode != "dual_path":
            return None
        return self.hspm_fusion_gate_max * torch.sigmoid(self.fusion_gate_raw)

    def build_spatial_gate(self, coarse_logits, uncertainty):
        coarse_prob = torch.sigmoid(coarse_logits).detach()
        uncertainty = uncertainty.detach()
        predicted_area = coarse_prob.mean(dim=(2, 3), keepdim=True)
        smallness = torch.sigmoid(
            (self.hspm_small_area_threshold - predicted_area)
            / self.hspm_small_area_temperature
        )
        spatial_gate = smallness * coarse_prob + (1.0 - smallness) * uncertainty
        return spatial_gate, smallness, predicted_area

    def _record_fusion_diagnostics(
        self,
        effective_gate,
        deep_feature,
        injection,
        spatial_gate=None,
        smallness=None,
        predicted_area=None,
    ):
        eps = torch.finfo(deep_feature.dtype).eps
        injection_rms = injection.detach().square().mean(dim=(1, 2, 3)).sqrt()
        deep_feature_rms = deep_feature.detach().square().mean(dim=(1, 2, 3)).sqrt().clamp_min(eps)
        injection_deep_rms_ratio = injection_rms / deep_feature_rms
        diagnostics = {
            "effective_fusion_gate": effective_gate.detach().mean(),
            "injection_deep_rms_ratio": injection_deep_rms_ratio.mean(),
        }
        if spatial_gate is not None:
            sample_area = predicted_area.detach().flatten()
            small_mask = (sample_area < self.hspm_small_area_threshold).to(injection_deep_rms_ratio.dtype)
            large_mask = (sample_area > 0.2).to(injection_deep_rms_ratio.dtype)
            diagnostics.update(
                {
                    "predicted_area": sample_area.mean(),
                    "smallness": smallness.detach().mean(),
                    "spatial_gate": spatial_gate.detach().mean(),
                    "small_injection_deep_rms_ratio_sum": (
                        injection_deep_rms_ratio * small_mask
                    ).sum(),
                    "small_injection_count": small_mask.sum(),
                    "large_injection_deep_rms_ratio_sum": (
                        injection_deep_rms_ratio * large_mask
                    ).sum(),
                    "large_injection_count": large_mask.sum(),
                }
            )
        self.last_fusion_diagnostics = diagnostics

    def set_warmup_scale(self, residual_scale=1.0):
        self.barm.set_residual_scale(residual_scale)

    def get_barm_diagnostics(self):
        return self.barm.last_diagnostics

    def forward(self, x):
        self.last_fusion_diagnostics = None
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))

        hspm_feature = self.high_resolution_context(x4)
        hspm_feature, coarse_logits, uncertainty = self.prototype_mixer(
            hspm_feature,
            use_prototype=self.hspm_mode == "full",
        )

        if self.hspm_backbone_mode == "dual_path":
            x5 = self.encoder5(self.Maxpool(x4))
            deep_feature = self.Up5(x5)
            deep_feature = self.Up_conv5(torch.cat((x4, deep_feature), dim=1))
            hspm_residual = self.hspm_projection(hspm_feature)
            effective_gate = self.effective_fusion_gate()
            if self.hspm_fusion_mode == "size_aware":
                spatial_gate, smallness, predicted_area = self.build_spatial_gate(
                    coarse_logits,
                    uncertainty,
                )
                injection = effective_gate * spatial_gate * hspm_residual
                self._record_fusion_diagnostics(
                    effective_gate,
                    deep_feature,
                    injection,
                    spatial_gate=spatial_gate,
                    smallness=smallness,
                    predicted_area=predicted_area,
                )
            else:
                injection = effective_gate * hspm_residual
                self._record_fusion_diagnostics(effective_gate, deep_feature, injection)
            decoder_input = deep_feature + injection
        else:
            decoder_input = hspm_feature

        d4 = self.Up4(decoder_input)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))
        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        base_seg = self.Conv_1x1(d2)
        seg, edge, band, logit_correction = self.barm(
            d2,
            base_seg,
            uncertainty=uncertainty,
            hf_feature=x1 if self.barm_hf_bypass else None,
        )
        return {
            "seg": seg,
            "base_seg": base_seg,
            "coarse": coarse_logits,
            "uncertainty": uncertainty,
            "edge": edge,
            "band": band,
            "logit_correction": logit_correction,
        }


def cmunext_hspm_barm(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    hspm_mode="full",
    hspm_mixer_mode="bounded",
    hspm_gamma_init=0.1,
    hspm_gamma_max=0.35,
    hspm_temperature=0.1,
    hspm_prototype_dropout=0.0,
    hspm_backbone_mode="dual_path",
    hspm_fusion_gate_init=0.05,
    hspm_fusion_gate_max=0.3,
    hspm_fusion_mode="global",
    hspm_small_area_threshold=0.05,
    hspm_small_area_temperature=0.02,
    barm_gate_init=0.05,
    barm_gate_max=0.5,
    barm_hf_keep_init=0.3,
    barm_hf_bypass=False,
):
    return CMUNeXt_HSPM_BARM(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        hspm_mode=hspm_mode,
        hspm_mixer_mode=hspm_mixer_mode,
        hspm_gamma_init=hspm_gamma_init,
        hspm_gamma_max=hspm_gamma_max,
        hspm_temperature=hspm_temperature,
        hspm_prototype_dropout=hspm_prototype_dropout,
        hspm_backbone_mode=hspm_backbone_mode,
        hspm_fusion_gate_init=hspm_fusion_gate_init,
        hspm_fusion_gate_max=hspm_fusion_gate_max,
        hspm_fusion_mode=hspm_fusion_mode,
        hspm_small_area_threshold=hspm_small_area_threshold,
        hspm_small_area_temperature=hspm_small_area_temperature,
        barm_gate_init=barm_gate_init,
        barm_gate_max=barm_gate_max,
        barm_hf_keep_init=barm_hf_keep_init,
        barm_hf_bypass=barm_hf_bypass,
    )


def cmunext_hspm_barm_hfbypass(**kwargs):
    kwargs["barm_hf_bypass"] = True
    return cmunext_hspm_barm(**kwargs)
