import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.conv_based.CMUNeXt_HSPM import CMUNeXt_HSPM, _inverse_sigmoid


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
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]]
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0],
              [0.0, 0.0, 0.0],
              [1.0, 2.0, 1.0]]]
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
        use_hspm_prior=True,
        detach_hspm_prior=True,
        semantic_uncertainty_weight=0.7,
        semantic_coarse_weight=0.3,
        semantic_coarse_mode="prob",
        semantic_gate_base=0.7,
        gate_init=0.03,
        gate_max=0.2,
        edge_aux_only=False,
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
        if semantic_uncertainty_weight < 0.0 or semantic_coarse_weight < 0.0:
            raise ValueError("semantic prior weights must be non-negative.")
        if semantic_coarse_mode not in {"prob", "edge"}:
            raise ValueError("semantic_coarse_mode must be either 'prob' or 'edge'.")
        if not 0.0 <= semantic_gate_base <= 1.0:
            raise ValueError("semantic_gate_base must be in [0, 1].")

        self.gate_max = float(gate_max)
        self.use_hspm_prior = bool(use_hspm_prior)
        self.detach_hspm_prior = bool(detach_hspm_prior)
        self.semantic_uncertainty_weight = float(semantic_uncertainty_weight)
        self.semantic_coarse_weight = float(semantic_coarse_weight)
        self.semantic_coarse_mode = semantic_coarse_mode
        self.semantic_gate_base = float(semantic_gate_base)
        self.edge_aux_only = bool(edge_aux_only)
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
            nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False),
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
        self.last_boundary_feature = None
        self.last_edge_logits = None

    def effective_gamma(self):
        return self.gate_max * torch.sigmoid(self.gamma_raw) * self.residual_scale

    def set_residual_scale(self, scale):
        self.residual_scale = min(max(float(scale), 0.0), 1.0)

    def build_boundary_prior(self, x, coarse_logits=None, uncertainty=None):
        edge_prior = self.sobel_edge(x)
        if not self.use_hspm_prior:
            semantic_prior = torch.ones_like(edge_prior)
            return edge_prior, semantic_prior, edge_prior

        if coarse_logits is None or uncertainty is None:
            raise ValueError("FBDM requires coarse_logits and uncertainty when use_hspm_prior=True.")

        coarse_prob = torch.sigmoid(coarse_logits)
        uncertainty_prior = uncertainty
        if self.detach_hspm_prior:
            coarse_prob = coarse_prob.detach()
            uncertainty_prior = uncertainty_prior.detach()

        target_size = x.shape[-2:]
        coarse_prob = F.interpolate(
            coarse_prob,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        uncertainty_prior = F.interpolate(
            uncertainty_prior,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        weight_sum = max(
            self.semantic_uncertainty_weight + self.semantic_coarse_weight,
            self.eps,
        )
        coarse_prior = coarse_prob
        if self.semantic_coarse_mode == "edge":
            coarse_prior = self.sobel_edge(coarse_prob)
        semantic_prior = (
            self.semantic_uncertainty_weight / weight_sum * uncertainty_prior
            + self.semantic_coarse_weight / weight_sum * coarse_prior
        ).clamp(0.0, 1.0)
        boundary_gate = edge_prior * (
            self.semantic_gate_base
            + (1.0 - self.semantic_gate_base) * semantic_prior
        )
        return edge_prior, semantic_prior, boundary_gate

    def forward(self, x, coarse_logits=None, uncertainty=None):
        low_feature = self.low_branch(x)
        _, lh, hl, hh = self.dwt(x)
        high_feature = self.high_branch(torch.cat((lh, hl, hh), dim=1))
        high_feature = F.interpolate(
            high_feature,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        _, semantic_prior, boundary_gate = self.build_boundary_prior(
            x,
            coarse_logits=coarse_logits,
            uncertainty=uncertainty,
        )
        edge_feature = self.edge_branch(boundary_gate)

        residual = self.fusion(torch.cat((low_feature, high_feature, edge_feature), dim=1))
        residual_mean = residual.mean(dim=1, keepdim=True)
        gate_input = torch.cat((residual_mean, boundary_gate, semantic_prior), dim=1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        out = x + self.effective_gamma() * gate * residual
        edge_logits = self.edge_head(out)
        self.last_boundary_gate = boundary_gate
        self.last_boundary_feature = residual
        self.last_edge_logits = edge_logits
        main_feature = x if self.edge_aux_only else out
        return main_feature, edge_logits


class FBDMLogitCorrection(nn.Module):
    def __init__(
        self,
        channels,
        logit_channels=1,
        correction_scale_init=0.05,
        correction_scale_max=0.3,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if logit_channels <= 0:
            raise ValueError("logit_channels must be positive.")
        if not 0.0 < correction_scale_init < correction_scale_max:
            raise ValueError(
                "correction_scale_init must be in (0, correction_scale_max)."
            )

        self.correction_scale_max = float(correction_scale_max)
        self.schedule_scale = 1.0
        self.last_boundary_gate = None
        self.last_base_logits = None
        self.last_raw_correction = None
        self.last_logit_correction = None
        self.last_final_logits = None

        correction_in_channels = channels + 2 * logit_channels + 1
        self.correction_features = nn.Sequential(
            nn.Conv2d(correction_in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
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
        )
        self.correction_head = nn.Conv2d(channels, logit_channels, kernel_size=1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

        correction_ratio = float(correction_scale_init) / self.correction_scale_max
        self.correction_scale_raw = nn.Parameter(
            torch.tensor(_inverse_sigmoid(correction_ratio), dtype=torch.float32)
        )

    def effective_correction_scale(self):
        return self.correction_scale_max * torch.sigmoid(self.correction_scale_raw)

    def set_schedule_scale(self, scale):
        self.schedule_scale = min(max(float(scale), 0.0), 1.0)

    def diagnostics(self):
        if self.last_boundary_gate is None:
            return None
        boundary_gate = self.last_boundary_gate.detach()
        logit_correction = self.last_logit_correction.detach()
        base_prediction = self.last_base_logits.detach() > 0
        final_prediction = self.last_final_logits.detach() > 0
        return {
            "correction_schedule_scale": boundary_gate.new_tensor(self.schedule_scale),
            "effective_correction_scale": self.effective_correction_scale().detach(),
            "boundary_gate_mean": boundary_gate.mean(),
            "boundary_gate_over_05": (
                boundary_gate > 0.5
            ).to(boundary_gate.dtype).mean(),
            "logit_correction_abs_mean": logit_correction.abs().mean(),
            "logit_correction_abs_max": logit_correction.abs().max(),
            "prediction_flip_ratio": (
                base_prediction != final_prediction
            ).to(boundary_gate.dtype).mean(),
        }

    def forward(self, feature, base_logits, edge_logits, boundary_gate):
        detached_feature = feature.detach()
        base_probability = torch.sigmoid(base_logits.detach())
        edge_probability = torch.sigmoid(edge_logits)
        detached_boundary_gate = boundary_gate.detach()
        raw_correction = self.correction_head(
            self.correction_features(
                torch.cat(
                    [
                        detached_feature,
                        base_probability,
                        edge_probability,
                        detached_boundary_gate,
                    ],
                    dim=1,
                )
            )
        )
        logit_correction = (
            self.schedule_scale
            * detached_boundary_gate
            * self.effective_correction_scale()
            * torch.tanh(raw_correction)
        )
        final_logits = base_logits + logit_correction

        self.last_boundary_gate = detached_boundary_gate
        self.last_base_logits = base_logits
        self.last_raw_correction = raw_correction
        self.last_logit_correction = logit_correction
        self.last_final_logits = final_logits
        return final_logits, logit_correction


class CMUNeXt_HSPM_FBDM(CMUNeXt_HSPM):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        hspm_mode="full",
        hspm_mixer_mode="legacy",
        hspm_gamma_init=0.1,
        hspm_gamma_max=0.3,
        hspm_temperature=0.1,
        hspm_prototype_dropout=0.0,
        hspm_backbone_mode="highres_only",
        hspm_fusion_gate_init=0.05,
        hspm_fusion_gate_max=0.3,
        hspm_fusion_mode="global",
        hspm_small_area_threshold=0.05,
        hspm_small_area_temperature=0.02,
        fbdm_use_hspm_prior=True,
        fbdm_detach_hspm_prior=True,
        fbdm_semantic_uncertainty_weight=0.7,
        fbdm_semantic_coarse_weight=0.3,
        fbdm_semantic_coarse_mode="prob",
        fbdm_semantic_gate_base=0.7,
        fbdm_gate_init=0.03,
        fbdm_gate_max=0.2,
        fbdm_edge_aux_only=False,
    ):
        super().__init__(
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
        )
        self.fbdm1 = FBDM(
            channels=dims[0],
            edge_channels=num_classes,
            use_hspm_prior=fbdm_use_hspm_prior,
            detach_hspm_prior=fbdm_detach_hspm_prior,
            semantic_uncertainty_weight=fbdm_semantic_uncertainty_weight,
            semantic_coarse_weight=fbdm_semantic_coarse_weight,
            semantic_coarse_mode=fbdm_semantic_coarse_mode,
            semantic_gate_base=fbdm_semantic_gate_base,
            gate_init=fbdm_gate_init,
            gate_max=fbdm_gate_max,
            edge_aux_only=fbdm_edge_aux_only,
        )

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

        x1_fbdm, edge_logits = self.fbdm1(
            x1,
            coarse_logits=coarse_logits,
            uncertainty=uncertainty,
        )
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1_fbdm, d2), dim=1))

        return {
            "seg": self.Conv_1x1(d2),
            "coarse": coarse_logits,
            "uncertainty": uncertainty,
            "edge": edge_logits,
        }


class CMUNeXt_HSPM_FBDM_V2(CMUNeXt_HSPM_FBDM):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        hspm_mode="full",
        hspm_mixer_mode="legacy",
        hspm_gamma_init=0.1,
        hspm_gamma_max=0.3,
        hspm_temperature=0.1,
        hspm_prototype_dropout=0.0,
        hspm_backbone_mode="highres_only",
        hspm_fusion_gate_init=0.05,
        hspm_fusion_gate_max=0.3,
        hspm_fusion_mode="global",
        hspm_small_area_threshold=0.05,
        hspm_small_area_temperature=0.02,
        fbdm_use_hspm_prior=True,
        fbdm_detach_hspm_prior=True,
        fbdm_semantic_uncertainty_weight=0.7,
        fbdm_semantic_coarse_weight=0.3,
        fbdm_semantic_gate_base=0.7,
        fbdm_gate_init=0.03,
        fbdm_gate_max=0.2,
        fbdm_correction_scale_init=0.05,
        fbdm_correction_scale_max=0.3,
    ):
        super().__init__(
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
            fbdm_use_hspm_prior=fbdm_use_hspm_prior,
            fbdm_detach_hspm_prior=fbdm_detach_hspm_prior,
            fbdm_semantic_uncertainty_weight=fbdm_semantic_uncertainty_weight,
            fbdm_semantic_coarse_weight=fbdm_semantic_coarse_weight,
            fbdm_semantic_gate_base=fbdm_semantic_gate_base,
            fbdm_gate_init=fbdm_gate_init,
            fbdm_gate_max=fbdm_gate_max,
            fbdm_edge_aux_only=True,
        )
        self.fbdm_correction = FBDMLogitCorrection(
            channels=dims[0],
            logit_channels=num_classes,
            correction_scale_init=fbdm_correction_scale_init,
            correction_scale_max=fbdm_correction_scale_max,
        )

    def set_fbdm_correction_schedule_scale(self, scale):
        self.fbdm_correction.set_schedule_scale(scale)

    def get_fbdm_v2_diagnostics(self):
        return self.fbdm_correction.diagnostics()

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

        x1_fbdm, edge_logits = self.fbdm1(
            x1,
            coarse_logits=coarse_logits,
            uncertainty=uncertainty,
        )
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1_fbdm, d2), dim=1))

        base_logits = self.Conv_1x1(d2)
        boundary_gate = self.fbdm1.last_boundary_gate
        if boundary_gate is None:
            raise RuntimeError("FBDM boundary gate was not recorded before V2 correction.")
        final_logits, logit_correction = self.fbdm_correction(
            d2,
            base_logits,
            edge_logits,
            boundary_gate,
        )
        return {
            "seg": final_logits,
            "base_seg": base_logits,
            "coarse": coarse_logits,
            "uncertainty": uncertainty,
            "edge": edge_logits,
            "logit_correction": logit_correction,
        }


def cmunext_hspm_fbdm(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    **kwargs,
):
    return CMUNeXt_HSPM_FBDM(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        **kwargs,
    )


def cmunext_hspm_fbdm_v2(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    **kwargs,
):
    return CMUNeXt_HSPM_FBDM_V2(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        **kwargs,
    )


def cmunext_hspm_fbdm_s(
    input_channel=3,
    num_classes=1,
    dims=(8, 16, 32, 64, 128),
    depths=(1, 1, 1, 1, 1),
    kernels=(3, 3, 7, 7, 9),
    **kwargs,
):
    return CMUNeXt_HSPM_FBDM(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        **kwargs,
    )


def cmunext_hspm_fbdm_l(
    input_channel=3,
    num_classes=1,
    dims=(32, 64, 128, 256, 512),
    depths=(1, 1, 1, 6, 3),
    kernels=(3, 3, 7, 7, 7),
    **kwargs,
):
    return CMUNeXt_HSPM_FBDM(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        **kwargs,
    )
