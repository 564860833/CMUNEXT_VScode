import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.conv_based.CMUNeXt import (
    CMUNeXtBlock,
    conv_block,
    fusion_conv,
    up_conv,
)
from src.network.conv_based.CMUNeXt_HSPM import (
    ConfidenceAwarePrototypeMixer,
    HighResolutionContext,
    _inverse_sigmoid,
)


class CurrentBaselineProgressiveRefinement(nn.Module):
    def __init__(
        self,
        channels,
        gate_kernel,
        mode="full",
        feature_scale_init=0.05,
        feature_scale_max=0.3,
    ):
        super().__init__()
        if mode not in {"full", "no_ambiguity", "no_detail"}:
            raise ValueError("mode must be 'full', 'no_ambiguity', or 'no_detail'.")
        if gate_kernel <= 0 or gate_kernel % 2 == 0:
            raise ValueError("gate_kernel must be a positive odd integer.")
        if not 0.0 < feature_scale_init < feature_scale_max:
            raise ValueError("feature_scale_init must be in (0, feature_scale_max).")

        self.mode = mode
        self.gate_kernel = int(gate_kernel)
        self.feature_scale_max = float(feature_scale_max)
        self.route_scale = 1.0

        self.last_previous_probability = None
        self.last_base_probability = None
        self.last_raw_gate = None
        self.last_active_gate = None
        self.last_ambiguity_gate = None
        self.last_oracle_recovery_target = None
        self.last_oracle_recovery_gate = None
        self.last_combined_gate = None
        self.last_recovery_added = None
        self.last_pre_base_logits = None
        self.last_feature_refined_logits = None
        self.last_correction_logits = None
        self.last_refined_logits = None

        self.detail_extractor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
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
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2 + 3, channels, kernel_size=1, bias=False),
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
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.base_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.correction_head = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

        feature_ratio = float(feature_scale_init) / self.feature_scale_max
        self.feature_scale_raw = nn.Parameter(
            torch.tensor(_inverse_sigmoid(feature_ratio), dtype=torch.float32)
        )

    def effective_feature_scale(self):
        return self.feature_scale_max * torch.sigmoid(self.feature_scale_raw)

    def set_route_scale(self, scale):
        self.route_scale = min(max(float(scale), 0.0), 1.0)

    def build_route(self, base_probability, oracle_target=None):
        base_probability = base_probability.detach()
        if self.mode == "no_ambiguity":
            ambiguity_gate = torch.ones_like(base_probability)
        else:
            ambiguity = 4.0 * base_probability * (1.0 - base_probability)
            ambiguity_gate = F.max_pool2d(
                ambiguity,
                kernel_size=self.gate_kernel,
                stride=1,
                padding=self.gate_kernel // 2,
            )
        ambiguity_gate = ambiguity_gate.detach()

        oracle_recovery_target = torch.zeros_like(base_probability)
        oracle_recovery_gate = torch.zeros_like(base_probability)
        if oracle_target is not None:
            resized_target = F.interpolate(
                oracle_target.float(),
                size=base_probability.shape[-2:],
                mode="nearest",
            ).detach()
            confidence = 2.0 * torch.abs(base_probability - 0.5)
            oracle_recovery_target = (
                torch.abs(resized_target - base_probability) * confidence
            ).detach()
            oracle_recovery_gate = F.max_pool2d(
                oracle_recovery_target,
                kernel_size=self.gate_kernel,
                stride=1,
                padding=self.gate_kernel // 2,
            ).detach()

        combined_gate = torch.maximum(
            ambiguity_gate,
            oracle_recovery_gate,
        ).detach()
        self.last_ambiguity_gate = ambiguity_gate
        self.last_oracle_recovery_target = oracle_recovery_target
        self.last_oracle_recovery_gate = oracle_recovery_gate
        self.last_combined_gate = combined_gate
        self.last_recovery_added = (combined_gate - ambiguity_gate).clamp_min(0)
        return combined_gate

    def route_diagnostics(self):
        if self.last_raw_gate is None or self.last_active_gate is None:
            return None
        raw_gate = self.last_raw_gate.detach()
        active_gate = self.last_active_gate.detach()
        ambiguity_gate = self.last_ambiguity_gate.detach()
        oracle_recovery_target = self.last_oracle_recovery_target.detach()
        oracle_recovery_gate = self.last_oracle_recovery_gate.detach()
        combined_gate = self.last_combined_gate.detach()
        recovery_added = self.last_recovery_added.detach()
        correction_abs = self.last_correction_logits.detach().abs()
        feature_delta_abs = (
            self.last_feature_refined_logits.detach() - self.last_pre_base_logits.detach()
        ).abs()
        total_delta_abs = (
            self.last_refined_logits.detach() - self.last_pre_base_logits.detach()
        ).abs()
        return {
            "route_scale": raw_gate.new_tensor(self.route_scale),
            "raw_gate_mean": raw_gate.mean(),
            "active_gate_mean": active_gate.mean(),
            "raw_gate_over_05": (raw_gate > 0.5).to(raw_gate.dtype).mean(),
            "raw_gate_over_08": (raw_gate > 0.8).to(raw_gate.dtype).mean(),
            "ambiguity_gate_mean": ambiguity_gate.mean(),
            "oracle_recovery_target_mean": oracle_recovery_target.mean(),
            "oracle_recovery_gate_mean": oracle_recovery_gate.mean(),
            "combined_gate_mean": combined_gate.mean(),
            "recovery_added_mean": recovery_added.mean(),
            "recovery_dominant_ratio": (
                oracle_recovery_gate > ambiguity_gate
            ).to(raw_gate.dtype).mean(),
            "base_probability_mean": self.last_base_probability.detach().mean(),
            "feature_logit_delta_abs_mean": feature_delta_abs.mean(),
            "correction_logit_abs_mean": correction_abs.mean(),
            "correction_logit_abs_p99": torch.quantile(correction_abs.flatten(), 0.99),
            "correction_logit_abs_max": correction_abs.max(),
            "total_logit_delta_abs_mean": total_delta_abs.mean(),
            "effective_feature_scale": self.effective_feature_scale().detach(),
        }

    def forward(
        self,
        decoder_feature,
        skip_feature,
        previous_logits,
        oracle_target=None,
    ):
        target_size = decoder_feature.shape[-2:]
        if skip_feature.shape[-2:] != target_size:
            skip_feature = F.interpolate(
                skip_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        previous_logits = F.interpolate(
            previous_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        previous_probability = torch.sigmoid(previous_logits).detach()
        pre_base_logits = self.base_head(decoder_feature)
        base_probability = torch.sigmoid(pre_base_logits).detach()

        raw_gate = self.build_route(
            base_probability,
            oracle_target=oracle_target,
        )
        active_gate = self.route_scale * raw_gate

        if self.mode == "no_detail":
            detail = torch.zeros_like(decoder_feature)
        else:
            detail = active_gate * self.detail_extractor(skip_feature)

        refinement = self.refine(
            torch.cat(
                [
                    decoder_feature,
                    detail,
                    previous_probability,
                    base_probability,
                    active_gate,
                ],
                dim=1,
            )
        )
        refined_feature = (
            decoder_feature
            + self.effective_feature_scale() * active_gate * refinement
        )
        feature_refined_logits = self.base_head(refined_feature)
        correction_logits = self.correction_head(refinement)
        refined_logits = feature_refined_logits + active_gate * correction_logits

        self.last_previous_probability = previous_probability
        self.last_base_probability = base_probability
        self.last_raw_gate = raw_gate
        self.last_active_gate = active_gate
        self.last_pre_base_logits = pre_base_logits
        self.last_feature_refined_logits = feature_refined_logits
        self.last_correction_logits = correction_logits
        self.last_refined_logits = refined_logits
        return refined_feature, refined_logits


class CMUNeXt_HSPM_APBR_V2(nn.Module):
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
        hspm_fusion_gate_init=0.05,
        hspm_fusion_gate_max=0.3,
        hspm_fusion_mode="global",
        hspm_small_area_threshold=0.05,
        hspm_small_area_temperature=0.02,
        apbr_mode="full",
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("CMUNeXt_HSPM_APBR_V2 supports binary segmentation only; num_classes must be 1.")
        if hspm_mode not in {"full", "context_only"}:
            raise ValueError("hspm_mode must be either 'full' or 'context_only'.")
        if hspm_fusion_mode not in {"global", "size_aware"}:
            raise ValueError("hspm_fusion_mode must be either 'global' or 'size_aware'.")
        if not 0.0 < hspm_fusion_gate_init < hspm_fusion_gate_max:
            raise ValueError("hspm_fusion_gate_init must be in (0, hspm_fusion_gate_max).")
        if not 0.0 < hspm_small_area_threshold < 1.0:
            raise ValueError("hspm_small_area_threshold must be in (0, 1).")
        if hspm_small_area_temperature <= 0.0:
            raise ValueError("hspm_small_area_temperature must be positive.")

        self.hspm_mode = hspm_mode
        self.hspm_fusion_gate_max = float(hspm_fusion_gate_max)
        self.hspm_fusion_mode = hspm_fusion_mode
        self.hspm_small_area_threshold = float(hspm_small_area_threshold)
        self.hspm_small_area_temperature = float(hspm_small_area_temperature)
        self.apbr_mode = apbr_mode
        self.last_fusion_diagnostics = None
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        self.high_resolution_context = HighResolutionContext(ch_in=dims[3], ch_out=dims[4])
        self.prototype_mixer = ConfidenceAwarePrototypeMixer(
            channels=dims[4],
            temperature=hspm_temperature,
            gamma_init=hspm_gamma_init,
            gamma_max=hspm_gamma_max,
            dropout=hspm_prototype_dropout,
            mixer_mode=hspm_mixer_mode,
        )
        self.hspm_projection = nn.Sequential(
            nn.Conv2d(dims[4], dims[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[3]),
        )
        gate_ratio = float(hspm_fusion_gate_init) / self.hspm_fusion_gate_max
        self.fusion_gate_raw = nn.Parameter(
            torch.tensor(_inverse_sigmoid(gate_ratio), dtype=torch.float32)
        )

        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])

        self.apbr_half = CurrentBaselineProgressiveRefinement(
            channels=dims[1],
            gate_kernel=5,
            mode=apbr_mode,
        )
        self.apbr_full = CurrentBaselineProgressiveRefinement(
            channels=dims[0],
            gate_kernel=3,
            mode=apbr_mode,
        )

    def effective_fusion_gate(self):
        return self.hspm_fusion_gate_max * torch.sigmoid(self.fusion_gate_raw)

    def set_apbr_route_scale(self, scale):
        self.apbr_half.set_route_scale(scale)
        self.apbr_full.set_route_scale(scale)

    def get_apbr_diagnostics(self):
        diagnostics = {}
        for stage_name, module in (
            ("half", self.apbr_half),
            ("full", self.apbr_full),
        ):
            stage_diagnostics = module.route_diagnostics()
            if stage_diagnostics is not None:
                diagnostics.update(
                    {
                        f"{stage_name}_{name}": value
                        for name, value in stage_diagnostics.items()
                    }
                )
        return diagnostics or None

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

    def _forward_impl(self, x, oracle_target=None, oracle_stages=frozenset()):
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

        d4 = self.Up4(deep_feature + injection)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))
        d3, refine_half = self.apbr_half(
            d3,
            x2,
            coarse_logits,
            oracle_target=oracle_target if "half" in oracle_stages else None,
        )

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))
        _, final_logits = self.apbr_full(
            d2,
            x1,
            refine_half,
            oracle_target=oracle_target if "full" in oracle_stages else None,
        )

        return {
            "seg": final_logits,
            "coarse": coarse_logits,
            "uncertainty": uncertainty,
            "refine_half": refine_half,
        }

    def forward(self, x):
        return self._forward_impl(x)

    def forward_oracle_recovery(
        self,
        x,
        target,
        oracle_stages=frozenset({"half", "full"}),
    ):
        oracle_stages = frozenset(oracle_stages)
        unsupported_stages = oracle_stages.difference({"half", "full"})
        if unsupported_stages:
            raise ValueError(
                f"Unsupported Oracle Recovery stages: {sorted(unsupported_stages)}."
            )
        if target is None:
            raise ValueError("target is required for Oracle Recovery.")
        if target.ndim != 4 or target.shape[1] != 1:
            raise ValueError("Oracle Recovery target must have shape [N, 1, H, W].")
        if target.shape[0] != x.shape[0]:
            raise ValueError("Oracle Recovery target batch size must match the input.")
        return self._forward_impl(
            x,
            oracle_target=target,
            oracle_stages=oracle_stages,
        )


def cmunext_hspm_apbr_v2(
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
    hspm_fusion_gate_init=0.05,
    hspm_fusion_gate_max=0.3,
    hspm_fusion_mode="global",
    hspm_small_area_threshold=0.05,
    hspm_small_area_temperature=0.02,
    apbr_mode="full",
):
    return CMUNeXt_HSPM_APBR_V2(
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
        hspm_fusion_gate_init=hspm_fusion_gate_init,
        hspm_fusion_gate_max=hspm_fusion_gate_max,
        hspm_fusion_mode=hspm_fusion_mode,
        hspm_small_area_threshold=hspm_small_area_threshold,
        hspm_small_area_temperature=hspm_small_area_temperature,
        apbr_mode=apbr_mode,
    )
