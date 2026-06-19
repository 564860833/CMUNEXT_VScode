import torch
import torch.nn as nn

from src.network.conv_based.CMUNeXt_HSPM import CMUNeXt_HSPM, _inverse_sigmoid
from src.network.conv_based.CMUNeXt_HSPM_FBDM import FBDM


class SafeBoundaryLogitCorrection(nn.Module):
    """Boundary-limited logit correction with no gradient path into HSPM features."""

    def __init__(
        self,
        channels,
        logit_channels=1,
        correction_scale_init=0.05,
        correction_scale_max=0.20,
        boundary_gate_floor=0.20,
    ):
        super().__init__()
        if channels <= 0 or logit_channels <= 0:
            raise ValueError("channels and logit_channels must be positive.")
        if not 0.0 < correction_scale_init < correction_scale_max:
            raise ValueError(
                "correction_scale_init must be in (0, correction_scale_max)."
            )
        if not 0.0 <= boundary_gate_floor < 1.0:
            raise ValueError("boundary_gate_floor must be in [0, 1).")

        self.correction_scale_max = float(correction_scale_max)
        self.boundary_gate_floor = float(boundary_gate_floor)
        self.schedule_scale = 1.0
        self.last_boundary_gate = None
        self.last_base_logits = None
        self.last_boundary_feature = None
        self.last_safe_boundary_gate = None
        self.last_raw_correction = None
        self.last_logit_correction = None
        self.last_final_logits = None

        correction_in_channels = 2 * channels + 2 * logit_channels + 1
        norm_groups = min(4, channels)
        while channels % norm_groups != 0:
            norm_groups -= 1
        self.correction_features = nn.Sequential(
            nn.Conv2d(correction_in_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(norm_groups, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(norm_groups, channels),
            nn.GELU(),
        )
        self.correction_head = nn.Conv2d(channels, logit_channels, kernel_size=1)
        nn.init.normal_(self.correction_head.weight, mean=0.0, std=1e-3)
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
        boundary_feature = self.last_boundary_feature.detach()
        safe_boundary_gate = self.last_safe_boundary_gate.detach()
        raw_correction = self.last_raw_correction.detach()
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
            "safe_boundary_gate_mean": safe_boundary_gate.mean(),
            "boundary_feature_rms": boundary_feature.square().mean().sqrt(),
            "raw_correction_abs_mean": raw_correction.abs().mean(),
            "raw_correction_abs_max": raw_correction.abs().max(),
            "logit_correction_abs_mean": logit_correction.abs().mean(),
            "logit_correction_abs_max": logit_correction.abs().max(),
            "prediction_flip_ratio": (
                base_prediction != final_prediction
            ).to(boundary_gate.dtype).mean(),
        }

    def forward(
        self,
        feature,
        boundary_feature,
        base_logits,
        edge_logits,
        boundary_gate,
    ):
        detached_feature = feature.detach()
        base_probability = torch.sigmoid(base_logits.detach())
        edge_probability = torch.sigmoid(edge_logits)
        detached_boundary_gate = boundary_gate.detach()
        safe_boundary_gate = (
            self.boundary_gate_floor
            + (1.0 - self.boundary_gate_floor) * detached_boundary_gate
        )
        raw_correction = self.correction_head(
            self.correction_features(
                torch.cat(
                    [
                        detached_feature,
                        boundary_feature,
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
            * safe_boundary_gate
            * self.effective_correction_scale()
            * torch.tanh(raw_correction)
        )
        final_logits = base_logits + logit_correction

        self.last_boundary_gate = detached_boundary_gate
        self.last_boundary_feature = boundary_feature
        self.last_safe_boundary_gate = safe_boundary_gate
        self.last_base_logits = base_logits
        self.last_raw_correction = raw_correction
        self.last_logit_correction = logit_correction
        self.last_final_logits = final_logits
        return final_logits, logit_correction


class CMUNeXt_HSPM_FBDM_Best0616(CMUNeXt_HSPM):
    """Bounded HSPM with an independently trained x1 boundary correction branch."""

    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        hspm_gamma_init=0.1,
        hspm_gamma_max=0.35,
        hspm_fusion_gate_init=0.05,
        hspm_fusion_gate_max=0.3,
        fbdm_semantic_uncertainty_weight=0.7,
        fbdm_semantic_coarse_weight=0.3,
        fbdm_semantic_gate_base=0.7,
        fbdm_gate_init=0.01,
        fbdm_gate_max=0.06,
        fbdm_correction_scale_init=0.05,
        fbdm_correction_scale_max=0.20,
        fbdm_boundary_gate_floor=0.20,
    ):
        super().__init__(
            input_channel=input_channel,
            num_classes=num_classes,
            dims=dims,
            depths=depths,
            kernels=kernels,
            hspm_mode="full",
            hspm_mixer_mode="bounded",
            hspm_gamma_init=hspm_gamma_init,
            hspm_gamma_max=hspm_gamma_max,
            hspm_temperature=0.1,
            hspm_prototype_dropout=0.0,
            hspm_backbone_mode="dual_path",
            hspm_fusion_gate_init=hspm_fusion_gate_init,
            hspm_fusion_gate_max=hspm_fusion_gate_max,
            hspm_fusion_mode="global",
            hspm_small_area_threshold=0.05,
            hspm_small_area_temperature=0.02,
        )

        # Preserve the RNG state that a standalone HSPM leaves for data shuffling.
        with torch.random.fork_rng(devices=[]):
            self.fbdm1 = FBDM(
                channels=dims[0],
                edge_channels=num_classes,
                use_hspm_prior=True,
                detach_hspm_prior=True,
                semantic_uncertainty_weight=fbdm_semantic_uncertainty_weight,
                semantic_coarse_weight=fbdm_semantic_coarse_weight,
                semantic_gate_base=fbdm_semantic_gate_base,
                gate_init=fbdm_gate_init,
                gate_max=fbdm_gate_max,
                edge_aux_only=True,
            )
            self.fbdm_correction = SafeBoundaryLogitCorrection(
                channels=dims[0],
                logit_channels=num_classes,
                correction_scale_init=fbdm_correction_scale_init,
                correction_scale_max=fbdm_correction_scale_max,
                boundary_gate_floor=fbdm_boundary_gate_floor,
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
            use_prototype=True,
        )

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
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        # FBDM sees HSPM features and priors but cannot backpropagate into them.
        _, edge_logits = self.fbdm1(
            x1.detach(),
            coarse_logits=coarse_logits.detach(),
            uncertainty=uncertainty.detach(),
        )
        boundary_gate = self.fbdm1.last_boundary_gate
        boundary_feature = self.fbdm1.last_boundary_feature
        if boundary_gate is None:
            raise RuntimeError("FBDM boundary gate was not recorded before correction.")
        if boundary_feature is None:
            raise RuntimeError("FBDM boundary feature was not recorded before correction.")

        base_logits = self.Conv_1x1(d2)
        final_logits, logit_correction = self.fbdm_correction(
            d2,
            boundary_feature,
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
            "boundary_gate": boundary_gate.detach(),
            "logit_correction": logit_correction,
        }


def cmunext_hspm_fbdm_best0616(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    **kwargs,
):
    return CMUNeXt_HSPM_FBDM_Best0616(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        **kwargs,
    )
