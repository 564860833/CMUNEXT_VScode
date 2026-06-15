import torch
import torch.nn as nn

from src.network.conv_based.CMUNeXt_HSPM import _inverse_sigmoid
from src.network.conv_based.CMUNeXt_HSPM_SDFR import CMUNeXt_HSPM_SDFR


class SignedDistanceLogitCorrection(nn.Module):
    def __init__(
        self,
        channels,
        boundary_temperature=0.2,
        correction_scale_init=1.0,
        correction_scale_max=3.0,
    ):
        super().__init__()
        if boundary_temperature <= 0:
            raise ValueError("boundary_temperature must be positive.")
        if not 0.0 < correction_scale_init < correction_scale_max:
            raise ValueError(
                "correction_scale_init must be in (0, correction_scale_max)."
            )

        self.boundary_temperature = float(boundary_temperature)
        self.correction_scale_max = float(correction_scale_max)
        self.schedule_scale = 1.0

        self.last_sdf_prior = None
        self.last_boundary_gate = None
        self.last_base_logits = None
        self.last_raw_correction = None
        self.last_bounded_correction = None
        self.last_logit_correction = None
        self.last_final_logits = None

        self.sdf_features = nn.Sequential(
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
        )
        self.sdf_head = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.sdf_head.weight)
        nn.init.zeros_(self.sdf_head.bias)

        self.correction_features = nn.Sequential(
            nn.Conv2d(channels + 3, channels, kernel_size=1, bias=False),
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
        self.correction_head = nn.Conv2d(channels, 1, kernel_size=1)
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
        raw_correction = self.last_raw_correction.detach()
        logit_correction = self.last_logit_correction.detach()
        base_prediction = self.last_base_logits.detach() > 0
        final_prediction = self.last_final_logits.detach() > 0
        return {
            "schedule_scale": boundary_gate.new_tensor(self.schedule_scale),
            "effective_correction_scale": self.effective_correction_scale().detach(),
            "boundary_gate_mean": boundary_gate.mean(),
            "boundary_gate_over_05": (
                boundary_gate > 0.5
            ).to(boundary_gate.dtype).mean(),
            "sdf_abs_mean": self.last_sdf_prior.detach().abs().mean(),
            "raw_correction_abs_mean": raw_correction.abs().mean(),
            "logit_correction_abs_mean": logit_correction.abs().mean(),
            "logit_correction_abs_max": logit_correction.abs().max(),
            "prediction_flip_ratio": (
                base_prediction != final_prediction
            ).to(boundary_gate.dtype).mean(),
        }

    def forward(self, feature, base_logits):
        predicted_sdf = torch.tanh(self.sdf_head(self.sdf_features(feature)))
        sdf_prior = predicted_sdf.detach()
        boundary_gate = torch.exp(
            -sdf_prior.abs() / self.boundary_temperature
        )
        base_probability = torch.sigmoid(base_logits).detach()
        raw_correction = self.correction_head(
            self.correction_features(
                torch.cat(
                    [feature, base_probability, sdf_prior, boundary_gate],
                    dim=1,
                )
            )
        )
        bounded_correction = (
            self.effective_correction_scale() * torch.tanh(raw_correction)
        )
        logit_correction = (
            self.schedule_scale * boundary_gate * bounded_correction
        )
        final_logits = base_logits + logit_correction

        self.last_sdf_prior = sdf_prior
        self.last_boundary_gate = boundary_gate
        self.last_base_logits = base_logits
        self.last_raw_correction = raw_correction
        self.last_bounded_correction = bounded_correction
        self.last_logit_correction = logit_correction
        self.last_final_logits = final_logits
        return final_logits, predicted_sdf, logit_correction


class CMUNeXt_HSPM_SDFR_V2(CMUNeXt_HSPM_SDFR):
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
        sdfr_boundary_temperature=0.2,
        sdfr_v2_correction_scale_init=1.0,
        sdfr_v2_correction_scale_max=3.0,
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
            hspm_fusion_gate_init=hspm_fusion_gate_init,
            hspm_fusion_gate_max=hspm_fusion_gate_max,
            hspm_fusion_mode=hspm_fusion_mode,
            hspm_small_area_threshold=hspm_small_area_threshold,
            hspm_small_area_temperature=hspm_small_area_temperature,
            sdfr_boundary_temperature=sdfr_boundary_temperature,
        )
        self.sdfr = SignedDistanceLogitCorrection(
            channels=dims[0],
            boundary_temperature=sdfr_boundary_temperature,
            correction_scale_init=sdfr_v2_correction_scale_init,
            correction_scale_max=sdfr_v2_correction_scale_max,
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
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        base_logits = self.seg_head(d2)
        final_logits, predicted_sdf, logit_correction = self.sdfr(d2, base_logits)
        return {
            "seg": final_logits,
            "base_seg": base_logits,
            "coarse": coarse_logits,
            "uncertainty": uncertainty,
            "sdf": predicted_sdf,
            "logit_correction": logit_correction,
        }


def cmunext_hspm_sdfr_v2(
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
    sdfr_boundary_temperature=0.2,
    sdfr_v2_correction_scale_init=1.0,
    sdfr_v2_correction_scale_max=3.0,
):
    return CMUNeXt_HSPM_SDFR_V2(
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
        sdfr_boundary_temperature=sdfr_boundary_temperature,
        sdfr_v2_correction_scale_init=sdfr_v2_correction_scale_init,
        sdfr_v2_correction_scale_max=sdfr_v2_correction_scale_max,
    )
