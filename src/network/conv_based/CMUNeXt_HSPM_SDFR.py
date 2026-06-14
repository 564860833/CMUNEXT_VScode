import torch
import torch.nn as nn

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


class SignedDistanceBoundaryRefinement(nn.Module):
    def __init__(
        self,
        channels,
        boundary_temperature=0.2,
        refine_scale_init=0.05,
        refine_scale_max=0.3,
    ):
        super().__init__()
        if boundary_temperature <= 0:
            raise ValueError("boundary_temperature must be positive.")
        if not 0.0 < refine_scale_init < refine_scale_max:
            raise ValueError("refine_scale_init must be in (0, refine_scale_max).")

        self.boundary_temperature = float(boundary_temperature)
        self.refine_scale_max = float(refine_scale_max)
        self.schedule_scale = 1.0
        self.last_sdf_prior = None
        self.last_boundary_gate = None
        self.last_boundary_residual = None
        self.last_refined_feature = None

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

        self.refine = nn.Sequential(
            nn.Conv2d(channels + 2, channels, kernel_size=1, bias=False),
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
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

        refine_ratio = float(refine_scale_init) / self.refine_scale_max
        self.refine_scale_raw = nn.Parameter(
            torch.tensor(_inverse_sigmoid(refine_ratio), dtype=torch.float32)
        )

    def effective_refine_scale(self):
        return self.refine_scale_max * torch.sigmoid(self.refine_scale_raw)

    def set_schedule_scale(self, scale):
        self.schedule_scale = min(max(float(scale), 0.0), 1.0)

    def diagnostics(self):
        if self.last_boundary_gate is None:
            return None
        residual = self.last_boundary_residual.detach()
        return {
            "schedule_scale": residual.new_tensor(self.schedule_scale),
            "effective_refine_scale": self.effective_refine_scale().detach(),
            "boundary_gate_mean": self.last_boundary_gate.detach().mean(),
            "boundary_gate_over_05": (
                self.last_boundary_gate.detach() > 0.5
            ).to(residual.dtype).mean(),
            "sdf_abs_mean": self.last_sdf_prior.detach().abs().mean(),
            "residual_abs_mean": residual.abs().mean(),
        }

    def forward(self, feature):
        predicted_sdf = torch.tanh(self.sdf_head(self.sdf_features(feature)))
        sdf_prior = predicted_sdf.detach()
        boundary_gate = torch.exp(
            -sdf_prior.abs() / self.boundary_temperature
        )
        boundary_residual = self.refine(
            torch.cat([feature, sdf_prior, boundary_gate], dim=1)
        )
        refined_feature = (
            feature
            + self.schedule_scale
            * self.effective_refine_scale()
            * boundary_gate
            * boundary_residual
        )

        self.last_sdf_prior = sdf_prior
        self.last_boundary_gate = boundary_gate
        self.last_boundary_residual = boundary_residual
        self.last_refined_feature = refined_feature
        return refined_feature, predicted_sdf


class CMUNeXt_HSPM_SDFR(nn.Module):
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
        sdfr_refine_scale_init=0.05,
        sdfr_refine_scale_max=0.3,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("CMUNeXt_HSPM_SDFR supports binary segmentation only.")
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

        self.sdfr = SignedDistanceBoundaryRefinement(
            channels=dims[0],
            boundary_temperature=sdfr_boundary_temperature,
            refine_scale_init=sdfr_refine_scale_init,
            refine_scale_max=sdfr_refine_scale_max,
        )
        self.seg_head = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def effective_fusion_gate(self):
        return self.hspm_fusion_gate_max * torch.sigmoid(self.fusion_gate_raw)

    def set_sdfr_refine_schedule_scale(self, scale):
        self.sdfr.set_schedule_scale(scale)

    def get_sdfr_diagnostics(self):
        return self.sdfr.diagnostics()

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
        deep_rms = deep_feature.detach().square().mean(dim=(1, 2, 3)).sqrt().clamp_min(eps)
        diagnostics = {
            "effective_fusion_gate": effective_gate.detach().mean(),
            "injection_deep_rms_ratio": (injection_rms / deep_rms).mean(),
        }
        if spatial_gate is not None:
            diagnostics.update(
                {
                    "predicted_area": predicted_area.detach().mean(),
                    "smallness": smallness.detach().mean(),
                    "spatial_gate": spatial_gate.detach().mean(),
                }
            )
        self.last_fusion_diagnostics = diagnostics

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

        refined_d2, predicted_sdf = self.sdfr(d2)
        return {
            "seg": self.seg_head(refined_d2),
            "coarse": coarse_logits,
            "uncertainty": uncertainty,
            "sdf": predicted_sdf,
        }


def cmunext_hspm_sdfr(
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
    sdfr_refine_scale_init=0.05,
    sdfr_refine_scale_max=0.3,
):
    return CMUNeXt_HSPM_SDFR(
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
        sdfr_refine_scale_init=sdfr_refine_scale_init,
        sdfr_refine_scale_max=sdfr_refine_scale_max,
    )
