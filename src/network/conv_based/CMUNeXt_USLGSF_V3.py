import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.conv_based.CMUNeXt_USLGSF import (
    CMUNeXt_USLGSF,
    USLGSF,
)


class USLGSFV3(USLGSF):
    """Dynamic structure-relevance US-LGSF with an identity-safe residual."""

    MODES = {"full", "context_only", "structure_only", "relevance_only"}

    def __init__(self, *args, residual_init_scale=0.05, **kwargs):
        mode = kwargs.pop("mode", "full")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}.")
        if residual_init_scale <= 0:
            raise ValueError("residual_init_scale must be positive.")

        super().__init__(*args, mode="full", **kwargs)
        self.mode = mode
        self.residual_init_scale = float(residual_init_scale)
        self.route_scale = 1.0
        mix_channels = max(self.channels // 4, 8)
        self.mix_predictor = nn.Sequential(
            nn.Conv2d(4, mix_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mix_channels),
            nn.GELU(),
            nn.Conv2d(mix_channels, 2, kernel_size=1),
        )
        self.residual_projection = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.residual_projection.weight,
            mode="fan_out",
            nonlinearity="linear",
        )
        with torch.no_grad():
            self.residual_projection.weight.mul_(self.residual_init_scale)
        self.last_mix_weights = None
        self.last_active_gate = None

    def set_route_scale(self, scale):
        self.route_scale = min(max(float(scale), 0.0), 1.0)

    def forward(self, encoder_feature, decoder_feature):
        if encoder_feature.shape[1] != self.channels:
            raise ValueError(
                f"encoder_feature must have {self.channels} channels, "
                f"got {encoder_feature.shape[1]}."
            )
        if decoder_feature.shape[1] != self.channels:
            raise ValueError(
                f"decoder_feature must have {self.channels} channels, "
                f"got {decoder_feature.shape[1]}."
            )
        if decoder_feature.shape[-2:] != encoder_feature.shape[-2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=encoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        small_smooth = self._smooth(encoder_feature, self.smooth_kernels[0])
        large_smooth = self._smooth(encoder_feature, self.smooth_kernels[1])
        high_frequency = encoder_feature - small_smooth
        scale_difference = torch.abs(small_smooth - large_smooth)

        structure_input = torch.cat(
            [
                small_smooth,
                large_smooth,
                torch.abs(high_frequency),
                scale_difference,
            ],
            dim=1,
        )
        structure_reliability = torch.sigmoid(self.structure_predictor(structure_input))

        encoder_query = self.encoder_projection(small_smooth)
        decoder_query = self.decoder_projection(decoder_feature)
        relevance_input = torch.cat(
            [
                encoder_query,
                decoder_query,
                encoder_query * decoder_query,
                torch.abs(encoder_query - decoder_query),
            ],
            dim=1,
        )
        decoder_relevance = torch.sigmoid(self.relevance_predictor(relevance_input))

        mix_input = torch.cat(
            [
                structure_reliability,
                decoder_relevance,
                structure_reliability * decoder_relevance,
                torch.abs(structure_reliability - decoder_relevance),
            ],
            dim=1,
        )
        mix_weights = torch.softmax(self.mix_predictor(mix_input), dim=1)
        structure_weight = mix_weights[:, 0:1]
        relevance_weight = mix_weights[:, 1:2]

        if self.mode == "context_only":
            active_gate = torch.ones_like(structure_reliability)
        elif self.mode == "structure_only":
            active_gate = structure_reliability
        elif self.mode == "relevance_only":
            active_gate = decoder_relevance
        else:
            active_gate = (
                structure_weight * structure_reliability
                + relevance_weight * decoder_relevance
            )

        context_feature = self._context_path(encoder_feature)
        if self.mode == "context_only":
            candidate = context_feature
        else:
            detail_feature = self.detail_projection(high_frequency)
            candidate = self.fusion(torch.cat([context_feature, detail_feature], dim=1))

        residual_delta = torch.tanh(self.residual_projection(candidate))
        alpha = self.effective_alpha()
        injection = self.route_scale * alpha * active_gate * residual_delta
        fused_feature = encoder_feature + injection

        eps = torch.finfo(encoder_feature.dtype).eps
        injection_rms = injection.detach().square().mean(dim=(1, 2, 3)).sqrt()
        encoder_rms = (
            encoder_feature.detach()
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt()
            .clamp_min(eps)
        )
        self.last_mix_weights = mix_weights.detach()
        self.last_active_gate = active_gate.detach()
        self.last_diagnostics = {
            "structure_reliability_mean": structure_reliability.detach().mean(),
            "decoder_relevance_mean": decoder_relevance.detach().mean(),
            "structure_weight_mean": structure_weight.detach().mean(),
            "relevance_weight_mean": relevance_weight.detach().mean(),
            "active_gate_mean": active_gate.detach().mean(),
            "route_scale": active_gate.new_tensor(self.route_scale),
            "effective_alpha": alpha.detach(),
            "residual_delta_abs_mean": residual_delta.detach().abs().mean(),
            "injection_encoder_rms_ratio": (injection_rms / encoder_rms).mean(),
        }
        return fused_feature


class CMUNeXt_USLGSF_V3(CMUNeXt_USLGSF):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        uslgsf_stages=(0, 1),
        uslgsf_smooth_kernels=(3, 7),
        uslgsf_context_downsample=2,
        uslgsf_alpha_init=0.05,
        uslgsf_alpha_max=0.5,
        uslgsf_mode="full",
        uslgsf_residual_init_scale=0.05,
    ):
        normalized_stages = tuple(dict.fromkeys(int(stage) for stage in uslgsf_stages))
        invalid_stages = set(normalized_stages) - {0, 1}
        if invalid_stages:
            raise ValueError(
                "CMUNeXt_USLGSF_V3 supports shallow skip stages 0 and 1 only; "
                f"got {sorted(invalid_stages)}."
            )
        super().__init__(
            input_channel=input_channel,
            num_classes=num_classes,
            dims=dims,
            depths=depths,
            kernels=kernels,
            uslgsf_stages=normalized_stages,
            uslgsf_smooth_kernels=uslgsf_smooth_kernels,
            uslgsf_context_downsample=uslgsf_context_downsample,
            uslgsf_alpha_init=uslgsf_alpha_init,
            uslgsf_alpha_max=uslgsf_alpha_max,
            uslgsf_mode="full",
        )

        skip_dims = dims[:4]
        self.uslgsf_modules = nn.ModuleDict(
            {
                str(stage): USLGSFV3(
                    channels=skip_dims[stage],
                    smooth_kernels=uslgsf_smooth_kernels,
                    context_downsample=(
                        1 if stage == 0 else uslgsf_context_downsample
                    ),
                    alpha_init=uslgsf_alpha_init,
                    alpha_max=uslgsf_alpha_max,
                    mode=uslgsf_mode,
                    residual_init_scale=uslgsf_residual_init_scale,
                )
                for stage in self.uslgsf_stages
            }
        )

    def set_uslgsf_route_scale(self, scale):
        for module in self.uslgsf_modules.values():
            module.set_route_scale(scale)


def cmunext_uslgsf_v3(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    uslgsf_stages=(0, 1),
    uslgsf_smooth_kernels=(3, 7),
    uslgsf_context_downsample=2,
    uslgsf_alpha_init=0.05,
    uslgsf_alpha_max=0.5,
    uslgsf_mode="full",
    uslgsf_residual_init_scale=0.05,
):
    return CMUNeXt_USLGSF_V3(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        uslgsf_stages=uslgsf_stages,
        uslgsf_smooth_kernels=uslgsf_smooth_kernels,
        uslgsf_context_downsample=uslgsf_context_downsample,
        uslgsf_alpha_init=uslgsf_alpha_init,
        uslgsf_alpha_max=uslgsf_alpha_max,
        uslgsf_mode=uslgsf_mode,
        uslgsf_residual_init_scale=uslgsf_residual_init_scale,
    )
