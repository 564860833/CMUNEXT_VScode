import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.conv_based.CMUNeXt_USLGSF import (
    CMUNeXt_USLGSF,
    USLGSF,
)


class USLGSFV2(USLGSF):
    """Identity-safe US-LGSF with uncertainty-routed residual injection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_projection = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.residual_projection.weight)

    @staticmethod
    def build_decoder_uncertainty(decoder_relevance):
        relevance = decoder_relevance.detach()
        uncertainty = 4.0 * relevance * (1.0 - relevance)
        return F.max_pool2d(uncertainty, kernel_size=3, stride=1, padding=1)

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
        decoder_uncertainty = self.build_decoder_uncertainty(decoder_relevance)

        if self.mode == "structure_only":
            active_gate = structure_reliability
        elif self.mode == "relevance_only":
            active_gate = decoder_uncertainty
        else:
            active_gate = structure_reliability * decoder_uncertainty

        context_feature = self._context_path(encoder_feature)
        if self.mode == "context_only":
            candidate = context_feature
        else:
            detail_feature = self.detail_projection(high_frequency)
            if self.mode != "structure_only":
                detail_feature = decoder_relevance * detail_feature
            candidate = self.fusion(torch.cat([context_feature, detail_feature], dim=1))

        residual_delta = torch.tanh(self.residual_projection(candidate))
        alpha = self.effective_alpha()
        injection = alpha * active_gate * residual_delta
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
        self.last_diagnostics = {
            "structure_reliability": structure_reliability.detach(),
            "decoder_relevance": decoder_relevance.detach(),
            "decoder_uncertainty": decoder_uncertainty.detach(),
            "active_gate": active_gate.detach(),
            "effective_alpha": alpha.detach(),
            "residual_delta_abs_mean": residual_delta.detach().abs().mean(),
            "injection_encoder_rms_ratio": (injection_rms / encoder_rms).mean(),
        }
        return fused_feature


class CMUNeXt_USLGSF_V2(CMUNeXt_USLGSF):
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
    ):
        super().__init__(
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
        )

        skip_dims = dims[:4]
        self.uslgsf_modules = nn.ModuleDict(
            {
                str(stage): USLGSFV2(
                    channels=skip_dims[stage],
                    smooth_kernels=uslgsf_smooth_kernels,
                    context_downsample=(
                        1 if stage == 0 else uslgsf_context_downsample
                    ),
                    alpha_init=uslgsf_alpha_init,
                    alpha_max=uslgsf_alpha_max,
                    mode=uslgsf_mode,
                )
                for stage in self.uslgsf_stages
            }
        )


def cmunext_uslgsf_v2(
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
):
    return CMUNeXt_USLGSF_V2(
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
    )
