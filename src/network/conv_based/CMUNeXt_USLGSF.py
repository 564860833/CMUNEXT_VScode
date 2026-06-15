import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.conv_based.CMUNeXt import (
    CMUNeXtBlock,
    conv_block,
    fusion_conv,
    up_conv,
)


def _inverse_sigmoid(value, eps=1e-6):
    value = min(max(float(value), eps), 1.0 - eps)
    return torch.logit(torch.tensor(value)).item()


class USLGSF(nn.Module):
    """Ultrasound local-structure guided skip fusion."""

    MODES = {"full", "context_only", "structure_only", "relevance_only"}

    def __init__(
        self,
        channels,
        smooth_kernels=(3, 7),
        context_downsample=2,
        alpha_init=0.05,
        alpha_max=0.5,
        reduced_channels=None,
        mode="full",
    ):
        super().__init__()
        if len(smooth_kernels) != 2:
            raise ValueError("smooth_kernels must contain exactly two kernel sizes.")
        small_kernel, large_kernel = (int(kernel) for kernel in smooth_kernels)
        if small_kernel <= 0 or large_kernel <= 0:
            raise ValueError("smooth kernels must be positive.")
        if small_kernel % 2 == 0 or large_kernel % 2 == 0:
            raise ValueError("smooth kernels must be odd.")
        if small_kernel >= large_kernel:
            raise ValueError("smooth_kernels must be ordered from small to large.")
        if context_downsample < 1:
            raise ValueError("context_downsample must be at least 1.")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be in (0, alpha_max).")
        if alpha_max <= 0.0:
            raise ValueError("alpha_max must be positive.")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}.")

        reduced_channels = reduced_channels or max(channels // 4, 8)
        if reduced_channels <= 0:
            raise ValueError("reduced_channels must be positive.")

        self.channels = int(channels)
        self.smooth_kernels = (small_kernel, large_kernel)
        self.context_downsample = int(context_downsample)
        self.alpha_max = float(alpha_max)
        self.mode = mode
        self.last_diagnostics = None

        self.structure_predictor = nn.Sequential(
            nn.Conv2d(channels * 4, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.GELU(),
            nn.Conv2d(reduced_channels, 1, kernel_size=3, padding=1),
        )

        self.encoder_projection = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.GELU(),
        )
        self.decoder_projection = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.GELU(),
        )
        self.relevance_predictor = nn.Sequential(
            nn.Conv2d(reduced_channels * 4, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.GELU(),
            nn.Conv2d(reduced_channels, 1, kernel_size=3, padding=1),
        )

        self.context_projection = nn.Sequential(
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
        self.detail_projection = nn.Sequential(
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
        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels * 2,
                kernel_size=3,
                padding=1,
                groups=channels * 2,
                bias=False,
            ),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        alpha_ratio = float(alpha_init) / self.alpha_max
        self.alpha_raw = nn.Parameter(
            torch.tensor(_inverse_sigmoid(alpha_ratio), dtype=torch.float32)
        )

    def effective_alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    @staticmethod
    def _smooth(feature, kernel_size):
        return F.avg_pool2d(
            feature,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )

    def _context_path(self, encoder_feature):
        if self.context_downsample == 1:
            context = encoder_feature
        else:
            context = F.avg_pool2d(
                encoder_feature,
                kernel_size=self.context_downsample,
                stride=self.context_downsample,
                count_include_pad=False,
            )
        context = self.context_projection(context)
        if context.shape[-2:] != encoder_feature.shape[-2:]:
            context = F.interpolate(
                context,
                size=encoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return context

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

        if self.mode == "structure_only":
            final_reliability = structure_reliability
        elif self.mode == "relevance_only":
            final_reliability = decoder_relevance
        else:
            final_reliability = structure_reliability * decoder_relevance

        context_feature = self._context_path(encoder_feature)
        if self.mode == "context_only":
            target_feature = context_feature
        else:
            detail_feature = final_reliability * self.detail_projection(high_frequency)
            target_feature = self.fusion(torch.cat([context_feature, detail_feature], dim=1))

        alpha = self.effective_alpha()
        fused_feature = encoder_feature + alpha * (target_feature - encoder_feature)

        self.last_diagnostics = {
            "structure_reliability": structure_reliability.detach(),
            "decoder_relevance": decoder_relevance.detach(),
            "final_reliability": final_reliability.detach(),
            "effective_alpha": alpha.detach(),
        }
        return fused_feature


class CMUNeXt_USLGSF(nn.Module):
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
        super().__init__()
        self.uslgsf_stages = tuple(dict.fromkeys(int(stage) for stage in uslgsf_stages))
        invalid_stages = set(self.uslgsf_stages) - {0, 1, 2, 3}
        if invalid_stages:
            raise ValueError(f"Unsupported US-LGSF stages: {sorted(invalid_stages)}")
        if not self.uslgsf_stages:
            raise ValueError("uslgsf_stages must include at least one stage.")

        self.last_uslgsf_diagnostics = None
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(
            ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]
        )
        self.encoder2 = CMUNeXtBlock(
            ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]
        )
        self.encoder3 = CMUNeXtBlock(
            ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]
        )
        self.encoder4 = CMUNeXtBlock(
            ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]
        )
        self.encoder5 = CMUNeXtBlock(
            ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]
        )

        skip_dims = dims[:4]
        self.uslgsf_modules = nn.ModuleDict(
            {
                str(stage): USLGSF(
                    channels=skip_dims[stage],
                    smooth_kernels=uslgsf_smooth_kernels,
                    context_downsample=uslgsf_context_downsample,
                    alpha_init=uslgsf_alpha_init,
                    alpha_max=uslgsf_alpha_max,
                    mode=uslgsf_mode,
                )
                for stage in self.uslgsf_stages
            }
        )

        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def _refine_skip(self, stage, encoder_feature, decoder_feature):
        if stage not in self.uslgsf_stages:
            return encoder_feature
        return self.uslgsf_modules[str(stage)](encoder_feature, decoder_feature)

    def _record_diagnostics(self):
        self.last_uslgsf_diagnostics = {
            int(stage): module.last_diagnostics
            for stage, module in self.uslgsf_modules.items()
        }

    def forward(self, x):
        self.last_uslgsf_diagnostics = None
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))
        x5 = self.encoder5(self.Maxpool(x4))

        d5 = self.Up5(x5)
        s4 = self._refine_skip(3, x4, d5)
        d5 = self.Up_conv5(torch.cat((s4, d5), dim=1))

        d4 = self.Up4(d5)
        s3 = self._refine_skip(2, x3, d4)
        d4 = self.Up_conv4(torch.cat((s3, d4), dim=1))

        d3 = self.Up3(d4)
        s2 = self._refine_skip(1, x2, d3)
        d3 = self.Up_conv3(torch.cat((s2, d3), dim=1))

        d2 = self.Up2(d3)
        s1 = self._refine_skip(0, x1, d2)
        d2 = self.Up_conv2(torch.cat((s1, d2), dim=1))

        self._record_diagnostics()
        return self.Conv_1x1(d2)


def cmunext_uslgsf(
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
    return CMUNeXt_USLGSF(
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
