import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.conv_based.CMUNeXt import (
    CMUNeXtBlock,
    conv_block,
    fusion_conv,
    up_conv,
)


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
    def __init__(self, channels, temperature=0.1, gamma_init=0.1, eps=1e-6):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.temperature = float(temperature)
        self.eps = float(eps)
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
        enhanced_feature = feature + self.gamma * self.context_proj(prototype_context)
        return enhanced_feature, coarse_logits, uncertainty


class CMUNeXt_HSPM(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        hspm_mode="full",
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("CMUNeXt_HSPM supports binary segmentation only; num_classes must be 1.")
        if hspm_mode not in {"full", "context_only"}:
            raise ValueError("hspm_mode must be either 'full' or 'context_only'.")

        self.hspm_mode = hspm_mode
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])

        self.high_resolution_context = HighResolutionContext(ch_in=dims[3], ch_out=dims[4])
        self.prototype_mixer = ConfidenceAwarePrototypeMixer(channels=dims[4], temperature=0.1)

        self.Up4 = up_conv(ch_in=dims[4], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))

        bottleneck = self.high_resolution_context(x4)
        bottleneck, coarse_logits, uncertainty = self.prototype_mixer(
            bottleneck,
            use_prototype=self.hspm_mode == "full",
        )

        d4 = self.Up4(bottleneck)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        return {
            "seg": self.Conv_1x1(d2),
            "coarse": coarse_logits,
            "uncertainty": uncertainty,
        }


def cmunext_hspm(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    hspm_mode="full",
):
    return CMUNeXt_HSPM(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        hspm_mode=hspm_mode,
    )
