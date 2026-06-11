import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.conv_based.CMUNeXt import CMUNeXtBlock, conv_block
from src.network.conv_based.CMUNeXt_HSPM import (
    ConfidenceAwarePrototypeMixer,
    HighResolutionContext,
)


class SobelBoundaryEvidence(nn.Module):
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

    def forward(self, probability):
        sobel_x = self.sobel_x.to(dtype=probability.dtype, device=probability.device)
        sobel_y = self.sobel_y.to(dtype=probability.dtype, device=probability.device)
        probability = F.pad(probability, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(probability, sobel_x)
        grad_y = F.conv2d(probability, sobel_y)
        magnitude = (
            torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + self.eps)
            - math.sqrt(self.eps)
        ).clamp_min(0.0)
        spatial_max = magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return (magnitude / spatial_max).clamp(0.0, 1.0)


def _inverse_sigmoid(value):
    return math.log(value / (1.0 - value))


class UncertaintyBoundaryRouter(nn.Module):
    def __init__(self, channels, residual_floor=0.1, route_init=0.1):
        super().__init__()
        self.residual_floor = float(residual_floor)
        self.boundary_evidence = SobelBoundaryEvidence()

        self.skip_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        route_raw = _inverse_sigmoid(route_init)
        self.alpha_raw = nn.Parameter(torch.tensor(route_raw))
        self.beta_raw = nn.Parameter(torch.tensor(route_raw))
        self.lambda_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, skip, coarse_probability, uncertainty, mode="full"):
        if mode not in {"add_only", "semantic_only", "full"}:
            raise ValueError("mode must be 'add_only', 'semantic_only', or 'full'.")

        target_size = skip.shape[-2:]
        projected_skip = self.skip_proj(skip)
        if mode == "add_only":
            route_shape = (skip.shape[0], 1, target_size[0], target_size[1])
            return (
                projected_skip,
                skip.new_ones(route_shape),
                skip.new_zeros(route_shape),
            )

        semantic_route = F.interpolate(
            coarse_probability,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        semantic_feature = semantic_route * self.semantic_conv(skip)

        alpha = torch.sigmoid(self.alpha_raw)
        skip_route = semantic_route
        routed_skip = (
            self.residual_floor + (1.0 - self.residual_floor) * skip_route
        ) * projected_skip
        routed_skip = routed_skip + alpha * semantic_feature

        if mode == "semantic_only":
            return routed_skip, semantic_route, torch.zeros_like(semantic_route)

        uncertainty_route = F.interpolate(
            uncertainty,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        boundary_evidence = self.boundary_evidence(semantic_route)
        boundary_route = (uncertainty_route * boundary_evidence).clamp(0.0, 1.0)
        high_frequency = skip - F.avg_pool2d(
            skip,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        boundary_feature = boundary_route * self.boundary_conv(high_frequency)
        beta = torch.sigmoid(self.beta_raw)
        route_lambda = 2.0 * torch.sigmoid(self.lambda_raw)
        skip_route = torch.clamp(semantic_route + route_lambda * boundary_route, min=0.0, max=1.0)
        routed_skip = (
            self.residual_floor + (1.0 - self.residual_floor) * skip_route
        ) * projected_skip
        routed_skip = routed_skip + alpha * semantic_feature + beta * boundary_feature
        return routed_skip, semantic_route, boundary_route


class ResidualDepthwiseRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.refine(x)


class ProgressiveAddDecoderStage(nn.Module):
    def __init__(self, decoder_channels, skip_channels, global_channels):
        super().__init__()
        self.decoder_proj = nn.Sequential(
            nn.Conv2d(decoder_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.GELU(),
        )
        self.global_proj = nn.Sequential(
            nn.Conv2d(global_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.GELU(),
        )
        self.refine = ResidualDepthwiseRefine(skip_channels)

    def forward(self, decoder_feature, routed_skip, global_feature):
        target_size = routed_skip.shape[-2:]
        decoder_feature = F.interpolate(
            decoder_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        global_feature = F.interpolate(
            global_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        fused = self.decoder_proj(decoder_feature) + routed_skip + self.global_proj(global_feature)
        return self.refine(fused)


class CMUNeXt_HSPM_UBRD(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        hspm_mode="full",
        ubrd_mode="full",
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("CMUNeXt_HSPM_UBRD supports binary segmentation only; num_classes must be 1.")
        if hspm_mode not in {"full", "context_only"}:
            raise ValueError("hspm_mode must be either 'full' or 'context_only'.")
        if ubrd_mode not in {"add_only", "semantic_only", "full"}:
            raise ValueError("ubrd_mode must be 'add_only', 'semantic_only', or 'full'.")

        self.hspm_mode = hspm_mode
        self.ubrd_mode = ubrd_mode
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])

        self.high_resolution_context = HighResolutionContext(ch_in=dims[3], ch_out=dims[4])
        self.prototype_mixer = ConfidenceAwarePrototypeMixer(channels=dims[4], temperature=0.1)

        self.router3 = UncertaintyBoundaryRouter(channels=dims[2])
        self.router2 = UncertaintyBoundaryRouter(channels=dims[1])
        self.router1 = UncertaintyBoundaryRouter(channels=dims[0])
        self.decoder3 = ProgressiveAddDecoderStage(dims[4], dims[2], dims[4])
        self.decoder2 = ProgressiveAddDecoderStage(dims[2], dims[1], dims[4])
        self.decoder1 = ProgressiveAddDecoderStage(dims[1], dims[0], dims[4])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))

        global_feature = self.high_resolution_context(x4)
        global_feature, coarse_logits, uncertainty = self.prototype_mixer(
            global_feature,
            use_prototype=self.hspm_mode == "full",
        )
        coarse_probability = torch.sigmoid(coarse_logits)

        skip3, _, _ = self.router3(x3, coarse_probability, uncertainty, mode=self.ubrd_mode)
        d3 = self.decoder3(global_feature, skip3, global_feature)

        skip2, _, _ = self.router2(x2, coarse_probability, uncertainty, mode=self.ubrd_mode)
        d2 = self.decoder2(d3, skip2, global_feature)

        skip1, semantic_route, boundary_route = self.router1(
            x1,
            coarse_probability,
            uncertainty,
            mode=self.ubrd_mode,
        )
        d1 = self.decoder1(d2, skip1, global_feature)

        return {
            "seg": self.Conv_1x1(d1),
            "coarse": coarse_logits,
            "uncertainty": uncertainty,
            "semantic_route": semantic_route,
            "boundary_route": boundary_route,
        }


def cmunext_hspm_ubrd(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    hspm_mode="full",
    ubrd_mode="full",
):
    return CMUNeXt_HSPM_UBRD(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        hspm_mode=hspm_mode,
        ubrd_mode=ubrd_mode,
    )
