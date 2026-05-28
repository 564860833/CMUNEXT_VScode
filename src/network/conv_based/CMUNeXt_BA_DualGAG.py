import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.conv_based.CMUNeXt_DualGAG import (
    CMUNeXt_DualGAG,
    _GAG_STAGE_ATTRS,
    _normalize_gag_stages,
)


class BoundaryAwareDualGatedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, groups=4, reduction=8):
        super().__init__()
        actual_groups = groups if (F_int % groups == 0 and F_l % groups == 0) else 1
        self.groups = actual_groups

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=3, stride=1, padding=1, groups=actual_groups, bias=False),
            nn.BatchNorm2d(F_int),
            nn.GELU(),
        )
        self.spatial_gate = nn.Conv2d(F_int, actual_groups, kernel_size=1, stride=1, padding=0, bias=True)
        self.spatial_scale = nn.Parameter(torch.full((actual_groups,), 0.1))

        self.boundary_gate = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=3, stride=1, padding=1, groups=actual_groups, bias=False),
            nn.BatchNorm2d(F_int),
            nn.GELU(),
            nn.Conv2d(F_int, actual_groups, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.boundary_scale = nn.Parameter(torch.full((actual_groups,), 0.1))
        self.boundary_mix = nn.Parameter(torch.full((actual_groups,), -2.0))

        channel_mid = max(8, F_int // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(F_int, channel_mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_mid, F_l, kernel_size=1, bias=False),
        )
        self.channel_scale = nn.Parameter(torch.tensor(0.1))
        self.relu = nn.ReLU(inplace=True)

    def _expand_group_gate(self, gate, x):
        if self.groups == 1:
            return gate

        gate_chunks = []
        for group_idx, chunk in enumerate(torch.chunk(x, self.groups, dim=1)):
            gate_chunks.append(gate[:, group_idx:group_idx + 1].expand(-1, chunk.size(1), -1, -1))
        return torch.cat(gate_chunks, dim=1)

    def forward(self, g, x):
        fused = self.relu(self.W_g(g) + self.W_x(x))
        fused = self.refine(fused)

        spatial_gate = torch.sigmoid(self.spatial_gate(fused))
        spatial_gate = 2.0 * spatial_gate - 1.0
        spatial_gate = 1.0 + torch.tanh(self.spatial_scale).view(1, self.groups, 1, 1) * spatial_gate

        boundary_prior = fused - F.avg_pool2d(
            fused,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        boundary_gate = torch.sigmoid(self.boundary_gate(boundary_prior))
        boundary_gate = 2.0 * boundary_gate - 1.0
        boundary_gate = 1.0 + torch.tanh(self.boundary_scale).view(1, self.groups, 1, 1) * boundary_gate

        boundary_mix = torch.sigmoid(self.boundary_mix).view(1, self.groups, 1, 1)
        joint_gate = spatial_gate + boundary_mix * (boundary_gate - 1.0)
        spatial_mod = self._expand_group_gate(joint_gate, x)

        channel_gate = self.channel_mlp(self.avg_pool(fused)) + self.channel_mlp(self.max_pool(fused))
        channel_gate = torch.sigmoid(channel_gate)
        channel_gate = 2.0 * channel_gate - 1.0
        channel_gate = 1.0 + torch.tanh(self.channel_scale) * channel_gate

        return x * spatial_mod * channel_gate


def _make_ba_gag(stage, dims):
    if stage == 3:
        return BoundaryAwareDualGatedAttentionGate(F_g=dims[3], F_l=dims[3], F_int=max(8, dims[3] // 2), groups=4)
    if stage == 2:
        return BoundaryAwareDualGatedAttentionGate(F_g=dims[2], F_l=dims[2], F_int=max(8, dims[2] // 2), groups=4)
    if stage == 1:
        return BoundaryAwareDualGatedAttentionGate(F_g=dims[1], F_l=dims[1], F_int=max(8, dims[1] // 2), groups=4)
    if stage == 0:
        return BoundaryAwareDualGatedAttentionGate(F_g=dims[0], F_l=dims[0], F_int=max(8, dims[0] // 2), groups=2)
    raise ValueError(f"Unsupported BA-DualGAG stage: {stage}")


class CMUNeXt_BA_DualGAG(CMUNeXt_DualGAG):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        gag_stages=None,
        use_shallow_gates=False,
    ):
        super().__init__(
            input_channel=input_channel,
            num_classes=num_classes,
            dims=dims,
            depths=depths,
            kernels=kernels,
            gag_stages=gag_stages,
            use_shallow_gates=use_shallow_gates,
        )
        self.gag_stages = set(_normalize_gag_stages(gag_stages, use_shallow_gates))
        for stage in sorted(self.gag_stages, reverse=True):
            setattr(self, _GAG_STAGE_ATTRS[stage], _make_ba_gag(stage, dims))


def cmunext_ba_dualgag(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    gag_stages=(2, 3),
):
    return CMUNeXt_BA_DualGAG(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        gag_stages=gag_stages,
    )


def cmunext_ba_dualgag_s(input_channel=3, num_classes=1, gag_stages=(2, 3)):
    return CMUNeXt_BA_DualGAG(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(8, 16, 32, 64, 128),
        depths=(1, 1, 1, 1, 1),
        kernels=(3, 3, 7, 7, 9),
        gag_stages=gag_stages,
    )


def cmunext_ba_dualgag_l(input_channel=3, num_classes=1, gag_stages=(2, 3)):
    return CMUNeXt_BA_DualGAG(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 3),
        kernels=(3, 3, 7, 7, 7),
        gag_stages=gag_stages,
    )
