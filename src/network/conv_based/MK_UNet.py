"""MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation.

Adapted from https://github.com/SLDGroup/MK-UNet.

BSD 3-Clause License

Copyright (c) 2025, System Level Design Group

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from math import gcd

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["MK_UNet"]


def _init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == "relu":
        return nn.ReLU(inplace)
    if act == "relu6":
        return nn.ReLU6(inplace)
    if act == "leakyrelu":
        return nn.LeakyReLU(neg_slope, inplace)
    if act == "prelu":
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    if act == "gelu":
        return nn.GELU()
    if act == "hswish":
        return nn.Hardswish(inplace)
    raise NotImplementedError(f"activation layer [{act}] is not found")


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, height, width)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation="relu"):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes or in_planes
        ratio = min(in_planes, ratio)
        reduced_channels = in_planes // ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(in_planes, reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(reduced_channels, self.out_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.apply(_init_weights)

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7, 11):
            raise ValueError("kernel_size must be 3, 7, or 11")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.apply(_init_weights)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class GroupedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation="relu"):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g,
                F_int,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l,
                F_int,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.activation = act_layer(activation, inplace=True)
        self.apply(_init_weights)

    def forward(self, g, x):
        psi = self.activation(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


class MultiKernelDepthwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_sizes,
        stride,
        activation="relu6",
        dw_parallel=True,
    ):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size,
                        stride,
                        kernel_size // 2,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels),
                    act_layer(activation, inplace=True),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.apply(_init_weights)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MultiKernelInvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        stride,
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        kernel_sizes=(1, 3, 5),
        activation="relu6",
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.add = add
        self.use_skip_connection = stride == 1
        self.ex_c = int(in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True),
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(
            self.ex_c,
            kernel_sizes,
            stride,
            activation,
            dw_parallel=dw_parallel,
        )
        self.combined_channels = self.ex_c if add else self.ex_c * len(kernel_sizes)
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        if self.use_skip_connection and in_c != out_c:
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.apply(_init_weights)

    def forward(self, x):
        dwconv_outs = self.multi_scale_dwconv(self.pconv1(x))
        if self.add:
            dw_out = 0
            for scale_out in dwconv_outs:
                dw_out = dw_out + scale_out
        else:
            dw_out = torch.cat(dwconv_outs, dim=1)
        dw_out = channel_shuffle(dw_out, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dw_out)

        if not self.use_skip_connection:
            return out
        if self.in_c != self.out_c:
            x = self.conv1x1(x)
        return x + out


def mk_irb_bottleneck(
    in_c,
    out_c,
    n,
    stride,
    expansion_factor=2,
    dw_parallel=True,
    add=True,
    kernel_sizes=(1, 3, 5),
    activation="relu6",
):
    blocks = [
        MultiKernelInvertedResidualBlock(
            in_c,
            out_c,
            stride,
            expansion_factor,
            dw_parallel,
            add,
            kernel_sizes,
            activation,
        )
    ]
    for _ in range(1, n):
        blocks.append(
            MultiKernelInvertedResidualBlock(
                out_c,
                out_c,
                1,
                expansion_factor,
                dw_parallel,
                add,
                kernel_sizes,
                activation,
            )
        )
    return nn.Sequential(*blocks)


class MK_UNet(nn.Module):
    """Base MK-UNet model with the official [16, 32, 64, 96, 160] channels."""

    def __init__(
        self,
        num_classes=1,
        in_channels=3,
        channels=(16, 32, 64, 96, 160),
        depths=(1, 1, 1, 1, 1),
        kernel_sizes=(1, 3, 5),
        expansion_factor=2,
        gag_kernel=3,
        **kwargs,
    ):
        super().__init__()
        if len(channels) != 5 or len(depths) != 5:
            raise ValueError("channels and depths must each contain five values")

        self.encoder1 = mk_irb_bottleneck(
            in_channels, channels[0], depths[0], 1, expansion_factor, True, True, kernel_sizes
        )
        self.encoder2 = mk_irb_bottleneck(
            channels[0], channels[1], depths[1], 1, expansion_factor, True, True, kernel_sizes
        )
        self.encoder3 = mk_irb_bottleneck(
            channels[1], channels[2], depths[2], 1, expansion_factor, True, True, kernel_sizes
        )
        self.encoder4 = mk_irb_bottleneck(
            channels[2], channels[3], depths[3], 1, expansion_factor, True, True, kernel_sizes
        )
        self.encoder5 = mk_irb_bottleneck(
            channels[3], channels[4], depths[4], 1, expansion_factor, True, True, kernel_sizes
        )

        self.AG1 = GroupedAttentionGate(
            channels[3], channels[3], channels[3] // 2, gag_kernel, channels[3] // 2
        )
        self.AG2 = GroupedAttentionGate(
            channels[2], channels[2], channels[2] // 2, gag_kernel, channels[2] // 2
        )
        self.AG3 = GroupedAttentionGate(
            channels[1], channels[1], channels[1] // 2, gag_kernel, channels[1] // 2
        )
        self.AG4 = GroupedAttentionGate(
            channels[0], channels[0], channels[0] // 2, gag_kernel, channels[0] // 2
        )

        self.decoder1 = mk_irb_bottleneck(
            channels[4], channels[3], 1, 1, expansion_factor, True, True, kernel_sizes
        )
        self.decoder2 = mk_irb_bottleneck(
            channels[3], channels[2], 1, 1, expansion_factor, True, True, kernel_sizes
        )
        self.decoder3 = mk_irb_bottleneck(
            channels[2], channels[1], 1, 1, expansion_factor, True, True, kernel_sizes
        )
        self.decoder4 = mk_irb_bottleneck(
            channels[1], channels[0], 1, 1, expansion_factor, True, True, kernel_sizes
        )
        self.decoder5 = mk_irb_bottleneck(
            channels[0], channels[0], 1, 1, expansion_factor, True, True, kernel_sizes
        )

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)
        self.SA = SpatialAttention()

        # Keep the official auxiliary heads for checkpoint and parameter-count compatibility.
        self.out1 = nn.Conv2d(channels[2], num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(channels[1], num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-2] % 32 != 0 or x.shape[-1] % 32 != 0:
            raise ValueError("MK_UNet input height and width must be divisible by 32")

        out = F.max_pool2d(self.encoder1(x), 2, 2)
        t1 = out
        out = F.max_pool2d(self.encoder2(out), 2, 2)
        t2 = out
        out = F.max_pool2d(self.encoder3(out), 2, 2)
        t3 = out
        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out
        out = F.max_pool2d(self.encoder5(out), 2, 2)

        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode="bilinear"))
        out = out + self.AG1(g=out, x=t4)

        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode="bilinear"))
        _p1 = F.interpolate(self.out1(out), scale_factor=8, mode="bilinear")
        out = out + self.AG2(g=out, x=t3)

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode="bilinear"))
        _p2 = F.interpolate(self.out2(out), scale_factor=4, mode="bilinear")
        out = out + self.AG3(g=out, x=t2)

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode="bilinear"))
        _p3 = F.interpolate(self.out3(out), scale_factor=2, mode="bilinear")
        out = out + self.AG4(g=out, x=t1)

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode="bilinear"))
        return self.out4(out)
