"""MA-UNet for binary medical image segmentation.

Adapted from https://github.com/HZ-LL/MA-UNet.

This module keeps the upstream model architecture as a plain PyTorch
segmentation network. The forward pass returns raw logits with shape
``[B, num_classes, H, W]`` and does not apply sigmoid or softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["MA_UNet", "ma_unet"]


class Attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super().__init__()
        if temperature % 3 != 1:
            raise ValueError("temperature must satisfy temperature % 3 == 1")
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_planes = int(in_planes * ratios) + 1 if in_planes != 3 else K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3

    def forward(self, x):
        x = self.avgpool(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class AT(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        return ca + sa + ca * sa


class DynamicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        padding,
        groups,
        ratio=0.25,
        dilation=1,
        bias=False,
        K=4,
        temperature=34,
        init_weight=True,
    ):
        super().__init__()
        if in_planes % groups != 0:
            raise ValueError("in_planes must be divisible by groups")
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.K = K
        self.attention = Attention2d(in_planes, ratio, K, temperature)
        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.zeros(K, out_planes)) if bias else None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for index in range(self.K):
            nn.init.kaiming_uniform_(self.weight[index])

    def update_temperature(self):
        self.attention.update_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, _, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_attention, weight).view(
            batch_size * self.out_planes,
            self.in_planes // self.groups,
            self.kernel_size,
            self.kernel_size,
        )

        aggregate_bias = None
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)

        output = F.conv2d(
            x,
            weight=aggregate_weight,
            bias=aggregate_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * batch_size,
        )
        return output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))


class FEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1, bias=False)
        self.conv3_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3, bias=False)
        self.conv3_5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.br = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        y = self.spatial_attention(self.conv3_1(x))
        z = self.channel_attention(self.conv3_3(x + y))
        x = self.spatial_attention(self.conv3_5(x + z))
        return self.br(x + y + z)


class REM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dyconv3 = nn.Sequential(
            DynamicConv2d(in_channels, out_channels, 3, 1, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DynamicConv2d(out_channels, out_channels, 3, 1, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dyconv3(x) + self.conv5(x)


class UpDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ChannelAttention(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SpatialAttention(),
            nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.br = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.br(torch.cat([self.conv3(x), self.conv5(x)], dim=1))


class GroupConvShuffle(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.groups = in_channels
        self.group_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def shuffle_channels(self, x):
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, self.groups, num_channels // self.groups, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, height, width)

    def forward(self, x):
        return self.shuffle_channels(self.group_conv(x))


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.reshape = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
        self.query = nn.Conv2d(in_channels, embed_dim, 1)
        self.key = nn.Conv2d(in_channels, embed_dim, 1)
        self.value = nn.Conv2d(in_channels, embed_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size, _, height, width = x.size()
        seq_len = height * width
        x_activated = self.relu(self.bn(x))

        query = self.query(x_activated).view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(2, 3)
        key = self.key(x_activated).view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(3, 2)
        value = self.value(x_activated).view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(2, 3)

        attention_scores = torch.matmul(query, key.transpose(2, 3)) / self.temperature
        attention_scores = attention_scores - attention_scores.amax(dim=-1, keepdim=True)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, value)
        out = out.transpose(2, 3).contiguous().view(batch_size, -1, height, width)
        out = self.gamma * out + self.conv(x)
        return self.reshape(out)


class SSA(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, num_blocks):
        super().__init__()
        self.ssa_blocks = nn.ModuleList(
            [MultiHeadSelfAttentionBlock(in_channels, embed_dim, num_heads) for _ in range(num_blocks)]
        )

    def forward(self, x):
        for ssa in self.ssa_blocks:
            x = ssa(x)
        return x


class MA_UNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=1):
        super().__init__()
        features = [64, 128, 256, 512]
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.sk = nn.ModuleList()
        self.br = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = input_channel
        for feature in features:
            if feature <= 128:
                self.downs.append(REM(in_channels, feature))
            else:
                self.downs.append(FEM(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.sk.append(AT(feature))
            self.br.append(
                nn.Sequential(
                    nn.Conv2d(feature, feature, 1, bias=False),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                )
            )
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(UpDoubleConv(feature * 2, feature))

        self.shuffleconv = nn.Sequential(
            GroupConvShuffle(features[-1]),
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=1),
        )
        self.sa = SSA(features[-1] * 2, features[-1] * 3, num_heads=8, num_blocks=1)
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.shuffleconv(x)
        temp = x
        x = self.sa(x)
        temp = temp + x
        skip_connections = skip_connections[::-1]

        for index, skip_connection in enumerate(skip_connections):
            temp = self.ups[index * 2](temp)
            temp = temp + skip_connection
            skip_connections[index] = self.br[index](self.sk[index](temp))

        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip_connection = skip_connections[index // 2]
            x = self.ups[index + 1](torch.cat((skip_connection, x), dim=1))

        return self.final_conv(x)


def ma_unet(input_channel=3, num_classes=1):
    return MA_UNet(input_channel=input_channel, num_classes=num_classes)
