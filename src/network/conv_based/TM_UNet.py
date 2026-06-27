"""TM-UNet for binary medical image segmentation.

Adapted from https://github.com/xq141839/TM-UNet.

The upstream implementation is kept as a plain PyTorch segmentation network.
The forward pass returns raw logits with shape ``[B, num_classes, H, W]`` and
does not apply sigmoid or softmax.
"""

from enum import Enum
import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_


__all__ = ["TM_UNet", "tm_unet"]


TMUNET_SIZE_DIMS = {
    "t": (8, 16, 32, 64, 128),
    "s": (16, 32, 64, 128, 256),
    "b": (32, 64, 128, 256, 512),
}


def _resolve_embed_dims(model_size):
    normalized_size = str(model_size).lower()
    if normalized_size not in TMUNET_SIZE_DIMS:
        raise ValueError("model_size must be one of: t, s, b.")
    return normalized_size, TMUNET_SIZE_DIMS[normalized_size]


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )

    def forward(self, x):
        return self.pool(x) - x


class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def bias_linspace_init_(param, start=3.4, end=6.0):
    if param.dim() != 1:
        raise ValueError(f"param must be 1-dimensional, got {param.dim()}")
    init_vals = torch.linspace(start, end, param.shape[0], device=param.device, dtype=param.dtype)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param, dim):
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param, dim, num_blocks):
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def parallel_stabilized_simple(
    queries,
    keys,
    values,
    igate_preact,
    fgate_preact,
    lower_triangular_matrix=None,
    stabilize_rowwise=True,
    eps=1e-6,
):
    batch_size, num_heads, sequence_length, head_dim = queries.shape
    dtype, device = queries.dtype, queries.device

    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)
    if lower_triangular_matrix is None or sequence_length < lower_triangular_matrix.size(-1):
        lower_triangular = torch.tril(
            torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device)
        )
    else:
        lower_triangular = lower_triangular_matrix
    if lower_triangular.dtype != torch.bool:
        raise TypeError(f"lower_triangular_matrix must be bool, got {lower_triangular.dtype}")

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((batch_size, num_heads, 1, 1), dtype=dtype, device=device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )
    repeated_log_fgates = log_fgates_cumsum.repeat(1, 1, 1, sequence_length + 1)
    log_fg_matrix_full = repeated_log_fgates - repeated_log_fgates.transpose(-2, -1)
    log_fg_matrix = torch.where(
        lower_triangular,
        log_fg_matrix_full[:, :, 1:, 1:],
        -float("inf"),
    )

    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)
    else:
        max_log_D = torch.max(
            log_D_matrix.view(batch_size, num_heads, -1),
            dim=-1,
            keepdim=True,
        )[0].unsqueeze(-1)
    D_matrix = torch.exp(log_D_matrix - max_log_D)

    keys_scaled = keys / math.sqrt(head_dim)
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)
    C_matrix = qk_matrix * D_matrix
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))
    return (C_matrix / (normalizer + eps)) @ values


class LinearHeadwiseExpand(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        self.bias = nn.Parameter(torch.empty(dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x):
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(x, self.weight, "... nh d, nh out_d d -> ... nh out_d")
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x


class CausalConv1d(nn.Module):
    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x):
        x = einops.rearrange(x, "b l d -> b d l")
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        return einops.rearrange(x, "b d l -> b l d")


class LayerNorm(nn.Module):
    def __init__(
        self,
        ndim=-1,
        weight=True,
        bias=False,
        eps=1e-5,
        residual_weight=True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self):
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        return self.weight

    def forward(self, x):
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Input must be a 4D tensor with shape [B, NH, S, DH].")
        batch_size, num_heads, sequence_length, head_dim = x.shape

        out = x.transpose(1, 2).reshape(batch_size * sequence_length, num_heads * head_dim)
        out = F.group_norm(
            out,
            num_groups=num_heads,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )
        return out.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)


class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def forward(self, q, k, v):
        batch_size, sequence_length, _ = q.shape
        gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)

        igate_preact = self.igate(gate_input).transpose(-1, -2).unsqueeze(-1)
        fgate_preact = self.fgate(gate_input).transpose(-1, -2).unsqueeze(-1)

        cache_key = (sequence_length, str(q.device))
        if cache_key in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[cache_key]
        else:
            causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[cache_key] = causal_mask

        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )
        h_state = self.outnorm(h_state)
        return h_state.transpose(1, 2).reshape(batch_size, sequence_length, -1)

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


class ViLLayer(nn.Module):
    def __init__(
        self,
        dim,
        direction,
        expansion=1,
        qkv_block_size=4,
        proj_bias=False,
        conv_bias=True,
        kernel_size=4,
    ):
        super().__init__()
        if dim % qkv_block_size != 0:
            raise ValueError("dim must be divisible by qkv_block_size")
        self.dim = dim
        self.direction = direction
        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)
        self.q_proj = LinearHeadwiseExpand(inner_dim, num_heads, bias=proj_bias)
        self.k_proj = LinearHeadwiseExpand(inner_dim, num_heads, bias=proj_bias)
        self.v_proj = LinearHeadwiseExpand(inner_dim, num_heads, bias=proj_bias)
        self.conv1d = CausalConv1d(inner_dim, kernel_size=kernel_size, bias=conv_bias)
        self.mlstm_cell = MatrixLSTMCell(inner_dim, qkv_block_size)
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)
        self.reset_parameters()

    def reset_parameters(self):
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)
        nn.init.ones_(self.learnable_skip)

        for projection in (self.q_proj, self.k_proj, self.v_proj):
            small_init_(projection.weight, dim=self.dim)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

        self.mlstm_cell.reset_parameters()

    def forward(self, x):
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_state = self.mlstm_cell(q=q, k=k, v=v)
        h_state = h_state + self.learnable_skip * x_mlstm_conv_act
        x = self.proj_down(h_state * F.silu(z))

        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        return x


class DWBNReLU(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, height, width):
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).view(batch_size, channels, height, width)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


class xLSTMLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0, no_kan=False):
        super().__init__()
        del act_layer, no_kan
        hidden_features = hidden_features or in_features
        self.fc2 = ViLLayer(hidden_features, direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
        self.fc3 = ViLLayer(hidden_features, direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        self.dwconv_2 = DWBNReLU(hidden_features)
        self.dwconv_3 = DWBNReLU(hidden_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, height, width):
        x = self.fc2(x)
        x = self.dwconv_2(x, height, width)
        x = self.fc3(x)
        x = self.dwconv_3(x, height, width)
        return x


class xLSTMBlock(nn.Module):
    def __init__(self, dim, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.layer = xLSTMLayer(
            in_features=dim,
            hidden_features=dim,
            act_layer=act_layer,
            drop=drop,
            no_kan=no_kan,
        )
        self.multi_scale_5x5 = Pooling(5)
        self.multi_scale_9x9 = Pooling(9)
        self.multi_scale_13x13 = Pooling(13)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, height, width):
        x_norm = self.norm1(x)
        x = x + self.multi_scale_5x5(x_norm) + self.multi_scale_9x9(x_norm) + self.multi_scale_13x13(x_norm)
        x = x + self.drop_path(self.layer(self.norm2(x), height, width))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), height, width


class TM_UNet(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        img_size=256,
        model_size="b",
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=(1, 1, 1),
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.model_size, embed_dims = _resolve_embed_dims(model_size)
        self.embed_dims = tuple(embed_dims)
        if len(depths) != 3:
            raise ValueError("depths must contain three values.")

        self.encoder1 = nn.Conv2d(input_channel, embed_dims[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(embed_dims[0], embed_dims[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(embed_dims[1], embed_dims[2], 3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(embed_dims[0])
        self.ebn2 = nn.BatchNorm2d(embed_dims[1])
        self.ebn3 = nn.BatchNorm2d(embed_dims[2])

        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList(
            [xLSTMBlock(dim=embed_dims[3], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)]
        )
        self.block2 = nn.ModuleList(
            [xLSTMBlock(dim=embed_dims[4], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)]
        )
        self.dblock1 = nn.ModuleList(
            [xLSTMBlock(dim=embed_dims[3], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)]
        )
        self.dblock2 = nn.ModuleList(
            [xLSTMBlock(dim=embed_dims[2], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)]
        )

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[3],
            embed_dim=embed_dims[4],
        )

        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(embed_dims[2], embed_dims[1], 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(embed_dims[1], embed_dims[0], 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(embed_dims[0], embed_dims[0] // 2, 3, stride=1, padding=1)
        self.dbn1 = nn.BatchNorm2d(embed_dims[3])
        self.dbn2 = nn.BatchNorm2d(embed_dims[2])
        self.dbn3 = nn.BatchNorm2d(embed_dims[1])
        self.dbn4 = nn.BatchNorm2d(embed_dims[0])
        self.final = nn.Conv2d(embed_dims[0] // 2, num_classes, kernel_size=1)

    def forward(self, x):
        batch_size = x.shape[0]

        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out, height, width = self.patch_embed3(out)
        for block in self.block1:
            out = block(out, height, width)
        out = self.norm3(out)
        out = out.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, height, width = self.patch_embed4(out)
        for block in self.block2:
            out = block(out, height, width)
        out = self.norm4(out)
        out = out.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode="bilinear"))
        out = out + t4
        _, _, height, width = out.shape
        out = out.flatten(2).transpose(1, 2)
        for block in self.dblock1:
            out = block(out, height, width)

        out = self.dnorm3(out)
        out = out.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode="bilinear"))
        out = out + t3
        _, _, height, width = out.shape
        out = out.flatten(2).transpose(1, 2)
        for block in self.dblock2:
            out = block(out, height, width)

        out = self.dnorm4(out)
        out = out.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode="bilinear"))
        out = out + t2
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode="bilinear"))
        out = out + t1
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode="bilinear"))
        return self.final(out)


def tm_unet(input_channel=3, num_classes=1, img_size=256, model_size="b"):
    return TM_UNet(
        input_channel=input_channel,
        num_classes=num_classes,
        img_size=img_size,
        model_size=model_size,
    )
