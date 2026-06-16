import torch
import torch.nn as nn

from src.network.conv_based.CMUNeXt import CMUNeXtBlock, conv_block, fusion_conv, up_conv
from src.network.conv_based.CMUNeXt_HSPM_FBDM import FBDM


class CMUNeXt_FBDM(nn.Module):
    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=(16, 32, 128, 160, 256),
        depths=(1, 1, 1, 3, 1),
        kernels=(3, 3, 7, 7, 7),
        fbdm_gate_init=0.03,
        fbdm_gate_max=0.2,
        fbdm_edge_aux_only=False,
    ):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.fbdm1 = FBDM(
            channels=dims[0],
            edge_channels=num_classes,
            use_hspm_prior=False,
            gate_init=fbdm_gate_init,
            gate_max=fbdm_gate_max,
            edge_aux_only=fbdm_edge_aux_only,
        )

    def forward(self, x):
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))
        x5 = self.encoder5(self.Maxpool(x4))

        d5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        x1_fbdm, edge_logits = self.fbdm1(x1)
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((x1_fbdm, d2), dim=1))

        return {
            "seg": self.Conv_1x1(d2),
            "edge": edge_logits,
        }


def cmunext_fbdm(
    input_channel=3,
    num_classes=1,
    dims=(16, 32, 128, 160, 256),
    depths=(1, 1, 1, 3, 1),
    kernels=(3, 3, 7, 7, 7),
    **kwargs,
):
    return CMUNeXt_FBDM(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        **kwargs,
    )


def cmunext_fbdm_s(
    input_channel=3,
    num_classes=1,
    dims=(8, 16, 32, 64, 128),
    depths=(1, 1, 1, 1, 1),
    kernels=(3, 3, 7, 7, 9),
    **kwargs,
):
    return CMUNeXt_FBDM(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        **kwargs,
    )


def cmunext_fbdm_l(
    input_channel=3,
    num_classes=1,
    dims=(32, 64, 128, 256, 512),
    depths=(1, 1, 1, 6, 3),
    kernels=(3, 3, 7, 7, 7),
    **kwargs,
):
    return CMUNeXt_FBDM(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims,
        depths=depths,
        kernels=kernels,
        **kwargs,
    )
