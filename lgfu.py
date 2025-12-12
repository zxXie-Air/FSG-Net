# 文件路径: models/my_block/lgfu.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..block.Base import Conv1Relu, Conv3Relu


class LGFU(nn.Module):
    """
    Lightweight Gated Fusion Unit.
    Fuses a deep feature map with a shallow feature map using a learned gate.
    """

    def __init__(self, in_channels_deep, in_channels_shallow):
        super(LGFU, self).__init__()

        # 1x1 Convolution to align the channels of the upsampled deep feature
        # to match the shallow feature's channel dimension.
        self.conv1x1_align = Conv1Relu(in_channels_deep, in_channels_shallow)

        # The Gating Unit (GU)
        # Input channels are C_shallow + C_shallow because we concatenate the
        # aligned deep feature and the shallow feature.
        self.gate_unit = nn.Sequential(
            Conv3Relu(in_channels_shallow * 2, in_channels_shallow),
            nn.Conv2d(in_channels_shallow, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f_deep, f_shallow):
        # f_deep: The deeper, lower-resolution feature map.
        # f_shallow: The shallower, higher-resolution feature map.

        # Upsample the deep feature to match the spatial resolution of the shallow feature.
        f_deep_upsampled = F.interpolate(f_deep, size=f_shallow.shape[2:], mode='bilinear', align_corners=False)

        # Align the channel dimension of the upsampled deep feature.
        f_prime_deep = self.conv1x1_align(f_deep_upsampled)

        # Concatenate for the gating unit.
        gate_input = torch.cat([f_prime_deep, f_shallow], dim=1)

        # Generate the gating map G.
        g = self.gate_unit(gate_input)

        # Modulate the shallow feature with the gate and add to the deep feature.
        f_fused = f_prime_deep + (g * f_shallow)

        return f_fused