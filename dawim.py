# 文件路径: models/my_block/dawim.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# 从父目录的 block/torch_wavelets 模块中导入DWT和IDWT功能
from ..block.torch_wavelets import DWT_2D, IDWT_2D


class SEWeightModule(nn.Module):
    """
    自适应加权模块: 结合了MaxPool和AvgPool的SENet.
    根据交互后的差异特征，为原始特征的每个通道生成一个权重。
    """
    # 类的构造函数，定义模块需要的网络层
    def __init__(self, channels, reduction=16):
        # 调用父类的构造函数，这是必须的
        super().__init__()
        # 定义一个自适应平均池化层，它能将任意大小的输入特征图池化成 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义一个自适应最大池化层，同样池化成 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 定义一个全连接网络(fc)，用 nn.Sequential 将多个层串联起来
        self.fc = nn.Sequential(
            # 第一个全连接层：输入维度是 channels * 2 (因为拼接了avg和max的结果)，输出维度是 channels // reduction，用于降维
            nn.Linear(channels * 2, channels // reduction, bias=False),
            # ReLU激活函数，增加非线性
            nn.ReLU(inplace=True),
            # 第二个全连接层：输入是降维后的维度，输出恢复到原始通道数 channels，用于升维
            nn.Linear(channels // reduction, channels, bias=False),
            # Sigmoid激活函数：将输出的权重值归一化到 (0, 1) 区间，使其可以作为注意力权重
            nn.Sigmoid()
        )

    # 模块的前向传播函数，定义数据如何流过这些层
    def forward(self, x):
        # 获取输入张量 x 的维度信息 (batch_size, channels, height, width)
        b, c, _, _ = x.size()
        # 对输入x进行平均池化，并用 .view(b, c) 将其形状从 [b, c, 1, 1] 变成 [b, c]
        avg_out = self.avg_pool(x).view(b, c)
        # 对输入x进行最大池化，同样改变形状
        max_out = self.max_pool(x).view(b, c)

        # 沿着通道维度(dim=1)将平均池化和最大池化的结果拼接起来，得到一个 [b, c*2] 的向量
        y = torch.cat([avg_out, max_out], dim=1)
        # 将拼接后的向量送入全连接网络fc，得到 [b, c] 的权重向量
        # 再用 .view(b, c, 1, 1) 将其形状变回 [b, c, 1, 1]，以便与原始特征图进行广播相乘
        y = self.fc(y).view(b, c, 1, 1)
        # 返回计算出的通道权重
        return y


class DAWIM(nn.Module):
    """
    Discrepancy-Aware Wavelet Interaction Module (DAWIM)

    这个模块实现了DAWIM的核心交互逻辑，针对单个尺度。
    输入：双时相特征 f1, f2
    输出：两个经过频率域交互和增强后，重建回空间域的特征图 f1_prime, f2_prime
    """

    # 类的构造函数
    def __init__(self, in_channels, dwt_wave='haar'):
        super().__init__()

        # 初始化离散小波变换(DWT)模块，使用'haar'小波
        self.dwt = DWT_2D(wave=dwt_wave)
        # 初始化逆离散小波变换(IDWT)模块
        self.idwt = IDWT_2D(wave=dwt_wave)

        # --- 1. 为不同频率分量定义的交互模块 ---
        # 定义用于低频(LL)交互的3D卷积层。kernel_size=(2,3,3)表示在时间维(2)和空间维(3x3)上同时卷积
        self.conv3d_ll = nn.Conv3d(in_channels, in_channels,kernel_size=(2, 3, 3), padding=(0, 1, 1), bias=False)
        #padding_Depth=0: 在时间维度上，不进行任何填充。这是因为输入的时间维度大小是2，卷积核大小也是2，卷积操作会直接将这个维度从2压缩到1，我们不希望改变这个行为。
        #为什么空间 padding 是1？这是一个经典的卷积尺寸计算问题。当卷积核大小为 K，填充为 P，步长为 S (默认为1)时，输出尺寸 O 和输入尺寸 I 的关系是：O = (I - K + 2*P) / S + 1。
        # 为了让输出尺寸 O 等于输入尺寸 I，在步长为1的情况下，需要满足 I = I - K + 2*P，简化后得到 P = (K - 1) / 2。
        # 在我们的空间维度上，卷积核大小 K=3。因此，为了保持尺寸不变，需要的填充 P = (3 - 1) / 2 = 1。
        # 所以，(padding_Height=1, padding_Width=1) 的设置，确保了经过 3x3 卷积后，特征图的 Height 和 Width 保持不变。'
        # 定义用于中频(LH, HL)交互的3D卷积层。kernel_size=(2,1,1)表示只在时间维(2)上进行卷积
        self.conv3d_lh_hl = nn.Conv3d(in_channels, in_channels,kernel_size=(2, 1, 1), padding=(0, 0, 0), bias=False)
        # 定义用于高频(HH)交互的2D卷积模块，包含1x1卷积、批归一化和ReLU激活
        self.conv2d_hh = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # --- 2. 实例化自适应加权SE模块 ---
        # 为每个频率分量(LL, LH, HL, HH)都创建一个独立的SEWeightModule实例
        self.se_ll = SEWeightModule(in_channels)
        self.se_lh = SEWeightModule(in_channels)
        self.se_hl = SEWeightModule(in_channels)
        self.se_hh = SEWeightModule(in_channels)

    # 模块的前向传播函数
    def forward(self, f1, f2):
        # --- DWT分解 ---
        # 对输入特征f1进行DWT，得到4个频率分量（拼接在通道维度）
        f1_dwt = self.dwt(f1)#DWT会将特征图分解为4个子带 (LL, LH, HL, HH)，每个子带的尺寸是原始的一半 (32x32)，通道数不变(64)。在 torch_wavelets 库中，这4个子带通常会被拼接在通道维度上。
        # 对输入特征f2进行DWT
        f2_dwt = self.dwt(f2)
        f1_dwt_ll, f1_dwt_lh, f1_dwt_hl, f1_dwt_hh = torch.chunk(f1_dwt, 4, dim=1)
        # 对f2_dwt做同样的操作
        f2_dwt_ll, f2_dwt_lh, f2_dwt_hl, f2_dwt_hh = torch.chunk(f2_dwt, 4, dim=1)
        # --- 1. 不同频率的交互 ---
        # 低频交互：用torch.stack在新的维度(dim=2)上堆叠f1和f2的LL分量，形成[B, C, 2, H, W]的5D张量
        # 然后送入3D卷积，并用.squeeze(2)移除时间维度，得到交互后的特征
        ll_interacted = self.conv3d_ll(torch.stack([f1_dwt_ll, f2_dwt_ll], dim=2)).squeeze(2)#torch.stack 会将一个张量列表（在这里是 [f1_dwt_ll, f2_dwt_ll]）沿着一个新插入的维度进行拼接。
        #torch.cat (concatenate) 是在已有的维度上进行拼接，它不会增加维度。例如 torch.cat(..., dim=1) 会使通道数增加。
        #我们指定了 dim=2，所以它会在第2个维度（从0开始数）的位置插入一个新维度，并将两个张量“堆”在这个新维度上。
        #squeeze() 函数的作用是移除张量中所有尺寸为1的维度。我们指定了 dim=2，所以它会精确地移除第2个维度，前提是这个维度的尺寸必须是1。
        # 中频LH交互：同上，但使用只在时间维卷积的conv3d_lh_hl
        lh_interacted = self.conv3d_lh_hl(torch.stack([f1_dwt_lh, f2_dwt_lh], dim=2)).squeeze(2)
        # 中频HL交互：同上
        hl_interacted = self.conv3d_lh_hl(torch.stack([f1_dwt_hl, f2_dwt_hl], dim=2)).squeeze(2)
        # 高频HH交互：先对f1和f2的HH分量逐元素求差并取绝对值，然后送入2D卷积模块处理
        hh_interacted = self.conv2d_hh(torch.abs(f1_dwt_hh - f2_dwt_hh))

        # --- 2. 自适应加权 ---
        # 将交互后的LL特征送入se_ll模块，得到LL分量的权重w_ll
        w_ll = self.se_ll(ll_interacted)

        # 同样的方法计算LH, HL, HH的权重
        w_lh = self.se_lh(lh_interacted)

        w_hl = self.se_hl(hl_interacted)

        w_hh = self.se_hh(hh_interacted)

        f1_ll_p = f1_dwt_ll * w_ll + f1_dwt_ll
        f2_ll_p = f2_dwt_ll * w_ll + f2_dwt_ll
        # ... 对所有分量和两个时相都进行同样的操作
        f1_lh_p = f1_dwt_lh * w_lh + f1_dwt_lh
        f2_lh_p = f2_dwt_lh * w_lh + f2_dwt_lh
        f1_hl_p = f1_dwt_hl * w_hl + f1_dwt_hl
        f2_hl_p = f2_dwt_hl * w_hl + f2_dwt_hl
        f1_hh_p = f1_dwt_hh * w_hh + f1_dwt_hh
        f2_hh_p = f2_dwt_hh * w_hh + f2_dwt_hh

        # --- 3. IDWT重建 ---
        # 将f1的所有加权后的频率分量(p代表prime)在通道维度上拼接回来
        # 然后送入IDWT模块，重建回空间域特征f1_prime
        f1_prime = self.idwt(torch.cat([f1_ll_p, f1_lh_p, f1_hl_p, f1_hh_p], dim=1))
        # 对f2做同样的操作
        f2_prime = self.idwt(torch.cat([f2_ll_p, f2_lh_p, f2_hl_p, f2_hh_p], dim=1))
        return f1_prime, f2_prime
