import torch.nn as nn
from models.block.Drop import DropBlock
# 从我们创建的 my_block 包中导入 DAWIM 和 STSAM 模块
from ..my_block.dawim import DAWIM
from ..my_block.stsam import STSAM

class MyNeck(nn.Module):
    def __init__(self, inplanes, neck_name=''):
        super().__init__()
        # inplanes = 64 (来自 main_model.py)

        # --- Instantiate Interaction Modules per Scale ---

        # Scale 1 (from backbone stage 1, 1/4 size, channels=64)
        self.dawim_s1 = DAWIM(in_channels=inplanes)
        self.stsam_s1 = STSAM(channels=inplanes)

        # Scale 2 (from backbone stage 2, 1/8 size, channels=128)
        self.dawim_s2 = DAWIM(in_channels=inplanes * 2)
        self.stsam_s2 = STSAM(channels=inplanes * 2)

        # Scale 3 (from backbone stage 3, 1/16 size, channels=256)
        self.dawim_s3 = DAWIM(in_channels=inplanes * 4)
        self.stsam_s3 = STSAM(channels=inplanes * 4)
        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = lambda x: x  # 如果不用，就定义一个什么都不做的 lambda 函数

    def forward(self, ms_feats):
        # 1. 解包从骨干网络传来的多尺度特征
        fa1, fa2, fa3, fb1, fb2, fb3 = ms_feats

        fa1, fa2, fa3, fb1, fb2, fb3 = self.drop([fa1, fa2, fa3,  fb1, fb2, fb3])
        # --- 2. 在每个尺度上依次应用 DAWIM 和 STSAM ---

        # 处理 stage 1
        fa1_d, fb1_d = self.dawim_s1(fa1, fb1)
        fa1_final, fb1_final = self.stsam_s1(fa1_d, fb1_d)

        # 处理 stage 2
        fa2_d, fb2_d = self.dawim_s2(fa2, fb2)
        fa2_final, fb2_final = self.stsam_s2(fa2_d, fb2_d)

        # 处理 stage 3
        fa3_d, fb3_d = self.dawim_s3(fa3, fb3)
        fa3_final, fb3_final = self.stsam_s3(fa3_d, fb3_d)

        # --- 3. 返回所有增强后的双时相特征图 ---
        # 后续模块将接收这个包含6个张量的元组，并执行自定义的融合策略。
        return fa1_final, fa2_final, fa3_final, fb1_final, fb2_final, fb3_final