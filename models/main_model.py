from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.efficientnetv2_timm import Efficientnetv2, ResNet_timm
from .block.Base import ChannelChecker
from .head.FCN import FCNHead
from .my_block.lgfu import LGFU
from .neck.my_neck import MyNeck


class ChangeDetection(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = 64
        self.dl = opt.dual_label

        self._create_backbone(opt.backbone)
        self._create_neck(opt.neck)

        self.lgfu_3_to_2 = LGFU(self.inplanes * 4, self.inplanes * 2)
        self.lgfu_2_to_1 = LGFU(self.inplanes * 2, self.inplanes)

        self._create_heads(opt.head)
        self.check_channels = ChannelChecker(self.backbone, self.inplanes, opt.input_size)

        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)

    def forward(self, xa, xb, tta=False):
        return self.forward_once(xa, xb)

    def forward_once(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, fa4 = self.backbone(xa)
        fa1, fa2, fa3, fa4 = self.check_channels(fa1, fa2, fa3, fa4)
        fb1, fb2, fb3, fb4 = self.backbone(xb)
        fb1, fb2, fb3, fb4 = self.check_channels(fb1, fb2, fb3, fb4)

        ms_feats_raw = (fa1, fa2, fa3, fb1, fb2, fb3)
        fa1_s, fa2_s, fa3_s, fb1_s, fb2_s, fb3_s = self.neck(ms_feats_raw)

        diff3 = torch.abs(fa3_s - fb3_s)
        diff2 = torch.abs(fa2_s - fb2_s)
        diff1 = torch.abs(fa1_s - fb1_s)

        fused_2 = self.lgfu_3_to_2(f_deep=diff3, f_shallow=diff2)
        change_feature_map = self.lgfu_2_to_1(f_deep=fused_2, f_shallow=diff1)

        out1 = F.interpolate(
            self.head1(change_feature_map),
            size=(h_input, w_input),
            mode="bilinear",
            align_corners=True,
        )
        out2 = (
            F.interpolate(
                self.head2(change_feature_map),
                size=(h_input, w_input),
                mode="bilinear",
                align_corners=True,
            )
            if self.dl
            else None
        )

        return (out1, out2) if self.dl else out1

    def _init_weight(self, pretrain=""):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        pretrained_dict = torch.load(pretrain)
        if isinstance(pretrained_dict, nn.DataParallel):
            pretrained_dict = pretrained_dict.module
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(OrderedDict(model_dict), strict=True)
        print(f"=> ChangeDetection load {len(pretrained_dict)}/{len(model_dict)} items from: {pretrain}")

    def _create_backbone(self, backbone_name):
        if "resnet" in backbone_name:
            self.backbone = ResNet_timm(name=backbone_name, pretrained=True)
        elif "efficientnet" in backbone_name:
            self.backbone = Efficientnetv2(backbone_name)
        else:
            raise NotImplementedError(f"Backbone not implemented yet: {backbone_name}")

    def _create_neck(self, neck):
        self.neck = MyNeck(self.inplanes, neck)

    def _select_head(self, head):
        if head == "fcn":
            return FCNHead(self.inplanes, 2)
        raise NotImplementedError(f"Head not implemented yet: {head}")

    def _create_heads(self, head):
        self.head1 = self._select_head(head)
        self.head2 = self._select_head(head) if self.dl else None
