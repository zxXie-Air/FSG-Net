import timm
import torch
import torch.nn as nn


class Efficientnetv2(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        if name.startswith("tf_efficientnetv2_s_in21k"):  
            self.extract = timm.create_model('tf_efficientnetv2_s_in21k', features_only=True,
                                             out_indices=(1, 2, 3, 4), pretrained=False)
        elif name.startswith("tf_efficientnetv2_s_in21ft1k"):
            self.extract = timm.create_model('tf_efficientnetv2_s_in21ft1k', features_only=True,
                                             out_indices=(1, 2, 3, 4), pretrained=pretrained)
        elif name.startswith("efficientnetv2_rw_s"):
            self.extract = timm.create_model('efficientnetv2_rw_s', features_only=True,
                                             out_indices=(1, 2, 3, 4), pretrained=pretrained)
        elif name.startswith("efficientnetv2_rw_m"):
            self.extract = timm.create_model('efficientnetv2_rw_m', features_only=True,
                                             out_indices=(1, 2, 3, 4), pretrained=pretrained)
        elif name.startswith("tf_efficientnetv2_l_in21ft1k"):
            self.extract = timm.create_model('tf_efficientnetv2_l_in21ft1k', features_only=True,
                                             out_indices=(1, 2, 3, 4), pretrained=pretrained)
        else:
            raise Exception("Error, please check the backbone name!")

        if pretrained:
            print("==> Load pretrained model for: {} successfully".format(name))

    def forward(self, x):
        f1, f2, f3, f4 = self.extract(x)

        return f1, f2, f3, f4

# class ResNet50(nn.Module):
#     def __init__(self, name, pretrained=True):
#         super().__init__()
#         if name.startswith("resnet50"):
#             # self.extract = timm.create_model('resnext101_32x4d', features_only=True,
#             #                                  out_indices=(1, 2, 3, 4), pretrained=False)
#             # self.extract = timm.create_model('resnet50', features_only=True,
#             #                                  out_indices=(1, 2, 3, 4), pretrained=False)
#             self.extract = timm.create_model('resnet18', features_only=True,
#                                               out_indices=(1, 2, 3, 4), pretrained=True)
#         else:
#             raise Exception("Error, please check the backbone name!")
#
#         if pretrained:
#             print("==> Load pretrained model for: {} successfully".format(name))
#
#     def forward(self, x):
#         f1, f2, f3, f4 = self.extract(x)
#
#         return f1, f2, f3, f4
# --- 在文件顶部确保导入了 timm ---

# --- 将类名从 ResNet50 改为 ResNet18_timm ---
class ResNet18_timm(nn.Module):
    def __init__(self, name, pretrained=True):  # name 参数暂时保留，虽然现在没用了
        super().__init__()
        # 直接加载 resnet18，不再需要 if 判断
        self.extract = timm.create_model('resnet18', features_only=True,
                                         out_indices=(1, 2, 3, 4), pretrained=True)

        if pretrained:
            print("==> Load pretrained model for: resnet18 (from timm) successfully")

    def forward(self, x):
        f1, f2, f3, f4 = self.extract(x)
        return f1, f2, f3, f4


class ResNet_timm(nn.Module):
    """
    一个通用的 ResNet 加载器，可以根据传入的 name 参数
    从 timm 库加载任何 ResNet 变体 (如 'resnet18', 'resnet34', 'resnet50' 等)。
    """

    def __init__(self, name='resnet18', pretrained=True):
        super().__init__()

        # 核心修改：不再硬编码 'resnet18'，而是使用传入的 name 参数
        self.extract = timm.create_model(
            model_name=name,  # 使用动态的模型名称
            features_only=True,  # 只返回特征图，不包括最后的分类头
            out_indices=(1, 2, 3, 4),  # 返回第1,2,3,4个阶段的输出
            pretrained=pretrained
        )

        if pretrained:
            print(f"==> Loaded pretrained backbone: {name} from timm successfully")

    def forward(self, x):
        # timm 的 features_only=True 会返回一个特征图列表
        features = self.extract(x)
        # 根据你的代码，你的模型需要4个输出
        # ResNet的4个阶段刚好对应4个输出
        f1, f2, f3, f4 = features
        return f1, f2, f3, f4

if __name__ == "__main__":
    # model_names = timm.list_models("*eff*v2*s*", pretrained=True)
    # for name in model_names:
    #     print(name)
    model = Efficientnetv2('efficientnetv2_rw_s')
    f1, f2, f3, f4 = model(torch.randn(2, 3, 512, 512))
    for x in (f1, f2, f3, f4):
        print(x.shape)
