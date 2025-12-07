# models/zoo/cls_res.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# 兼容新老 torchvision：新版本用 ResNet50_Weights，老版本还在用 pretrained=True
try:
    from torchvision.models import ResNet50_Weights
except Exception:
    ResNet50_Weights = None


class UpBlock(nn.Module):
    def __init__(self, in_channels, bridge_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + bridge_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, bridge):
        x = self.up(x)
        if x.size() != bridge.size():
            x = F.interpolate(x, size=bridge.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, bridge], dim=1)
        return self.conv(x)


class ResUNet50Classifier(nn.Module):
    """
    带 U-Net 解码器的分类 Head：输入 [N,3,H,W] 输出 [N,num_classes]
    默认不加载 torchvision 预训练（医生端离线更稳），如需可在 build() 里传 pretrained=True
    """
    def __init__(self, num_classes: int, use_pretrained_backbone: bool = False):
        super().__init__()

        # ---- 构建 ResNet50 主干（默认不加载预训练，避免联网） ----
        if ResNet50_Weights is not None:
            weights = ResNet50_Weights.DEFAULT if use_pretrained_backbone else None
            base_model = resnet50(weights=weights)
        else:
            base_model = resnet50(pretrained=bool(use_pretrained_backbone))

        self.initial = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.encoder1 = base_model.layer1  # 256
        self.encoder2 = base_model.layer2  # 512
        self.encoder3 = base_model.layer3  # 1024
        self.encoder4 = base_model.layer4  # 2048

        self.center = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.up4 = UpBlock(1024, 1024, 512)
        self.up3 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up1 = UpBlock(128, 64, 64)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x0 = self.initial(x)     # [B, 64,  H/4,  W/4]
        x1 = self.encoder1(x0)   # [B, 256, H/4,  W/4]
        x2 = self.encoder2(x1)   # [B, 512, H/8,  W/8]
        x3 = self.encoder3(x2)   # [B, 1024,H/16, W/16]
        x4 = self.encoder4(x3)   # [B, 2048,H/32, W/32]

        center = self.center(x4)

        d4 = self.up4(center, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)

        out = self.classifier(d1)  # [B, num_classes]
        return out


# ---- 供 manager 调用的构造器（关键）----
def build_cls_res(num_classes: int = 6, pretrained_backbone: bool = False) -> nn.Module:
    return ResUNet50Classifier(num_classes=num_classes, use_pretrained_backbone=pretrained_backbone)
