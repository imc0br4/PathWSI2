# models/zoo/cls_small.py
import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class ClsSmall(nn.Module):
    def __init__(self, in_ch=3, num_classes=6):
        super().__init__()
        self.stem = ConvBNReLU(in_ch, 32, 3, 2, 1)     # 112x112
        self.layer1 = nn.Sequential(
            ConvBNReLU(32, 64, 3, 2, 1),               # 56x56
            ConvBNReLU(64, 64)
        )
        self.layer2 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2, 1),              # 28x28
            ConvBNReLU(128, 128)
        )
        self.layer3 = nn.Sequential(
            ConvBNReLU(128, 256, 3, 2, 1),             # 14x14
            ConvBNReLU(256, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)              # [N, K]
        return x

def build_cls_small(num_classes=6):
    return ClsSmall(num_classes=num_classes)
