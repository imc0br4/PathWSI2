import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )
    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.d1 = DoubleConv(in_ch, 32)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.b = DoubleConv(64, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c2 = DoubleConv(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c1 = DoubleConv(64, 32)
        self.head = nn.Conv2d(32, num_classes, 1)


    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        xb = self.b(self.p2(x2))
        x = self.u2(xb)
        x = self.c2(torch.cat([x, x2], dim=1))
        x = self.u1(x)
        x = self.c1(torch.cat([x, x1], dim=1))
        return self.head(x)