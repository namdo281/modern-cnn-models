import torch.nn as nn
import torch
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depth_wise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.point_wise_conv = nn.Conv2d(in_channels, out_channels, (1, 1), 1, 0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.depth_wise_conv(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.point_wise_conv(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x


class BottleNeckResidualBlock(nn.Module):
    def __init__(self, inc, ouc, ef, stride):
        super().__init__()
        hic = ef * inc
        self.conv = nn.Sequential(
            nn.Conv2d(inc, hic, (1, 1), 1),
            nn.BatchNorm2d(hic),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hic, hic, (3, 3), stride, padding=1, groups=hic),
            nn.BatchNorm2d(hic),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hic, ouc, (1, 1), 1)
        )
        self.id_conv = nn.Conv2d(inc, ouc, (1, 1), stride)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.id_conv(x)
        return x1 + x2

