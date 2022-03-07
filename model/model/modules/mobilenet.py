import torch.nn as nn
import torch
import torch.nn.functional as F
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = None) -> None:
        super().__init__()
        if padding == None:
            padding = kernel_size // 2
        self.depth_wise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups= in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.point_wise_conv = nn.Conv2d(in_channels, out_channels, (1,1), 1, 0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.depth_wise_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.point_wise_conv(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x