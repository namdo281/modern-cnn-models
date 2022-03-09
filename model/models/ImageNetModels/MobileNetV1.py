import torch.nn as nn
import torch.nn.functional as F
from .BaseNet import ImageNetNet
from .modules import DepthwiseSeparableConv


class MobileNetV1(ImageNetNet):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.convs = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 256),
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 512, stride=2),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # print(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.convs(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

