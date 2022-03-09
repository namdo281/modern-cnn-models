import torch.nn as nn
import torch.nn.functional as F
from .BaseNet import ImageNetNet
from .modules import BottleNeckResidualBlock as Bnb


class MobileNetV2(ImageNetNet):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bnb = nn.Sequential(
            Bnb(32, 16, 1, 1),
            Bnb(16, 24, 6, 2),
            Bnb(24, 24, 6, 1),
            Bnb(24, 32, 6, 2),
            Bnb(32, 32, 6, 1),
            Bnb(32, 32, 6, 1),
            Bnb(32, 64, 6, 2),
            Bnb(64, 64, 6, 1),
            Bnb(64, 64, 6, 1),
            Bnb(64, 64, 6, 1),
            Bnb(64, 96, 6, 1),
            Bnb(96, 96, 6, 1),
            Bnb(96, 96, 6, 1),
            Bnb(96, 160, 6, 2),
            Bnb(160, 160, 6, 1),
            Bnb(160, 160, 6, 1),
            Bnb(160, 320, 6, 1),
        )
        self.conv2 = nn.Conv2d(320, 1280, (1, 1), stride=1)
        self.bn2 = nn.BatchNorm2d(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Conv2d(1280, num_classes, (1, 1))

        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.bnb(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_classes)
        x = F.log_softmax(x)
        return x
