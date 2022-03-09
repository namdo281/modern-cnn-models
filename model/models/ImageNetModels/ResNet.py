import torch.nn as nn
import torch.nn.functional as F
from .BaseNet import ImageNetNet


class ResNet(ImageNetNet):
    def __init__(self, in_channels=3, num_classes=1000):
        super(ResNet, self).__init__(in_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.residual1 = ResidualBlock(64, 64, 2)
        self.residual2 = ResidualBlock(64, 128, 2)
        self.residual3 = ResidualBlock(128, 256, 2)
        self.residual4 = ResidualBlock(256, 512, 2)
        self.pool = nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        # print(x.shape)
        x = self.residual1(x)
        # print(x.shape)
        x = self.residual2(x)
        # print(x.shape)
        x = self.residual3(x)
        # print(x.shape)
        x = self.residual4(x)
        # print(x.shape)
        x = self.pool(x)
        # `print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        x = F.log_softmax(x, dim=1)
        return x
