import torch.nn as nn
import torch.nn.functional as F
from .BaseNet import ImageNetNet


class GoogLeNet(ImageNetNet):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(in_channels, num_classes)
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=192,
            kernel_size=3,
            padding=1

        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.inception3a = Inception(
            in_channels=self.conv2.out_channels,
            c1=64,
            c2=(96, 128),
            c3=(16, 32),
            c4=32
        )
        self.inception3b = Inception(
            in_channels=self.inception3a.out_channels,
            c1=128,
            c2=(128, 192),
            c3=(32, 96),
            c4=64
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.inception4a = Inception(
            in_channels=self.inception3b.out_channels,
            c1=192,
            c2=(96, 208),
            c3=(16, 48),
            c4=64
        )
        self.inception4b = Inception(
            in_channels=self.inception4a.out_channels,
            c1=160,
            c2=(112, 224),
            c3=(24, 64),
            c4=64
        )
        self.inception4c = Inception(
            in_channels=self.inception4b.out_channels,
            c1=128,
            c2=(128, 256),
            c3=(24, 64),
            c4=64
        )
        self.inception4d = Inception(
            in_channels=self.inception4c.out_channels,
            c1=112,
            c2=(144, 288),
            c3=(32, 64),
            c4=64
        )
        self.inception4e = Inception(
            in_channels=self.inception4d.out_channels,
            c1=256,
            c2=(160, 320),
            c3=(32, 128),
            c4=128
        )
        self.pool4 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.inception5a = Inception(
            in_channels=self.inception4e.out_channels,
            c1=256,
            c2=(160, 320),
            c3=(32, 128),
            c4=128
        )
        self.inception5b = Inception(
            in_channels=self.inception5a.out_channels,
            c1=384,
            c2=(192, 384),
            c3=(48, 128),
            c4=128
        )
        self.pool5 = nn.AvgPool2d(
            kernel_size=7,
            stride=1
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        # print(x.shape)
        # print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # print(x.shape)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        # print(x.shape)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        # print(x.shape)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.pool5(x)
        # print(x.shape)
        # print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        # print(x.shape)
        x = F.log_softmax(x, dim=1)
        # print(x.shape)
        return x