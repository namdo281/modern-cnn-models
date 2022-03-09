import torch.nn as nn
import torch.nn.functional as F
from .BaseNet import ImageNetNet
from .modules import DenseBlock, TransitionLayer, ConvLayer


class DenseNet(ImageNetNet):
    def __init__(self, k, in_channels=3, num_classes=1000):
        super().__init__(in_channels, num_classes)
        self.conv1 = ConvLayer(self.in_channels, k, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense1 = DenseBlock(k, k, 6)
        dense1_oc = self.dense1.out_channels
        self.transition1 = TransitionLayer(dense1_oc, dense1_oc // 2)

        self.dense2 = DenseBlock(dense1_oc // 2, k, 12)
        dense2_oc = self.dense2.out_channels
        self.transition2 = TransitionLayer(dense2_oc, dense2_oc // 2)

        self.dense3 = DenseBlock(dense2_oc // 2, k, 12)
        dense3_oc = self.dense3.out_channels
        self.transition3 = TransitionLayer(dense3_oc, dense3_oc // 2)

        self.dense4 = DenseBlock(dense3_oc // 2, k, 12)

        self.pool2 = nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(262, self.num_classes)

    def forward(self, x):
        # print("forward")
        # print(x.shape,1)
        x = self.conv1(x)
        # print(x.shape,2)
        x = self.pool1(x)
        # print(x.shape,3)
        x = self.dense1(x)
        # print(x.shape)
        x = self.transition1(x)
        # print(x.shape)
        x = self.dense2(x)
        # print(x.shape)
        x = self.transition2(x)
        # print(x.shape)
        x = self.dense3(x)
        # print(x.shape)
        x = self.transition3(x)
        # print(x.shape)
        x = self.dense4(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
