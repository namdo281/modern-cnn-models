import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .modules import *

class ImageNet(BaseModel):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
    def forward(self, x):
        return x


class AlexNet(ImageNet):
    def __init__ (self, in_channels = 1, num_classes=10):
        super().__init__(in_channels,num_classes)
        self.conv1 = nn.Conv2d(
            in_channels= self.in_channels,
            out_channels=96,
            stride= 4,
            kernel_size= 11,
        ) 
        self.pool1 = nn.MaxPool2d(
            kernel_size= 3,
            stride = 2
        )
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=3,
            stride = 2
        )
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size= 3, 
            stride=2
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(9600, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, self.num_classes)
    def forward(self, x):
        #print(x.shape)
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        #print(x.shape)
        output = F.log_softmax(x, dim=1)
        #print(output.shape)
        return output


class VGGNet(ImageNet):
    def __init__ (self, in_channels = 3, num_classes = 26):
        super().__init__(in_channels, num_classes)
        self.conv1 = nn.Conv2d(
            in_channels= self.in_channels,
            out_channels=96,
            stride= 4,
            kernel_size= 11,
        ) 
        self.pool1 = nn.MaxPool2d(
            kernel_size= 3,
            stride = 2
        )
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=3,
            stride = 2
        )
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size= 3, 
            stride=2
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(9600, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        #print(x.shape)
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        #print(x.shape)
        output = F.log_softmax(x, dim=1)
        #print(output.shape)
        return output

class GoogLeNet(ImageNet):
    def __init__(self, in_channels = 3, num_classes=10):
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
            stride = 2,
            padding = 1
        )
        self.conv2 = nn.Conv2d(
            in_channels= self.conv1.out_channels,
            out_channels=192,
            kernel_size=3,
            padding=1

        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=3, 
            stride = 2,
            padding = 1
        )
        self.inception3a = Inception(
            in_channels = self.conv2.out_channels,
            c1 = 64,
            c2 = (96, 128),
            c3 = (16, 32),
            c4 = 32
        )
        self.inception3b = Inception(
            in_channels = self.inception3a.out_channels,
            c1 = 128,
            c2 = (128, 192),
            c3 = (32, 96),
            c4 = 64
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size=3,
            stride = 2,
            padding = 1
        )
        self.inception4a = Inception(
            in_channels= self.inception3b.out_channels,
            c1 = 192,
            c2 = (96, 208),
            c3 = (16, 48),
            c4 = 64
        )
        self.inception4b = Inception(
            in_channels= self.inception4a.out_channels,
            c1 = 160,
            c2 = (112, 224),
            c3 = (24, 64),
            c4 = 64
        )
        self.inception4c = Inception(
            in_channels = self.inception4b.out_channels,
            c1 = 128,
            c2 = (128, 256),
            c3 = (24, 64),
            c4 = 64
        )
        self.inception4d = Inception(
            in_channels = self.inception4c.out_channels,
            c1 = 112,
            c2 = (144, 288),
            c3 = (32, 64),
            c4 = 64
        )
        self.inception4e = Inception(
            in_channels = self.inception4d.out_channels,
            c1 = 256,
            c2 = (160, 320),
            c3 = (32, 128),
            c4 = 128
        )
        self.pool4 = nn.MaxPool2d(
            kernel_size=3,
            stride = 2,
            padding = 1
        )
        self.inception5a = Inception(
            in_channels = self.inception4e.out_channels,
            c1 = 256, 
            c2 = (160, 320),
            c3 = (32, 128),
            c4 = 128
        )
        self.inception5b = Inception(
            in_channels= self.inception5a.out_channels,
            c1 = 384,
            c2 = (192, 384),
            c3 = (48, 128),
            c4 = 128
        )
        self.pool5 = nn.AvgPool2d(
            kernel_size= 7,
            stride = 1
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024, self.num_classes)
    
    def forward(self, x):
        #print(x.shape)
        # print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        #print(x.shape)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        #print(x.shape)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        #print(x.shape)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.pool5(x)
        #print(x.shape)
        # print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        # print(x.shape)
        x = F.log_softmax(x, dim = 1)
        # print(x.shape)
        return x

class ResNet(ImageNet):
    def __init__(self, in_channels=3, num_classes = 1000):
        super(ResNet, self).__init__(in_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride = 2 , padding= 3 )
        self.bn = nn.BatchNorm2d(64)
        self.residual1 = ResidualBlock(64, 64, 2)
        self.residual2 = ResidualBlock(64, 128, 2)
        self.residual3 = ResidualBlock(128, 256, 2)
        self.residual4 = ResidualBlock(256, 512, 2)
        self.pool = nn.AvgPool2d(kernel_size= 7)
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
        #x = self.dropout(x)
        #x = F.relu(x)
        x = F.log_softmax(x, dim=1)
        return x

class DenseNet(ImageNet):
    def __init__(self, k, in_channels=3, num_classes = 1000):
        super().__init__(in_channels, num_classes)
        self.conv1 = ConvLayer(self.in_channels, k, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)


        self.dense1 = DenseBlock(k, k, 6)
        dense1_oc = self.dense1.out_channels
        self.transition1 = TransitionLayer(dense1_oc, dense1_oc//2)
        
        
        self.dense2 = DenseBlock(dense1_oc // 2, k, 12)
        dense2_oc = self.dense2.out_channels
        self.transition2 = TransitionLayer(dense2_oc, dense2_oc//2)

        self.dense3 = DenseBlock(dense2_oc // 2, k, 12)
        dense3_oc = self.dense3.out_channels
        self.transition3 = TransitionLayer(dense3_oc, dense3_oc//2)

        self.dense4 = DenseBlock(dense3_oc // 2, k, 12)

        self.pool2 = nn.AvgPool2d(kernel_size= 7)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(262, self.num_classes)

    def forward(self, x):
        #print("forward")
        #print(x.shape,1)
        x = self.conv1(x)
        #print(x.shape,2)
        x = self.pool1(x)
        #print(x.shape,3)
        x = self.dense1(x)
        #print(x.shape)
        x = self.transition1(x)
        #print(x.shape)
        x = self.dense2(x)
        #print(x.shape)
        x = self.transition2(x)
        #print(x.shape)
        x = self.dense3(x)
        #print(x.shape)
        x = self.transition3(x)
        #print(x.shape)
        x = self.dense4(x)
        #print(x.shape)
        x = self.pool2(x)
        #print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim = 1)
        return x


class MobileNet(ImageNet):
    def __init__(self, in_channels = 3, num_classes = 1000, alpha=1 , rho=1):
        super().__init__(in_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, 32*alpha, kernel_size= 3, padding= 1, stride = 1 )
        self.alpha = alpha
        self.rho = rho
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.ds_conv1 = DepthwiseSeparableConv(self.conv1.out_channels, 64*alpha)
        self.ds_conv2 = DepthwiseSeparableConv(self.ds_conv1.out_channels, 128*alpha, stride = 2)
        self.ds_conv3 = DepthwiseSeparableConv(self.ds_conv2.out_channels, 128*alpha)
        self.ds_conv4 = DepthwiseSeparableConv(self.ds_conv3.out_channels, 256*alpha)
        self.ds_conv5 = DepthwiseSeparableConv(self.ds_conv4.out_channels, 256*alpha)
        self.ds_conv6 = DepthwiseSeparableConv(self.ds_conv5.out_channels, 512*alpha, stride = 2)
        ds_conv7891011 = []
        for i in range(7, 12):
            conv = DepthwiseSeparableConv(512*alpha, 512*alpha)
            ds_conv7891011.append(conv)
        self.ds_conv7891011 = nn.Sequential(*ds_conv7891011)
        self.ds_conv12 = DepthwiseSeparableConv(512*alpha, 1024*alpha)
        self.ds_conv13 = DepthwiseSeparableConv(1024*alpha, 1024*alpha, stride= 2)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.interpolate(x, self.rho*x.shape[2:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.ds_conv1(x)
        x = self.ds_conv2(x)
        x = self.ds_conv3(x)
        x = self.ds_conv4(x)
        x = self.ds_conv5(x)
        x = self.ds_conv6(x)
        x = self.ds_conv7891011(x)
        x = self.ds_conv12(x)
        x = self.ds_conv13(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim = 1)
        return x


