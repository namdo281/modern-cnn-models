import torch
from torch import nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding)
    def forward(self, x):
        #print(x.shape)
        x = self.conv(self.relu(self.bn(x)))
        #print(x.shape)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=1, padding = 0)
        self.pool = nn.AvgPool2d(
            kernel_size= 2,
            stride = 2
        )
    def forward(self, x):
        #print(x.shape)
        #print(1)
        x = self.conv(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape) 
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_block):
        super().__init__()
        convs = []
        for i in range(num_block):
            if i == 0:
                conv1 = ConvLayer(in_channels, out_channels, kernel_size=1, padding = 0)
            else: 
                conv1 = ConvLayer(out_channels, out_channels, kernel_size=1, padding = 0)
            
            conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, padding=1)
            convs.append(nn.Sequential(*[conv1, conv2]))
        self.convs = convs
        self.out_channels = in_channels + out_channels*num_block

    def forward(self, x):
        xs = [x]
        for c in self.convs:
            #print(c)
            x = c(x)
            xs.append(x)
        return torch.concat(xs, axis = 1)
