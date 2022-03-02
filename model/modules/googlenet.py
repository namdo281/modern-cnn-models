import torch
from torch import nn
import torch.nn.functional as F
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels= in_channels,
            out_channels= c1,
            kernel_size=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels= c2[0],
            kernel_size=1,            
        )
        self.conv3 = nn.Conv2d(
            in_channels = c2[0],
            out_channels=c2[1],
            kernel_size=3,
            padding=1            
        )
        self.conv4 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c3[0],
            kernel_size=1,
        )
        self.conv5 = nn.Conv2d(
            in_channels=c3[0],
            out_channels=c3[1],
            kernel_size=5,
            padding=2
        )
        self.pool = nn.MaxPool2d(
            padding=1,
            kernel_size=3,
            stride = 1
        )
        self.conv6 = nn.Conv2d(
            in_channels=in_channels,
            kernel_size=1,
            out_channels=c4
        )
        self.out_channels = c1 + c2[1]+ c3[1] +c4


    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.conv2(x)
        x2 = F.relu(x2)
        x2 = self.conv3(x2)
        x2 = F.relu(x2)

        x3 = self.conv4(x)
        x3 = F.relu(x3)
        x3 = self.conv5(x3)
        x3 = F.relu(x3)
        
        x4 = self.pool(x)
        x4 = self.conv6(x4)
        x4 = F.relu(x4)

        return torch.concat([x1, x2, x3, x4], axis = 1)
