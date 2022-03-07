from torch import nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv = num_conv
        self.use_id_conv = in_channels != out_channels
        self.bn = nn.BatchNorm2d(out_channels)
        convs = []
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride = 2
        )
        for i in range(num_conv):
            if i == 0:
                conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride = 1,
                    padding = 1
                    )
            else:
                conv = nn.Conv2d(
                    out_channels, 
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            convs.append(conv)
            convs.append(nn.BatchNorm2d(out_channels))
            if i != num_conv -1:
                convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)
        if self.use_id_conv:
            self.id_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
    def forward(self, x):
        x = self.pool(x)
        x1 = self.convs(x)
        if self.use_id_conv:
            x2 = self.id_conv(x)    
        else:
            x2 = x
        # print(x1.shape)
        # print(x2.shape)
        return F.relu(x1+x2)

