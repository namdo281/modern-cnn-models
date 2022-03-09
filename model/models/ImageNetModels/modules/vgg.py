from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize
class  VGGBlock(nn.Module):
    def __init__(self, conv_arch):
        super().__init__()
        vm = []
        in_channels = 1
        for (num_conv, num_channels) in conv_arch:
            #in_channels = 1
            print(in_channels)
            for j in range(num_conv):
                vm.append(  
                    nn.Conv2d(
                        in_channels= in_channels,
                        out_channels=num_channels,
                        kernel_size=3,
                        padding = 1
                    ),
                    
                )
                vm.append(nn.ReLU())
                in_channels = num_channels
            vm.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vm = nn.Sequential(*vm)
    def forward(self, x):
        x = self.vm(x)
        return x
