from base import BaseModel
from model.models.ImageNetModels.modules import *


class CatDogNet(BaseModel):
    def __init__(self, in_channels=3, num_classes=26):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(40000, 1000)
        self.linear2 = nn.Linear(1000, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x
