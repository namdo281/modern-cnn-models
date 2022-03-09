from multiprocessing import pool
from torch import relu
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .modules import *


class ImageNetNet(BaseModel):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

    def forward(self, x):
        return x
