import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .modules import *

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return x
