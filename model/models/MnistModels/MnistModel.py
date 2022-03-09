from base import BaseModel


class MnistNet(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return x
