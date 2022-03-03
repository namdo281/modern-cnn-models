from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from base import BaseDataLoader
import datasets
from datasets import CatDogDataset

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CatDogDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle = True, validation_split=0.0, num_workers = 1, training = True):
        self.data_dir = data_dir
        compose_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Normalize(0, 1)
        ])
        self.dataset = CatDogDataset(data_dir, compose_transform)
        print(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)