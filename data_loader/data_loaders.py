from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from base import BaseDataLoader
import datasets
from torchvision import datasets as dts
from datasets import CatDogDataset, BallDataset
import os


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, resize = None, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if resize:
            trsfm = transforms.Compose([
                transforms.Resize(resize),
                trsfm
            ])
        self.data_dir = data_dir
        self.dataset = dts.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CatDogDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle = True, validation_split=0.0, num_workers = 1, training = True, resize = None):
        if training:
            self.data_dir = data_dir+"train/"
        else:
            self.data_dir = data_dir+"test/"
        self.label_file = os.path.join(self.data_dir, 'gt.txt')
        if resize:
            compose_transform = transforms.Compose([
                transforms.Resize(resize) ,
                transforms.ToTensor(),
                Normalize(0, 1)
            ])
        else:
            compose_transform = transforms.Compose([
                transforms.ToTensor(),
                Normalize(0,1)
            ])
        self.dataset = CatDogDataset(self.data_dir, self.label_file, compose_transform)
        #print(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BallDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle = True, validation_split=0.0, num_workers = 1, training = True, resize = None):
        class_dict_file = os.path.join(data_dir, "class_dict.csv")
        if training:
            self.data_dir = os.path.join(data_dir, "train/")
        else:
            self.data_dir = os.path.join(data_dir, "train/")
        
        self.class_dict_file = class_dict_file
        if resize:
            compose_transform = transforms.Compose([
                transforms.Resize(resize) ,
                transforms.ToTensor(),
                Normalize(0, 1)
            ])
        else:
            compose_transform = transforms.Compose([
                transforms.ToTensor(),
                Normalize(0,1)
            ])
        self.dataset = BallDataset(self.data_dir, self.class_dict_file, compose_transform)
        #print(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
