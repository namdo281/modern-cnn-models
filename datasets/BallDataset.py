import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from PIL import Image
import pandas as pd

class BallDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        super().__init__()
        self.root_dir = root_dir
        self.label_file = os.path.join(root_dir, "gt.csv")
        self.transform = transform

        label_df = pd.read_csv(self.label_file, index_col=0)
        print(label_df)
        self.label_df = label_df
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.tolist()
        path, label = self.label_df.loc[idx].to_numpy()
        print(path, label)
        path = os.path.join(self.root_dir, path)
        img = io.imread(path)
        if type(img) is np.ndarray:
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return (img, label)
    def __len__(self):
        return self.label_df.shape[0]


        



# bd = BallDataset('./data/balls/train')
# bd[50]