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
    def __init__(self, root_dir, class_dict_file, transform = None):
        super().__init__()
        self.root_dir = root_dir
        self.label_file = os.path.join(root_dir, "gt.csv")
        self.transform = transform
        self.class_dict_file = class_dict_file
        classes_df = pd.read_csv(class_dict_file, index_col=0)
        classes = classes_df["class"].to_list()
        # print(len(classes))
        # print(classes)
        self.encodeDict = dict()
        self.decodeDict = dict()
        for i, c in enumerate(classes):
            self.encodeDict[c] = i
            self.decodeDict[i] = c
        label_df = pd.read_csv(self.label_file, index_col=0)
        # print(label_df)
        self.label_df = label_df
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.tolist()
        path, label = self.label_df.loc[idx].to_numpy()
        label = self.encodeLabel(label)
        path = os.path.join(self.root_dir, path)
        img = io.imread(path)
        # print(img)
        plt.imshow(img)
        plt.show()
        if len(img.shape) == 2:
            img = img.reshape((img.shape[0], img.shape[1], 1))
            chs = []
            for i in range(3):
                chs.append(img)
            # print(img.shape)
            img = np.concatenate(chs, axis=2)
            # print(img.shape)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
            # print(img.shape)

        if type(img) is np.ndarray:
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return (img, label)
    def __len__(self):
        return self.label_df.shape[0]

    def encodeLabel(self, label):
        return self.encodeDict[label]
    def decodeLabel(self, num):
        return self.decodeDict[num]


        


# bd = BallDataset('data/balls/train/', 'data/balls/class_dict.csv')
# #print(cdd[:10])
# fig = plt.figure()
# random_nums = np.floor(np.random.rand(4)*len(bd))
# print(random_nums)
# for i, r in enumerate(random_nums):
#     sample = bd[r]
#     #print(i, sample[0].shape, sample[1])
#     ax = plt.subplot(1,4, i+1)
#     plt.tight_layout()
#     ax.set_title(bd.decodeLabel(sample[1]))
#     ax.axis('off')
#     plt.imshow(sample[0])
#     if i == 3:
#         plt.show()
#         break