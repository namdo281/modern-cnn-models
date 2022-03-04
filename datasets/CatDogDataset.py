import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from PIL import Image
class CatDogDataset(Dataset):
    def __init__(self, root_dir, label_file, transform = None):
        self.root_dir = root_dir
        self.label_file = label_file
        print(label_file)
        with open(label_file, "r") as f:
            labels = f.read().split('\n')
            for i, l in enumerate(labels):
                l_split = l.split("\t")
                labels[i] = {l_split[0]: l_split[1]}
            #labels = [dict(l.split("\t")) for l in labels]
            self.labels = labels
            #print(labels)
        self.transform = transform
    def __len__(self):
        return len(glob.glob(f"{self.root_dir}*.jpg"))
    def __getitem__(self, idx):
        images = glob.glob(f"{self.root_dir}*.jpg")
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # print(idx)
        # print("images[x]")
        # print(images[idx])
        img_name = images[idx]
        image = io.imread(img_name)
        
        if type(image) is np.ndarray:
            image = Image.fromarray(image)

        # print("shape")
        # print(image.shape)
        label = 1 if "Cat" in img_name else 0
        sample = [image, label]
        
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample



# cdd = CatDogDataset('data/catdog/train/', 'data/catdog/train/gt.txt')
# #print(cdd[:10])
# fig = plt.figure()
# for i in range(len(cdd)):
#     sample = cdd[i]
#     #print(i, sample[0].shape, sample[1])
#     ax = plt.subplot(1,4, i+1)
#     plt.tight_layout()
#     ax.set_title(sample[1])
#     ax.axis('off')
#     plt.imshow(sample[0])
#     if i == 3:
#         plt.show()
#         break

