import torch
from torch.utils.data import Dataset
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from PIL import Image
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(os.listdir(self.root_dir))
    def __getitem__(self, idx):
        images = os.listdir(self.root_dir)
        images.sort()
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # print(idx)
        # print("images[x]")
        # print(images[idx])
        img_name = os.path.join(self.root_dir, images[idx])
        image = io.imread(img_name)
        if type(image) is np.ndarray:
            image = Image.fromarray(image)
        # print("shape")
        # print(image.shape)
        label = 1 if "cat" in img_name else 0
        sample = [image, label]

        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


# cdd = CatDogDataset('data/catdog/train/')
# #print(cdd[:10])
# fig = plt.figure()
# for i in range(len(cdd)):
#     sample = cdd[i]
#     print(i, sample['image'].shape, sample['label'])
#     ax = plt.subplot(1,4, i+1)
#     plt.tight_layout()
#     ax.set_title(sample['label'])
#     ax.axis('off')
#     plt.imshow(sample['image'])
#     if i == 3:
#         plt.show()
#         break

