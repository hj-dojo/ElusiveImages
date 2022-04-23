import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import PIL
from PIL import Image
import random
import numpy as np


class ContrastiveData(Dataset):

    def __init__(self, img_dir, transforms=None, should_invert=False):
        self.imageFolderDataset = datasets.ImageFolder(root=img_dir)
        self.transform = transforms
        self.should_invert = should_invert
        self.__get_classes()

    def __len__(self):
        return len(self.imageFolderDataset)

    def __get_classes(self):
        # create hashmap of classes to image tuples
        self.image_indices_by_class = {}
        for idx in range(len(self.imageFolderDataset)):
            img_tuple = self.imageFolderDataset.imgs[idx]
            if img_tuple[1] in self.image_indices_by_class:
                self.image_indices_by_class[img_tuple[1]].append(idx)
            else:
                self.image_indices_by_class[img_tuple[1]] = [idx]

    def __getitem__(self, index):
        # Retrieve image by index
        img0_tuple = self.imageFolderDataset.imgs[index]
        # we need to make sure approx 50% of images are in the same class
        maintain_same_class = random.randint(0, 1)
        if maintain_same_class:
            image_indices = list(self.image_indices_by_class[img0_tuple[1]])
            image_indices.remove(index)
            img1_idx = random.choice(image_indices)
            img1_tuple = self.imageFolderDataset.imgs[img1_idx]
        else:
            img1_class = None
            while True:
                # keep looping till a different class image is found
                img1_class = random.choice(list(self.image_indices_by_class.keys()))
                if img0_tuple[1] != img1_class:
                    break
            img1_idx = random.choice(self.image_indices_by_class[img1_class])
            img1_tuple = self.imageFolderDataset.imgs[img1_idx]

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # include label for contrastive loss, label = 0 if similar or 1 otherwise
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
