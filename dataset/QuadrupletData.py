import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import PIL
from PIL import Image
import random
import numpy as np

class QuadrupletData(Dataset):

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
        img_anchor = self.imageFolderDataset.imgs[index]
        # we need one similar class and two dissimilar classes
        negative_class = None

        # Select image from same class
        image_indices = list(self.image_indices_by_class[img_anchor[1]])
        image_indices.remove(index)
        idx = random.choice(image_indices)
        img_positive = Image.open(self.imageFolderDataset.imgs[idx])

        # Create list of other classes
        all_classes = list(self.image_indices_by_class.keys())
        all_classes.remove(img_anchor[1])

        # select a negative image
        class_idx = random.choice(all_classes)
        all_classes.remove(class_idx)
        neg_idx1 = random.choice(list(self.image_indices_by_class[class_idx]))
        img_negative1 = Image.open(self.imageFolderDataset.imgs[neg_idx1])

        # Get the second negative image
        class_idx = random.choice(all_classes)
        neg_idx2 = random.choice(list(self.image_indices_by_class[class_idx]))
        img_negative2 = Image.open(self.imageFolderDataset.imgs[neg_idx2])

        if self.should_invert:
            img_anchor = PIL.ImageOps.invert(img_anchor)
            img_positive = PIL.ImageOps.invert(img_positive)
            img_negative1 = PIL.ImageOps.invert(img_negative1)
            img_negative2 = PIL.ImageOps.invert(img_negative2)

        if self.transform is not None:
            img_anchor = self.transform(img_anchor)
            img_positive = self.transform(img_positive)
            img_negative1 = self.transform(img_negative1)
            img_negative2 = self.transform(img_negative2)

        # include label for contrastive loss, label = 1 if similar or 0 otherwise
        return img_anchor, img_positive, img_negative1, img_negative2
