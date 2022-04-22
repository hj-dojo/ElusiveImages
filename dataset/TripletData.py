from torch.utils.data import Dataset
import torchvision.datasets as datasets
import os
from PIL import Image
import random


class TripletData(Dataset):
    def __init__(self, img_dir, transforms, cats=17, split="train", path="dataset/flowers/train"):
        self.img_dir = img_dir
        self.split = split
        self.cats = cats
        self.transforms = transforms
        self.path = path
        self.used = set()
        self.imageFolderDataset = datasets.ImageFolder(root=img_dir)
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

    def __get_triplet(self, idx):
        # our positive class & image for the triplet
        img1_tuple = self.imageFolderDataset.imgs[idx]

        # Retrieve image by class
        pos_indices = list(self.image_indices_by_class[img1_tuple[1]])
        pos_indices.remove(idx)
        pos_idx_selected = random.choice(pos_indices)
        img2_tuple = self.imageFolderDataset.imgs[pos_idx_selected]

        # Create list of other classes
        all_classes = list(self.image_indices_by_class.keys())
        all_classes.remove(img1_tuple[1])

        # select a negative image
        neg_class_idx = random.choice(all_classes)
        neg_idx_selected = random.choice(list(self.image_indices_by_class[neg_class_idx]))
        img3_tuple = self.imageFolderDataset.imgs[neg_idx_selected]

        return img1_tuple, img2_tuple, img3_tuple

    def __getitem__(self, idx):

        # Retrieve set of two positive and one negative image
        img1_tuple, img2_tuple, img3_tuple = self.__get_triplet(idx)

        max_tries = 50
        i = 0
        while (img1_tuple[0], img2_tuple[0], img3_tuple[0]) in self.used and i <= max_tries:
            if i == max_tries:
                print("Couldn't find new triplet")
            else:
                img1_tuple, img2_tuple, img3_tuple = self.__get_triplet(idx)
            i += 1

        self.used.add((img1_tuple[0], img2_tuple[0], img3_tuple[0]))
        im1 = self.transforms(Image.open(img1_tuple[0]))
        im2 = self.transforms(Image.open(img2_tuple[0]))
        im3 = self.transforms(Image.open(img3_tuple[0]))
        
        return [im1, im2, im3]