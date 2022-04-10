from numpy import positive
from torch.utils.data import Dataset
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

    def __len__(self):
        return 20*self.cats
        
    def __getitem__(self, idx):
        # our positive class for the triplet
        # print("IDX IZ",idx)
        idx = str(idx%self.cats + 1)
        # print(idx)
        
        # choosing our pair of positive images (im1, im2)
        positives = os.listdir(os.path.join(self.path, idx))
        im1, im2 = random.sample(positives, 2)
        
        # choosing a negative class and negative image (im3)
        negative_cats = [str(x+1) for x in range(self.cats)]
        negative_cats.remove(idx)
        negative_cat = str(random.choice(negative_cats))
        negatives = os.listdir(os.path.join(self.img_dir, negative_cat))
        im3 = random.choice(negatives)
        
        previm1, previm2, previm3 = im1, im2, im3
        im1,im2,im3 = os.path.join(self.img_dir, idx, im1), os.path.join(self.img_dir, idx, im2), os.path.join(self.img_dir, negative_cat, im3)
        while (im1, im2, im3) in self.used:
          im1,im2,im3 = os.path.join(self.img_dir, idx, previm1), os.path.join(self.img_dir, idx, previm2), os.path.join(self.img_dir, negative_cat, previm3)
        self.used.add((im1,im2,im3))
        im1 = self.transforms(Image.open(im1))
        im2 = self.transforms(Image.open(im2))
        im3 = self.transforms(Image.open(im3))
        
        return [im1, im2, im3]