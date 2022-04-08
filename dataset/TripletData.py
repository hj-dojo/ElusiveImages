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

    def __len__(self):
        return 8*self.cats
        
    def __getitem__(self, idx):
        idx = str(idx//80 + 1)

        positives = os.listdir(os.path.join(self.img_dir, idx))
        im1, im2 = random.sample(positives, 2)

        neg_cats = [str(x+1) for x in range(self.cats)]
        neg_cats.remove(idx)
        neg_cat = str(random.choice(neg_cats))
        negs = os.listdir(os.path.join(self.img_dir, neg_cat))
        im3 = random.choice(negs)

        im1, im2, im3 = os.path.join(self.img_dir, idx, im1), os.path.join(self.img_dir, idx, im2), os.path.join(self.path, neg_cat, im3)

        im1 = self.transforms(Image.open(im1))
        im2 = self.transforms(Image.open(im2))
        im3 = self.transforms(Image.open(im3))

        return [im1, im2, im3]