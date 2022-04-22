# Dataset reference: https://www.kaggle.com/competitions/dog-breed-identification/data?select=labels.csv
# Data prep ref: https://techvidvan.com/tutorials/dog-breed-classification/
# Other references: https://d2l.ai/chapter_computer-vision/kaggle-dog.html

import os
import pathlib
import random
import shutil
import json

import pandas as pd
from collections import OrderedDict
import json

random.seed(1)

path_to_downloaded_dataset = os.path.join("..", "..", "dog-breed-identification")
labels_csv = os.path.join(path_to_downloaded_dataset, "labels.csv")

# store training and testing images folder location
path_to_organized_dataset = os.path.join("..", "dataset", "dogs")
train_data_path = os.path.join(path_to_organized_dataset, "train")
test_data_path = os.path.join(path_to_organized_dataset, "test")

# specify number of breeds to use. Total number of breeds 120
num_breeds = 120
num_breeds_to_use = 30

# read the csv file
df_labels = pd.read_csv(labels_csv)

# check the total number of unique breed in our dataset file
print("Total number of unique Dog Breeds :", len(df_labels.breed.unique()))
print("Using {} Dog Breeds in current experiment".format(num_breeds))

breed_dict_desc_order_of_cnt = OrderedDict(df_labels['breed'].value_counts())
new_breed_list = list(breed_dict_desc_order_of_cnt.keys())[:num_breeds]

# change the dataset to have only those 60 unique breed records
df_labels = df_labels.query('breed in @new_breed_list')

# create new column which will contain image name with the image extension
df_labels['img_file'] = df_labels['id'].apply(lambda x: x + ".jpg")

breed_img_data = df_labels.groupby('breed')['img_file'].apply(list).to_dict()

# Organize the data
# Note: least number of images for a dog breede is 66. To make dataset balanced picking  66 images for each breed.
img_limit = len(breed_img_data[min(breed_img_data, key=lambda k: len(breed_img_data[k]))])

# img_limit = 30

with open('dogs_12066_accuracy.txt') as rfp:
    dogs_cat_accr = dict(json.load(rfp))


dog_breeds_to_use = set()
for breed in sorted(dogs_cat_accr.items(), key=lambda x: x[1])[:num_breeds_to_use]:
    dog_breeds_to_use.add(breed[0])


train_split_ratio = 0.8
breed_to_num = {}

breed_num = 0
for num, (breed, img_list) in enumerate(breed_img_data.items(), 1):
    if str(num) not in dog_breeds_to_use: continue
    breed_num += 1
    img_list = img_list[:img_limit]
    # Note: exist_ok -- deletes the directory if it already exists.
    train_dir, test_dir = os.path.join(train_data_path, str(breed_num)), os.path.join(test_data_path, str(breed_num))
    pathlib.Path(train_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True)
    random.shuffle(img_list)
    train_size = int(len(img_list) * train_split_ratio)
    breed_to_num[breed_num] = breed
    for i, img in enumerate(img_list):
        if i < train_size:
            shutil.copy2(os.path.join(path_to_downloaded_dataset, 'train', img), os.path.join(train_dir, img))
        else:
            shutil.copy2(os.path.join(path_to_downloaded_dataset, 'train', img), os.path.join(test_dir, img))

with open(os.path.join(path_to_organized_dataset, 'dogbreeds.json'), 'w') as fp:
    json.dump(breed_to_num, fp)