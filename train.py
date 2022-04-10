import argparse
from pickletools import optimize
from tqdm.notebook import tqdm
from numpy import argsort
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import resnet32
from dataset import TripletData
from loss import TripletLoss
from utils import compute_map
import torchvision.transforms as transforms
import torchvision.models as tvmodels
import faiss
import glob
import numpy as np
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/SimpleNetwork.yaml')

def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    if args.model == 'ResNet':
        # model = resnet32()
        model = tvmodels.resnet18()
    if args.dataset == 'TripletData':
        train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        val_dataset = TripletData(args.data+'/test', val_transforms, path="dataset/flowers/train")
        train_dataset = TripletData(args.data+'/train', train_transforms, path="dataset/flowers/train")
        print(len(train_dataset))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    if torch.cuda.is_available:
        model = model.cuda()
    
    if args.loss_type == 'TripletLoss':
        criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    for epoch in range(args.epochs):
        loss = 0.0
        model.train()
        loss = train(epoch, train_loader, model, optimizer, criterion, loss)
        # acc, cm = validate(epoch, val_loader, model, criterion)
    print('NOW WE VALIDate')
    valid(val_transforms, model)
def train(epoch, loader, model, opt, crit, prevloss):
    print("Start training")
    for data in tqdm(loader):
        # data, target = d
        # if torch.cuda.is_available():
        #     print(data)
        #     data = data.cuda()
        #     target = target.cuda()        
        opt.zero_grad()
        x1, x2, x3 = data
        e1, e2, e3 = model(x1.to('cuda')), model(x2.to('cuda')), model(x3.to('cuda'))
        loss = crit(e1, e2, e3)
        prevloss += loss
        loss.backward()
        opt.step()
        # batch_map = compute_map(out, target)
        # print("BATCH MAP IS: {batch_map}").format(batch_map)
    print("curr loss is", prevloss)
    return prevloss


def valid(val_transforms, model):
    faiss_index = faiss.IndexFlatL2(1000)   # build the index

    # storing the image representations
    im_indices = []
    with torch.no_grad():
        # print(glob.glob(os.path.join("dataset/flowers/train", '*/*')))
        for f in glob.glob(os.path.join("dataset/flowers/train", '*/*')):
            # print(f)
            im = Image.open(f)
            im = im.resize((224,224))
            im = torch.tensor([val_transforms(im).numpy()]).cuda()
        
            preds = model(im)
            preds = np.array([preds[0].cpu().numpy()])
            faiss_index.add(preds) #add the representation to index
            im_indices.append(f)   #store the image name to find it later on

    # Retrieval with a query image
    with torch.no_grad():
        for f in os.listdir("dataset/flowers/test"):
            # query/test image
            qimg = os.listdir(os.path.join("dataset/flowers/test",f))[0]
            print("QIMG",qimg)
            im = Image.open(os.path.join("dataset/flowers/test",f, qimg))
            im = im.resize((224,224))
            im = torch.tensor([val_transforms(im).numpy()]).cuda()
        
            test_embed = model(im).cpu().numpy()
            _, I = faiss_index.search(test_embed, 5)
            print("Retrieved Image: {}".format(im_indices[I[0][0]]))

    

if __name__ == '__main__':
    main()

