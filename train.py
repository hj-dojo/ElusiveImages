import argparse
from pickletools import optimize
from tqdm import tqdm
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
        model = resnet32()

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

        val_dataset = TripletData(args.data+'/test', val_transforms)
        train_dataset = TripletData(args.data+'/test', train_transforms, path="dataset/flowers/test")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # if torch.cuda.is_available:
    #     model = model.cuda()
    
    if args.loss_type == 'TripletLoss':
        criterion = TripletLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    for epoch in range(args.epochs):
        loss = 0.0
        model.train()
        loss = train(epoch, train_loader, model, optimizer, criterion, loss)
        # acc, cm = validate(epoch, val_loader, model, criterion)

def train(epoch, loader, model, opt, crit, prevloss):
    print("Start training")
    for data in tqdm(loader):
        # data, target = d
        # if torch.cuda.is_available():
            # data = data.cuda()
            # target = target.cuda()
        
        opt.zero_grad()
        x1, x2, x3 = data
        e1, e2, e3 = model(x1), model(x2), model(x3)
        loss = crit(e1, e2, e3)
        currloss = prevloss + loss
        loss.backward()
        opt.step()
        # batch_map = compute_map(out, target)
        # print("BATCH MAP IS: {batch_map}").format(batch_map)
    print("curr loss is", currloss)
    return currloss


def validate(epoch, loader, model, crit):
    for data in tqdm(loader):
        # if torch.cuda.is_available():
            # data = data.cuda()
            # target = target.cuda()
        with torch.no_grad():
            out = model(data)
            loss = crit(out, target)        

        batch_map = compute_map(out, target)
        print("BATCH MAP IS: {batch_map}").format(batch_map)
    

if __name__ == '__main__':
    main()

