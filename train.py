import argparse
from pickletools import optimize

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
        config = yaml.load(f)
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

        test_dataset = TripletData(args.data+'/test', val_transforms)
        train_dataset = TripletData(args.data+'/train', train_transforms)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if torch.cuda.is_available:
        model = model.cuda()
    
    if args.loss_type == 'TripletLoss':
        criterion = TripletLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    best = 0.0
    for epoch in range(args.epochs):
        train(epoch, train_loader, model, optimizer, criterion)
        acc, cm = validate(epoch, test_loader, model, criterion)

def train(epoch, loader, model, opt, crit):
    for idx, (data, target) in enumerate(loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        opt.zero_grad()
        out = model.forward(data)
        loss = crit(out, target)
        loss.backward()
        opt.step()
        batch_map = compute_map(out, target)
        print("BATCH MAP IS: {batch_map}").format(batch_map)


def validate(epoch, loader, model, crit):
    for idx, (data, target) in enumerate(loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            out = model(data)
            loss = crit(out, target)        

        batch_map = compute_map(out, target)
        print("BATCH MAP IS: {batch_map}").format(batch_map)
    

if __name__ == '__main__':
    main()

