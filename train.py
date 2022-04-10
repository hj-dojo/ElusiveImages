import argparse
import os

import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader

from dataset.TripletData import TripletData
from loss.TripletLoss import TripletLoss
from models import resnet32
from utils.MAP import compute_map

import faiss
import glob
from PIL import Image
import numpy as np

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
        model = resnet32().cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_size = 224
    cats = 16
    if args.dataset == 'TripletData':
        train_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        valid_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        test_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        train_path, valid_path, test_path = os.path.join(args.folder, 'train'), os.path.join(args.folder, 'valid'), os.path.join(args.folder, 'test')
        train_dataset = TripletData(train_path, train_transforms, cats=cats)
        val_dataset = TripletData(valid_path, valid_transforms, cats=cats)
        test_dataset = TripletData(test_path, test_transforms, cats=cats)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if torch.cuda.is_available:
        model = model.cuda()

    if args.loss == 'TripletLoss':
        criterion = TripletLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)

    # Train
    for epoch in range(args.epochs):
        train(epoch, train_loader, model, optimizer, criterion, device)

    # Create image search index
    faiss_index, im_indices = create_faiss_index(model, train_path, valid_transforms)

    # Validation data
    # test(model, valid_path, valid_transforms, fasis_index, im_indices)

    # Test data
    test(model, test_path, test_transforms, faiss_index, im_indices)


def train(epoch, loader, model, optimizer, criterion, device):
    epoch_loss = 0
    model.train()
    for idx, data in enumerate(loader):
        optimizer.zero_grad()
        x1, x2, x3 = data
        # Model forward + backward + Opt
        e1 = model(x1.to(device))
        e2 = model(x2.to(device))
        e3 = model(x3.to(device))

        loss = criterion(e1, e2, e3)
        print("Epoch {} idx {} loss {}".format(epoch, idx, loss))
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    print("Train Loss: {}".format(epoch_loss.item()))


def create_faiss_index(model, train_path, valid_transforms):
    faiss_index = faiss.IndexFlatL2(10)  # build the index

    im_indices = []
    with torch.no_grad():
        for f in glob.glob(os.path.join(train_path, '*/*')):
            im = Image.open(f)
            im = im.resize((224, 224))
            im = torch.tensor([valid_transforms(im).numpy()]).cuda()

            preds = model(im)
            preds = np.array([preds[0].cpu().numpy()])
            faiss_index.add(preds)  # add the representation to index
            im_indices.append(f)  # store the image name to find it later on
    return faiss_index, im_indices


def test(model, test_path, valid_transforms, faiss_index, im_indices):
    with torch.no_grad():
        for f in os.listdir(test_path):
            im = Image.open(os.path.join(test_path, f))
            im = im.resize((224, 224))
            im = torch.tensor([valid_transforms(im).numpy()]).cuda()

            test_embed = model(im).cpu().numpy()
            _, I = faiss_index.search(test_embed, 5)
            print("Input image: {}, Retrieved Image: {}".format(f, im_indices[I[0][0]]))


if __name__ == '__main__':
    main()
