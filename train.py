import argparse
import glob
import itertools
import os
import pathlib

import faiss
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from dataset.TripletData import TripletData
from loss.TripletLoss import TripletLoss
from models import resnet32

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/SimpleNetwork.yaml')


# Setting seed value to reproduce results
torch.manual_seed(1)
import random;random.seed(1)
np.random.seed(1)


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

    print(args)

    if args.dataset == 'TripletData':
        train_transforms = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        # ToDo: Why do we need two separate transforms?
        valid_transforms = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        # test_transforms = transforms.Compose([
        #     transforms.Resize((args.img_size, args.img_size)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        img_path, valid_path, test_path = os.path.join(args.folder, 'train'), os.path.join(args.folder,
                                                                                             'valid'), os.path.join(
            args.folder, 'test')
        train_dataset = TripletData(img_path, train_transforms, cats=args.cats)
        # valid_dataset = TripletData(valid_path, valid_transforms, cats=args.cats)
        # test_dataset = TripletData(test_path, test_transforms, cats=args.cats)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if torch.cuda.is_available:
        model = model.cuda()

    if args.loss == 'TripletLoss':
        criterion = TripletLoss()

    if args.optimizer.lower() == 'sdg':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    elif args.optimizer.lower() == 'adam' :
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        raise Exception("Invalid optimizer option".format(args.optimizer))

    # Train
    for epoch in range(args.epochs):
        train(epoch, train_loader, model, optimizer, criterion, device)

    # Create image search index
    faiss_index, im_indices = create_faiss_index(model, img_path, valid_transforms, args.cats)

    # Validation data
    evaluate(model, valid_path, valid_transforms, faiss_index, im_indices, args.cats)


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


def create_faiss_index(model, img_path, transforms, cats):
    faiss_index = faiss.IndexFlatL2(10)  # build the index

    im_indices = []
    with torch.no_grad():
        train_images_from_cats = list(
            itertools.chain(*[glob.glob(os.path.join(img_path, '{}/*'.format(idx))) for idx in range(1, cats + 1)]))

        for f in train_images_from_cats:
            im = Image.open(f)
            im = im.resize((224, 224))
            im = torch.tensor([transforms(im).numpy()]).cuda()

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


def evaluate(model, test_path, test_transforms, faiss_index, im_indices, cats):
    test_images_from_cats = list(
        itertools.chain(*[glob.glob(os.path.join(test_path, '{}/*'.format(idx))) for idx in range(1, cats + 1)]))
    correct_matches = 0
    with torch.no_grad():
        for test_image in test_images_from_cats:
            im = Image.open(test_image)
            im = im.resize((224, 224))
            im = torch.tensor([test_transforms(im).numpy()]).cuda()

            test_embed = model(im).cpu().numpy()
            _, I = faiss_index.search(test_embed, 5)
            output_image = im_indices[I[0][0]]
            input_imgage_cat = pathlib.Path(test_image).parts[2]
            output_imgage_cat = pathlib.Path(output_image).parts[2]
            if input_imgage_cat == output_imgage_cat:
                correct_matches += 1
            print("Input image: {}, Retrieved Image: {}".format(test_image, output_image))
    print(args)
    print("Accuracy: {}/{} : {}".format(correct_matches, len(test_images_from_cats),
                                        correct_matches / len(test_images_from_cats)))


if __name__ == '__main__':
    main()
