import argparse
import os
import pathlib
import random

import numpy as np
import torch
torch.cuda.empty_cache()
import torchvision.models as tvmodels
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from pytorch_pretrained_vit import ViT
from database import BaseDatabase
from models import SiameseNet
from dataset import TripletData
from dataset import SiameseData
from loss import TripletLoss
from loss import ContrastiveLoss
from MLPMixer.models.modeling import MlpMixer, CONFIGS

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/SimpleNetwork.yaml')
parser.add_argument('--mode', default='train')

def main():

    set_seed()
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    if args.mode != 'train':
        raise NotImplementedError('Only train mode implemented so far')

    if args.model == 'ResNet':
        # model = resnet32()
        model = tvmodels.resnet18()
    elif args.model == 'ViT':
        model = ViT('B_16_imagenet1k', pretrained=True)
        # model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    elif args.model == 'SiameseNet':
        if 'backbone' in config:
            backbone = args.backbone
        else:
            backbone = 'resnet18'
        model = SiameseNet(backbone)
    elif args.model == 'MLPMixer':
        # TO USE DOWNLOAD PRETRAINED MODEL: wget https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz
        c = CONFIGS['Mixer-B_16-21k']
        model = MlpMixer(c, args.img_h, num_classes=17, patch_size=16, zero_head=True)
        model.load_from(np.load(args.pretrained_path))
    else:
        raise NotImplementedError(args.model + " model not implemented!")
    if args.dataset == 'TripletData':
        train_transforms = transforms.Compose([
            transforms.Resize((args.img_w, args.img_h)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        val_transforms = transforms.Compose([
            transforms.Resize((args.img_w, args.img_h)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        # NOTE THIS IS SAME AS TEST, NEED A VAL DATASET
        val_dataset = TripletData(args.train_path, val_transforms, path=args.test_path)
        train_dataset = TripletData(args.train_path, train_transforms, path=args.train_path)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'SiameseData':
        train_transforms = transforms.Compose([transforms.Resize((args.img_w, args.img_h)),
                                               transforms.RandomResizedCrop(100),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomRotation(10),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

        val_transforms = transforms.Compose([
            transforms.Resize((args.img_w, args.img_h)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        val_dataset = SiameseData(args.train_path, val_transforms)
        train_dataset = SiameseData(args.train_path, train_transforms)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        raise NotImplementedError(args.dataset + " dataset not implemented!")

    if torch.cuda.is_available():
        model = model.cuda()

    if args.loss_type == 'TripletLoss':
        criterion = TripletLoss()
    elif args.loss_type == 'ContrastiveLoss':
        criterion = ContrastiveLoss()
    else:
        raise NotImplementedError(args.loss_type + " loss not implemented!")

    if args.optimizer.lower() == 'sdg':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        raise Exception("Invalid optimizer option".format(args.optimizer))

    v = vars(args)
    for epoch in range(args.epochs):
        if epoch % args.validevery == 0:
            print("RUNNING VALIDATION AT EPOCH", epoch)
            trainpath = args.train_path
            if epoch == 0 and 'save_db' in v and args.save_db == True:
                valdb = create_database(args.data_size, 'Base', val_transforms, model, trainpath, saveto=args.faiss_db)
            elif epoch == 0 and 'faiss_db' in v:
                valdb = create_database(args.data_size, 'Base', val_transforms, model, trainpath, npy=args.faiss_db+'.npy')
            else:
                valdb = create_database(args.data_size, 'Base', val_transforms, model, trainpath)
            test(valdb, args.val_path)
        loss = 0.0
        model.train()
        if args.model == 'SiameseNet':
            loss = train_siamese(epoch, train_loader, model, optimizer, criterion)
        else:
            train(epoch, train_loader, model, optimizer, criterion, loss)
        print("epoch {0}: Loss = {1}".format(epoch, loss))
        # acc, cm = validate(epoch, val_loader, model, criterion)

    trainpath = args.train_path
    testdb = create_database(args.data_size, 'Base', val_transforms, model, trainpath, saveto="testsave")
    test(testdb, args.test_path)


def set_seed():
    # Setting seed value to reproduce results
    torch.manual_seed(1)
    random.seed(1)


def train_siamese(epoch, loader, model, opt, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Epoch", str(epoch))
    cum_loss = 0
    for data in tqdm(loader):
        opt.zero_grad()
        img0, img1, label = data
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        opt.zero_grad()
        output0, output1 = model(img0.to(device)), model(img1.to(device))
        loss = criterion(output0, output1, label)
        loss.backward()
        opt.step()
        cum_loss += loss
    return cum_loss


def train(epoch, loader, model, opt, crit, loss):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Epoch", str(epoch))
    for data in tqdm(loader):
        opt.zero_grad()
        x1, x2, x3 = data
        e1, e2, e3 = model(x1.to(device)), model(x2.to(device)), model(x3.to(device))
        l = crit(e1, e2, e3)
        l.backward()
        opt.step()
        loss += l
        # batch_map = compute_map(out, target)
        # print("BATCH MAP IS: {batch_map}").format(batch_map)
    print("Current loss is", loss)
    return loss


def create_database(size, dbtype, transforms, model, path, saveto=None, npy=None):
    if dbtype == "Base":
        db = BaseDatabase(model, path, transforms, size=size, saveto=saveto, db=npy)
        return db
    else:
        raise NotImplementedError(dbtype + " database not implemented!")


def test(db, test_path, full_test=False):
    # Retrieval with a query image
    category_matches = 0
    total_queries = 0
    with torch.no_grad():
        for f in os.listdir(test_path):
            if full_test:
                qimgs = os.listdir(os.path.join(test_path, f))
                for qimg in qimgs:
                    total_queries += 1
                    im = Image.open(os.path.join(test_path, f, qimg))
                    I = db.search(im, 5)
                    print("CLASS {}.... QIMG {} Retrieved Image: {}".format(f, qimg, db.im_indices[I[0][0]]))
                    if str(pathlib.Path(db.im_indices[I[0][0]]).parts[3]) == f:
                        print("Found a match from", qimg, "class", f)
                        category_matches += 1
            else:
                qimg = os.listdir(os.path.join(test_path, f))[0]
                total_queries += 1
                print("CLASS", f, ".... IMG", qimg)
                im = Image.open(os.path.join(test_path, f, qimg))
                I = db.search(im, 5)
                print("Retrieved Image: {}".format(db.im_indices[I[0][0]]))
                if str(pathlib.Path(db.im_indices[I[0][0]]).parts[3]) == f:
                    print("Found a match from", qimg, "class", f)
                    category_matches += 1
    print("CATEGORY MATCHES: {}/{}: {:.4f}".format(category_matches, total_queries, category_matches/total_queries))    

if __name__ == '__main__':
    main()
