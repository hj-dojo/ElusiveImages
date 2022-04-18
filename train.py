import argparse
import os
import pathlib
import random
import sys

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
from torch import nn
import logging as log

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/SimpleNetwork.yaml')
parser.add_argument('--mode', default='train')

# Seed value to reproduce results
seed_value = 123456 # acc: 0.9963
seed_value = 123 # acc: 0.9963


def main():
    set_seed(seed_value)
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    log.basicConfig(level=args.loglevel.upper(), format='%(message)s')
    log.info("Args: {}".format(args))

    if args.mode.lower() != 'train':
        raise NotImplementedError('Only train mode implemented so far')

    # ----- Model ----- #
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
        model = SiameseNet(backbone, args.pretrain)
    elif args.model == 'MLPMixer':
        # TO USE DOWNLOAD PRETRAINED MODEL: wget https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz
        c = CONFIGS['Mixer-B_16-21k']
        model = MlpMixer(c, args.img_h, num_classes=17, patch_size=16, zero_head=True)
        model.load_from(np.load(args.pretrained_path))
    else:
        raise NotImplementedError(args.model + " model not implemented!")

    if torch.cuda.is_available():
        model = model.cuda()

    # ----- Dataset ----- #
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
    else:
        raise NotImplementedError(args.dataset + " dataset not implemented!")

    g = torch.Generator()
    g.manual_seed(0)

    train_dataset = TripletData(args.train_path, train_transforms, path=args.train_path, cats=len(os.listdir(args.train_path)))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    # NOTE THIS IS SAME AS TEST, NEED A VAL DATASET
    # val_dataset = TripletData(args.train_path, val_transforms, path=args.test_path)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    # ----- Loss ----- #
    if args.loss_type == 'TripletLoss':
        criterion = TripletLoss()
    elif args.loss_type == 'ContrastiveLoss':
        criterion = ContrastiveLoss()
    else:
        raise NotImplementedError(args.loss_type + " loss not implemented!")

    # ----- Optimizer ----- #
    if args.optimizer.lower() == 'sdg':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    elif args.optimizer.lower() == 'feature_extractor':
        log.info("With optimizer mode set to {}, final FC layer is being randomly initialized again for training".format(args.optimizer))
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 1000, device='cuda')

        params_to_update = []
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, args.learning_rate)
    else:
        raise Exception("Invalid optimizer option".format(args.optimizer))

    # ----- Train ----- #
    v = vars(args)
    trainpath = args.train_path
    for epoch in range(args.epochs):
        if epoch % args.validevery == 0:
            log.info("RUNNING VALIDATION AT EPOCH {}".format(epoch))
            if epoch == 0 and 'save_db' in v and args.save_db == True:
                valdb = create_database(args.data_size, 'Base', val_transforms, model, trainpath, args.img_w, args.img_h, saveto=args.faiss_db)
            elif epoch == 0 and 'faiss_db' in v:
                valdb = create_database(args.data_size, 'Base', val_transforms, model, trainpath, args.img_w, args.img_h,
                                        npy=args.faiss_db + '.npy')
            else:
                valdb = create_database(args.data_size, 'Base', val_transforms, model, trainpath, args.img_w, args.img_h)
            test(valdb, args.val_path)
        model.train()
        if args.model == 'SiameseNet':
            loss = train_siamese(epoch, train_loader, model, optimizer, criterion)
        else:
            loss = train(epoch, train_loader, model, optimizer, criterion)
        log.info("epoch {0}: Loss = {1}".format(epoch, loss))
        # acc, cm = validate(epoch, val_loader, model, criterion)

    # ---- Test ---- #
    # Set to eval mode
    model.eval()
    testdb = create_database(args.data_size, 'Base', val_transforms, model, trainpath, args.img_w, args.img_h, saveto="testsave")
    test(testdb, args.test_path)


def seed_worker(worker_id):
    log.debug(worker_id)
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed_value):
    # Don't use log info here, as logging level is not set when set_seed is called.
    print("Setting seed value to {} for reproducing results".format(seed_value))
    # CUDA Reproducibility: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    # Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    # Setting seed value to reproduce results
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    torch.use_deterministic_algorithms(True)


def train_siamese(epoch, loader, model, opt, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Epoch {}".format(epoch))
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


def train(epoch, loader, model, opt, crit):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Epoch {}".format(epoch))
    loss = 0
    for data in tqdm(loader):
        opt.zero_grad()
        x1, x2, x3 = data
        e1, e2, e3 = model(x1.to(device)), model(x2.to(device)), model(x3.to(device))
        l = crit(e1, e2, e3)
        l.backward()
        opt.step()
        loss += l
        # batch_map = compute_map(out, target)
        # log.info("BATCH MAP IS: {batch_map}").format(batch_map)
    return loss


def create_database(size, dbtype, transforms, model, path, img_w, img_h, saveto=None, npy=None):
    if dbtype == "Base":
        db = BaseDatabase(model, path, transforms, imgdims=(img_w, img_h), size=size, saveto=saveto, db=npy)
        return db
    else:
        raise NotImplementedError(dbtype + " database not implemented!")


def test(db, test_path, full_test=True):
    # Retrieval with a query image
    category_matches = 0
    total_queries = 0
    with torch.no_grad():
        for f in os.listdir(test_path):
            qimgs = os.listdir(os.path.join(test_path, f)) if full_test else os.listdir(os.path.join(test_path, f))[:1]
            for qimg in qimgs:
                total_queries += 1
                im = Image.open(os.path.join(test_path, f, qimg))
                I = db.search(im, 5)
                log.debug("CLASS {}.... QIMG {} Retrieved Image: {}".format(f, qimg, db.im_indices[I[0][0]]))
                if str(pathlib.Path(db.im_indices[I[0][0]]).parts[3]) == f:
                    log.debug("Found a match from {} class {}".format(qimg, f))
                    category_matches += 1
    log.info("Args: {}".format(args))
    log.info(
        "CATEGORY MATCHES: {}/{}: {:.4f}".format(category_matches, total_queries, category_matches / total_queries))


if __name__ == '__main__':
    main()
