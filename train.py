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
from database import BaseDatabase
from models import SiameseNet
from dataset import TripletData
from dataset import ContrastiveData
from dataset import QuadrupletData
from loss import TripletLoss
from loss import ContrastiveLoss
from loss import QuadrupletLoss
from MLPMixer.models.modeling import MlpMixer, CONFIGS
from torch import nn
import logging as log
from utils.MAP import compute_map
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/SimpleNetwork.yaml')
parser.add_argument('--mode', default='train')

# Seed value to reproduce results
seed_value = 111111



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def main():
    set_seed(seed_value)
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    all_params = {}
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
            all_params[k] = v

    # log.basicConfig(level=args.loglevel.upper(), format='%(message)s')
    log_file_name = 'analysis_{}_{}_{}_{}_{}_ep{}_lr{}_m{}_bs{}_w{}_h{}_seed{}'.format(args.model.lower(),
                                                                                       pathlib.Path(
                                                                                           args.train_path).parts[1],
                                                                                       args.loss_type.lower(),
                                                                                       args.dataset.lower(),
                                                                                       args.optimizer.lower(),
                                                                                       args.epochs,
                                                                                       str(args.learning_rate).replace(
                                                                                           '.', '_'),
                                                                                       str(args.momentum).replace('.',
                                                                                                                  '_'),
                                                                                       args.batch_size,
                                                                                       args.img_w,
                                                                                       args.img_h,
                                                                                       seed_value)

    log.basicConfig(
        level=args.loglevel.upper(),
        format="[%(levelname)s] %(message)s",
        handlers=[
            log.FileHandler("{}.txt".format(os.path.join(args.logdir, log_file_name))),
            log.StreamHandler(sys.stdout)
        ]
    )

    print("Args: {}, seed: {}".format(args, seed_value))

    if args.mode.lower() != 'train':
        raise NotImplementedError('Only train mode implemented so far')

    run_experiment(all_params, log_file_name)


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


def run_experiment(params, log_file_name):

    # ----- Model ----- #
    if params['model'] == 'MLPMixer':
        model = create_model(params['model'], params['category'], params['pretrain'],
                                   params['img_h'], params['img_w'], num_classes=17, patch_size=16, zero_head=True)
    else:
        model = create_model(params['model'], params['category'], params['pretrain'],
                                   params['img_h'], params['img_w'])

    # ----- Dataset ----- #
    train_loader, train_transforms, _, val_transforms = create_dataset(params['dataset'], params['batch_size'],
                                                                       params['train_path'], params['val_path'],
                                                                     params['img_w'], params['img_h'], create_validation=False)

    # ----- Loss ----- #
    if params['loss_type'] == 'TripletLoss':
        criterion = TripletLoss()
    elif params['loss_type'] == 'ContrastiveLoss':
        criterion = ContrastiveLoss()
    elif params['loss_type'] == 'QuadrupletLoss':
        criterion = QuadrupletLoss()
    else:
        raise NotImplementedError( params['loss_type'] + " loss not implemented!")

    # ----- Optimizer ----- #
    optimizer = create_optimizer(params['model'], model, params['optimizer'], params['learning_rate'], params['momentum'])

    # ----- Train ----- #
    v = vars(args)
    pre_train_accuracy = 'NA'
    loss_per_iter = []
    trainpath = params['train_path']
    num_epochs = params['epochs']
    for epoch in range(num_epochs):
        if epoch % params['validevery'] == 0:
            log.info("RUNNING VALIDATION AT EPOCH {}".format(epoch))
            if epoch == 0 and 'save_db' in params and params['save_db'] == True:
                valdb = create_database(params['data_size'], 'Base', val_transforms, model, trainpath, params['img_w'],
                                        params['img_h'], saveto=params['faiss_db'])
            elif epoch == 0 and 'faiss_db' in params:
                valdb = create_database(params['data_size'], 'Base', val_transforms, model, trainpath, params['img_w'],
                                        params['img_h'], npy=params['faiss_db'] + '.npy')
            else:
                valdb = create_database(params['data_size'], 'Base', val_transforms, model, trainpath, params['img_w'],
                                        params['img_h'])

            pre_train_accuracy = test(valdb, params['val_path'])
        model.train()
        loss = train(epoch, train_loader, model, optimizer, params['loss_type'], criterion)
        log.info("epoch {0}: Loss = {1}".format(epoch, loss))
        loss_per_iter.append(loss.item())
        # acc, cm = validate(epoch, val_loader, model, criterion)
    # ---- Test ---- #
    # Set to eval mode
    model.eval()

    testdb = create_database(params['data_size'], 'Base', val_transforms, model, trainpath, params['img_w'],
                                        params['img_h'], saveto="testsave")
    post_train_accuracy = test(testdb, params['test_path'])
    log.info("Accuracy change: pre-training {} --> post-training {}".format(pre_train_accuracy, post_train_accuracy))
    log.info("---" * 60)

    if loss_per_iter:
        plt.plot(loss_per_iter)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('{}'.format(os.path.join(params['logdir'], log_file_name)))
        plt.show()


def create_model(model_name, model_category, pretrain, img_height, img_width, **kwargs):
    # ----- Model ----- #
    if model_name == 'ResNet':
        # model = resnet32()
        model = tvmodels.resnet18()
    elif model_name == 'ViT':
        class_ = getattr(tvmodels, args.category)
        model = class_(pretrained=True)

        ###  FOR OTHER EXPERIMENTS
        # model.encoder.layers.encoder_layer_11 = Identity()
        # model.encoder.layers.encoder_layer_10 = Identity()
        # model.encoder.layers.encoder_layer_9 = Identity()
        # model.encoder.layers.encoder_layer_8 = Identity()
        # model.encoder.layers.encoder_layer_7 = Identity()
        # model.encoder.layers.encoder_layer_6 = Identity()
        # model.encoder.layers.encoder_layer_5 = Identity()
        # model.encoder.layers.encoder_layer_4 = Identity()
        # model.encoder.layers.encoder_layer_3 = Identity()
        # model.encoder.layers.encoder_layer_2 = Identity()
        # model.encoder.layers.encoder_layer_1 = Identity()
        # model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    elif model_name == 'SiameseNet':
        if model_category is not None:
            backbone = model_category
        else:
            backbone = 'resnet18'
        model = SiameseNet(backbone, pretrain)
    elif args.model == 'MLPMixer':
        # TO USE DOWNLOAD PRETRAINED MODEL: wget https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz
        c = CONFIGS[model_category]
        if kwargs is not None:
            patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else 16
            zero_head = kwargs['zero_head'] if 'zero_head' in kwargs else True
            num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 17
            model = MlpMixer(c, img_height, num_classes=num_classes, patch_size=patch_size, zero_head=zero_head)
        else:
            model = MlpMixer(c, img_height, num_classes=17, patch_size=16, zero_head=True)
        model.load_from(np.load(args.pretrained_path))
    else:
        raise NotImplementedError(args.model + " model not implemented!")

    if torch.cuda.is_available():
        model = model.cuda()
    return model


def create_dataset(dataset_type, batch_size, train_path, validation_path, img_width, img_height, create_validation):
    # ----- Dataset ----- #
    g = torch.Generator()
    g.manual_seed(0)
    if dataset_type == 'TripletData':
        train_transforms = transforms.Compose([
            transforms.Resize((img_width, img_height)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        val_transforms = transforms.Compose([
            transforms.Resize((img_width, img_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        train_dataset = TripletData(train_path, train_transforms, path=train_path, cats=len(os.listdir(train_path)))
        # NOTE THIS IS SAME AS TEST, NEED A VAL DATASET
        val_dataset = TripletData(validation_path, val_transforms, path=validation_path)
    elif dataset_type == 'ContrastiveData':
        train_transforms = transforms.Compose([transforms.Resize((img_width, img_height)),
                                               transforms.RandomResizedCrop(100),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomRotation(10),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

        val_transforms = transforms.Compose([
            transforms.Resize((img_width, img_height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        val_dataset = ContrastiveData(validation_path, val_transforms)
        train_dataset = ContrastiveData(args.train_path, train_transforms)
    else:
        raise NotImplementedError(dataset_type + " dataset not implemented!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                              generator=g)

    if create_validation:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                                generator=g)
    else:
        val_loader = None

    return train_loader, train_transforms, val_loader, val_transforms


def create_optimizer(model_type, model, optimizer_type, learning_rate, momentum):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- Optimizer ----- #
    if optimizer_type.lower() == 'sdg':
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    elif optimizer_type.lower() == 'feature_extractor':
        log.info(
            "With optimizer mode set to {}, final FC layer is being randomly initialized again for training".format(
                optimizer_type))
        for param in model.parameters():
            param.requires_grad = False
        if model_type == 'ViT':
          model.heads[0] = nn.Linear(model.heads[0].in_features, 1000, device=device)
        else:
          model.fc = nn.Linear(model.fc.in_features, 1000, device=device)
        params_to_update = []
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, learning_rate)
    else:
        raise Exception("Invalid optimizer option".format(optimizer_type))
    return optimizer


def train(epoch, loader, model, opt, loss_type, crit):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Epoch {}".format(epoch))
    loss = 0
    for data in tqdm(loader):
        opt.zero_grad()
        if loss_type == 'TripletLoss':
            x1, x2, x3 = data
            e1, e2, e3 = model(x1.to(device)), model(x2.to(device)), model(x3.to(device))
            l = crit(e1, e2, e3)
        elif loss_type == 'ContrastiveLoss':
            x1, x2, label = data
            e1, e2, label = model(x1.to(device)), model(x2.to(device)), label.to(device)
            l = crit(e1, e2, label)
        elif loss_type == 'QuadrupletLoss':
            x1, x2, x3, x4 = data
            e1, e2, e3, e4 = model(x1.to(device)), model(x2.to(device)), model(x3.to(device)), model(x4.to(device))
            l = crit(e1, e2, e3, e4)

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

def test(db, test_path, full_test=True, use_map=False, search_size=16):
    # Retrieval with a query image
    category_matches = 0
    total_queries = 0
    maps = []
    with torch.no_grad():
        for f in os.listdir(test_path):
            qimgs = os.listdir(os.path.join(test_path, f)) if full_test else os.listdir(os.path.join(test_path, f))[:1]
            curr_folder_maps = []
            for i, qimg in enumerate(qimgs):
                total_queries += 1
                im = Image.open(os.path.join(test_path, f, qimg))
                I = db.search(im, search_size)
                if use_map:
                  curr_map = compute_map(I, db, f, search_size)
                  curr_folder_maps.append(curr_map)
                else:
                  log.debug("CLASS {}.... QIMG {} Retrieved Image: {}".format(f, qimg, db.im_indices[I[0][0]]))
                  if str(pathlib.Path(db.im_indices[I[0][0]]).parts[3]) == f:
                    log.debug("Found a match from {} class {}".format(qimg, f))
                    category_matches += 1
            if use_map:
              maps.append(sum(curr_folder_maps)/float(search_size))
    log.info("Args: {}".format(args))
    if use_map:
        log.info("MEAN AVERAGE PRECISIONS BY CATEGORY: {}".format(maps))
    else:
        accuracy = round(category_matches / total_queries, 4)
        log.info("Args: {}, seed: {}".format(args, seed_value))
        log.info(
            "CATEGORY MATCHES: {}/{}: {:.4f}".format(category_matches, total_queries, accuracy))
        return accuracy


if __name__ == '__main__':
    main()
