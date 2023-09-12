from __future__ import print_function
import argparse
import random
from os.path import join, isfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.models as models

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import load as dataset
import numpy as np
import scripts.netvlad as netvlad

parser = argparse.ArgumentParser(description='Extracting DB Feature')
parser.add_argument('--cacheBatchSize', type=int, default=1, help='Batch size for caching and testing')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--dataset', type=str, default='NIA', help='select Dataset type [gazebo, NIA, iiclab]')

def saveFeature(eval_set):
    test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, 
                                  shuffle=False, pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        print('====> Extracting DB Features')
        pool_size = encoder_dim
        pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))
        for input, indices in test_data_loader:
            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)
            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

            del input, image_encoding, vlad_encoding
    del test_data_loader
    np.save('../../feature/dbFeature_' + opt.dataset, dbFeat)
    print('Save DB Feature successfully')

if __name__ == "__main__":
    opt = parser.parse_args()
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    
    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    if opt.dataset == 'gazebo':
        whole_test_set = dataset.get_gazebo_DB_test_set()
    elif opt.dataset == 'NIA':
        whole_test_set = dataset.get_DB_test_set()
    elif opt.dataset == 'iiclab':
        whole_test_set = dataset.get_iiclab_DB_test_set()

    # default opt.fromscratch = False
    pretrained = not opt.fromscratch
    encoder_dim = 512
    encoder = models.vgg16(pretrained=pretrained)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]

    if pretrained:
        # if using pretrained then only train conv5_1, conv5_2, and conv5_3
        for l in layers[:-5]: 
            for p in l.parameters():
                p.requires_grad = False

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)
    
    net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)

    model.add_module('pool', net_vlad)
    
    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))
    saveFeature(whole_test_set)
