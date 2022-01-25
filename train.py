'''Train CIFAR10 with PyTorch.'''
from socketserver import ThreadingUDPServer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import sys, os, warnings
from argparse import Namespace
warnings.filterwarnings("ignore")

import torch as ch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision

from helpers import classifier_helpers
import helpers.data_helpers as dh
import helpers.context_helpers as coh
import helpers.rewrite_helpers as rh
import helpers.vis_helpers as vh

import matplotlib.pyplot as plt
import random

import helpers.new_testing as nt
from helpers.new_testing import IMAGENET_CLASSES
from helpers.waterbirds import Waterbirds

import wandb

from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--checkpoint-name' , '-c', type=str, default=None, help="name of checkpoint")
parser.add_argument('--model', '-m', type=str, default='ResNet18', help='model architecture')
parser.add_argument('--dataset', '-d', type=str, default='Waterbirds', help='dataset name')
parser.add_argument('--batch', '-b', type=int, default=128, help='train/test batch size')
parser.add_argument('--eval-only', '-e', action='store_true', help='evaluate checkpoint')
parser.add_argument('--aug-mix', action='store_true', help='apply augmix')
parser.add_argument('--proj', '-p', default='waterbirds-training', type=str, help='project name')
parser.add_argument('--pretrained', action='store_true', help='for imagenet, load pretained')
args = parser.parse_args()

run = wandb.init(project=args.proj, group=args.dataset, config=args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

DATASET_NAME =  args.dataset
LAYERNUM = 24
REWRITE_MODE = 'editing'
ARCH = 'resnet50'
MODEL_PATH = '/home/lisabdunlap/EditingClassifiers/models/waterbirds.ckpt'

base_dataset, trainloader, testloader = dh.get_dataset(DATASET_NAME, '/shared/group/ilsvrc',
                                                        batch_size=96, workers=8)

# log some example images
example_indices = [0, 1, 4000, 4001]
wandb.log({"Example Images": [wandb.Image(nt.get_displ_img(base_dataset[i][0], norm=True)) for i in example_indices]})

# Model
print('==> Building model..')
ret = classifier_helpers.get_default_paths(DATASET_NAME, arch=ARCH)
DATASET_PATH, MODEL_PATH, net, ARCH, CD = ret
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.checkpoint_name:
        checkpoint = torch.load(f'./checkpoint/{args.checkpoint_name}.pth')
    else:
        if args.aug_mix:
            checkpoint = torch.load(f'./checkpoint/ckpt-{args.model}-AugMix.pth')
        else:
            checkpoint = torch.load(f'./checkpoint/ckpt-{args.model}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = 0 if args.eval_only else checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (input, label) in enumerate(trainloader):
        # print(batch)
        inputs, targets = input.to(device), label.long().to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    wandb.log({'train loss': train_loss, 'train acc': 100 * correct/total}, step=epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(testloader):
            inputs, targets = input.to(device), label.long().to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # log classifications

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        wandb.log({'val loss': test_loss, 'val acc': 100 * correct/total}, step=epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        if not args.resume:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if args.checkpoint_name:
                torch.save(state, f'./checkpoint/{args.checkpoint_name}.pth')
                wandb.save(f'./checkpoint/{args.checkpoint_name}.pth')
            else:
                if args.aug_mix:
                    torch.save(state, f'./checkpoint/ckpt-{args.model}-AugMix.pth')
                    wandb.save(f'./checkpoint/ckpt-{args.model}-AugMix.pth')
                else:
                    torch.save(state, f'./checkpoint/ckpt-{args.model}.pth')
                    wandb.save(f'./checkpoint/ckpt-{args.model}.pth')
        best_acc = acc
        wandb.summary['best acc'] = best_acc

if args.eval_only:
    test(start_epoch)
    # logger.end()
else:
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        # scheduler.step()
