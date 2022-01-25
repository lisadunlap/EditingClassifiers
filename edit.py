from ast import parse
from pathlib import PureWindowsPath
import sys, os, warnings
from argparse import Namespace
warnings.filterwarnings("ignore")

import torch as ch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
import argparse
import wandb

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

from helpers.waterbirds import Waterbirds, WaterbirdsEditing

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default="Waterbirds", help='dataset')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--arch', default='resnet50', help='model')
parser.add_argument('--rewrite-mode' , default='editing', help='choose from editing, local_finetuning, global_finetuning')
parser.add_argument('--layernum', '-m', default=12, type=int, help='layer to rewrite')
parser.add_argument('--proj', '-p', default='EditingClassifiers', type=str, help='project name')
parser.add_argument('--pattern-file', default='./data/waterbirds/forest_broadleaf.jpg', help='pattern file for waterbirds')
parser.add_argument('--restrict-rank', action='store_true', help="if true, does not perform low-rank update")
parser.add_argument('--nsteps-proj', default=10, type=int, help='Frequency of weight projection')
parser.add_argument('--rank', default=1, type=int, help='Rank of subspace to project weights')
parser.add_argument('--ntrain', default=1, type=int, help='Number of training samples')
parser.add_argument('--test-ext', action='store_true', help='extended test set')
args = parser.parse_args()

run = wandb.init(project=args.proj, group=args.dataset, config=args)

DATASET_NAME = args.dataset
LAYERNUM = args.layernum
REWRITE_MODE = args.rewrite_mode
ARCH = args.arch

# load model
ret = classifier_helpers.get_default_paths(DATASET_NAME, arch=ARCH)
DATASET_PATH, MODEL_PATH, MODEL_CLASS, ARCH, CD = ret
ret = classifier_helpers.load_classifier(MODEL_PATH, MODEL_CLASS, ARCH,
                            DATASET_NAME, LAYERNUM) 
model, context_model, target_model = ret[:3]


# load data
base_dataset, train_loader, val_loader = dh.get_dataset(DATASET_NAME, '/shared/group/ilsvrc',
                                                        batch_size=32, workers=8)
if args.dataset == 'Waterbirds':
    train_data, test_data = dh.get_waterbirds_data(pattern_img_path=args.pattern_file)
elif args.dataset == 'WaterbirdsSimple':
    train_data, test_data = dh.get_waterbirds_simple_data()
else:
    train_data, test_data, test_ext_data = dh.get_vehicles_on_snow_data(DATASET_NAME, CD)
    if args.test_ext:
        test_data = test_ext_data

# wandb save training
wandb_imgs = [wandb.Image(i) for i in train_data['imgs']]
wandb.log({'training images': wandb_imgs})

print("Original accuracy on testset")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
# Get orig accuracy
RESULTS = {k: {'preds': {}, 'acc': {}} for k in ['pre', 'post']}
GLOBAL_RESULTS = {k: {'preds': {}, 'acc': {}} for k in ['pre', 'post']}
sub_batch = 32
for c, x in test_data.items():
    generator = chunks(x, sub_batch)
    correct = []
    for b in generator:
        with ch.no_grad():
            pred = model(b.cuda()).argmax(axis=1)
        correct += [p for p in pred if p == c]
    acc = 100 * len(correct) / len(x)
    print(f'Class: {c}/{CD[c]} | Accuracy: {acc:.2f}',) 
    RESULTS['pre']['acc'][c] = acc
    RESULTS['pre']['preds'][c] = pred

print('------- SNOW CLASSES -------')
if 'Waterbirds' not in args.dataset:
    accuracy, pred, pre_results = nt.test_imagenet_snow(model)
    for c in pre_results:
        print(f'Class: {c}/{CD[c]} | Accuracy: {pre_results[c]:.2f}',) 
        wandb.log({f"{CD[c]} pre acc": pre_results[c]})

# edit
train_args = {'ntrain': args.ntrain, # Number of exemplars
            'arch': ARCH, # Network architecture
            'mode_rewrite': REWRITE_MODE, # Rewriting method ['editing', 'finetune_local', 'finetune_global']
            'layernum': LAYERNUM, # Layer to modify
            'nsteps': 20000 if REWRITE_MODE == 'editing' else 400, # Number of rewriting steps  
            'lr': args.lr, # Learning rate
            'restrict_rank': not args.restrict_rank, # Whether or not to perform low-rank update
            'nsteps_proj': args.nsteps_proj, # Frequency of weight projection
            'rank': args.rank, # Rank of subspace to project weights
            'use_mask': True # Whether or not to use mask
             }
train_args = Namespace(**train_args)

context_model = rh.edit_classifier(train_args, 
                                   train_data, 
                                   context_model, 
                                   target_model=target_model, 
                                   val_loader=val_loader,
                                   caching_dir=f"./cache/covariances/{DATASET_NAME}_{ARCH}_layer{LAYERNUM}") # caching_dir=f"./cache/covariances/{DATASET_NAME}_{ARCH}_layer{LAYERNUM}"

# eval modified classifier
print("Change in accuracy on testset \n")
changes = []
for c, x in test_data.items():
    generator = chunks(x, sub_batch)
    correct = []
    for b in generator:
        with ch.no_grad():
            pred = model(b.cuda()).argmax(axis=1)
        correct += [p for p in pred if p == c]
    acc = 100 * len(correct) / len(x)
    print(f'Class: {c}/{CD[c]} \n Accuracy change: {RESULTS["pre"]["acc"][c]:.2f} -> {acc:.2f} \n',) 
    RESULTS['post']['acc'][c] = acc
    RESULTS['post']['preds'][c] = pred
    changes.append(acc - RESULTS["pre"]["acc"][c])
    wandb.summary[f"{CD[c]} acc dif"] = acc - RESULTS["pre"]["acc"][c]

print('------- SNOW CLASSES -------')
if 'Waterbirds' not in args.dataset:
    accuracy, pred, post_results = nt.test_imagenet_snow(model)
    for c in post_results:
        print(f'Class: {c}/{CD[c]} | Accuracy: {post_results[c]:.2f}',) 
        wandb.log({f"{CD[c]} post acc": post_results[c]})
        wandb.summary[f"{CD[c]} acc diff"] = post_results[c] - pre_results[c]
    wandb.log({'pre acc': RESULTS['pre']['acc'], 'post acc': RESULTS['post']['acc'], 'imagenet snow pre': pre_results, 'imagenet snow post': post_results})
else:
    wandb.log({'pre acc': RESULTS['pre']['acc'], 'post acc': RESULTS['post']['acc']})
wandb.summary['avg acc change'] = np.mean(changes)