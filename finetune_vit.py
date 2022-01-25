import typing
import io
import os
import sys, os, warnings
from argparse import Namespace
warnings.filterwarnings("ignore")

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms

from models.vit import VisionTransformer, CONFIGS

from torchvision import transforms
from tqdm import tqdm
import torchvision

from scipy import spatial
import argparse

from helpers import classifier_helpers
import helpers.data_helpers as dh
import helpers.context_helpers as coh
import helpers.rewrite_helpers as rh
import helpers.vis_helpers as vh

import matplotlib.pyplot as plt
import random
import wandb

import helpers.new_testing as nt
from helpers.new_testing import IMAGENET_CLASSES

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--dataset', default="ImageNet", help='dataset')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--checkpoint-name' , '-c', type=str, default=None, help="name of checkpoint")
parser.add_argument('--model', '-m', type=str, default='ResNet18', help='model architecture')
parser.add_argument('--layernum', type=str, default='5.attn', help='layer to finetune')
parser.add_argument('--eval-imagenet', action='store_true', help='evaluate on all of imagenet')
parser.add_argument('--ntrain', type=int, default=5, help='number of training examples')
parser.add_argument('--niter', type=int, default=100, help='number of training iterations')
parser.add_argument('--edit-type', default='local_finetune', choices=['local_finetune', 'global_finetune', 'editing_local', 'editing_global', 'editing_scores', 'edit_mask'])
parser.add_argument('--proj', default='finetuneViT', type=str, help='WandB project name')
parser.add_argument('--test-ext', action='store_true', help='extended test set')
parser.add_argument('--match-context', action='store_true', help='loss is between context')
parser.add_argument('--l1', action='store_true', help='L1 instead of L2 loss')
args = parser.parse_args()


run = wandb.init(project=args.proj, group=args.edit_type, config=args)

os.makedirs("attention_data", exist_ok=True)
if not os.path.isfile("attention_data/ilsvrc2012_wordnet_lemmas.txt"):
    urlretrieve("https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt", "attention_data/ilsvrc2012_wordnet_lemmas.txt")
if not os.path.isfile("attention_data/ViT-B_16-224.npz"):
    urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz", "attention_data/ViT-B_16-224.npz")

imagenet_labels = dict(enumerate(open('attention_data/ilsvrc2012_wordnet_lemmas.txt')))

DATASET_NAME = args.dataset
LAYERNUM = 12
REWRITE_MODE = 'editing'
ARCH = 'vgg16'
ret = classifier_helpers.get_default_paths(DATASET_NAME, arch=ARCH)
DATASET_PATH, MODEL_PATH, MODEL_CLASS, ARCH, CD = ret
ret = classifier_helpers.load_classifier(MODEL_PATH, MODEL_CLASS, ARCH,
                            DATASET_NAME, LAYERNUM) 

model, context_model, target_model = ret[:3]
if args.dataset == 'Waterbirds':
    train_data, test_data = dh.get_waterbirds_data(pattern_img_path=args.pattern_file)
elif args.dataset == 'WaterbirdsSimple':
    train_data, test_data = dh.get_waterbirds_simple_data()
else:
    train_data, test_data, test_ext_data = dh.get_vehicles_on_snow_data(DATASET_NAME, CD)
    if args.test_ext:
        test_data = test_ext_data
if args.test_ext:
    test_data = test_ext_data

base_dataset, train_loader, val_loader = dh.get_dataset(DATASET_NAME, '/shared/group/ilsvrc',
                                                        batch_size=16, workers=8)

# Prepare Model
config = CONFIGS["ViT-B_16"]
model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
model.load_from(np.load("attention_data/ViT-B_16-224.npz"))
model = model.cuda()
model.eval()

print("Original accuracy on test vehicles-on-snow data")

RESULTS = {k: {'preds': {}, 'acc': {}} for k in ['pre', 'post']}
GLOBAL_RESULTS = {k: {'preds': {}, 'acc': {}} for k in ['pre', 'post']}

def eval(test_data, model, RESULTS, mode='pre', verbose=True):
    for c, x in test_data.items():
        with torch.no_grad():
            logits, att_mat = model(x.cuda())
            probs = torch.nn.Softmax(dim=-1)(logits)
            pred = probs.argmax(axis=1)
        correct = [p for p in pred if p == c]
        acc = 100 * len(correct) / len(x)
        if verbose:
            print(f'Class: {c}/{CD[c]} | Accuracy: {acc:.2f}',) 
        RESULTS[mode]['acc'][c] = acc
        RESULTS[mode]['preds'][c] = pred
    
eval(test_data, model, RESULTS, verbose=False)
if args.eval_imagenet: 
    RESULTS['pre']['acc']['global'] = nt.test_imagenet_val(model, GLOBAL_RESULTS)

print('------- SNOW CLASSES -------')
if 'Waterbirds' not in args.dataset:
    accuracy, pred, pre_results = nt.test_imagenet_snow(model, vit=True)
    for c in pre_results:
        print(f'Class: {c}/{CD[c]} | Accuracy: {pre_results[c]:.2f}',) 
        wandb.log({f"{CD[c]} pre acc": pre_results[c]})

model.train()
if 'editing' in args.edit_type:
    attns = {}
    def hook_act(module, input, output):
        attns['context'] = output[0]
        attns['scores'] = output[1]

    attns2 = {}
    def hook_act2(module, input, output):
        attns2['context'] = output[0]
        attns2['scores'] = output[1]
    
    if 'attn' in args.layernum:
        h = model.transformer.encoder.layer[int(args.layernum.replace('.attn', ''))].attn.register_forward_hook(hook_act)
    else:
        h = model.transformer.encoder.layer[int(args.layernum)].register_forward_hook(hook_act)

    def get_goal_attn(model, imgs):
        model.eval()
        with torch.no_grad():
            model(imgs.cuda())
        h.remove()
        model.train()
        return attns

    get_goal_attn(model, train_data['imgs'][:args.ntrain].float())

    # now train
    if 'attn' in args.layernum:
        hh = model.transformer.encoder.layer[int(args.layernum.replace('.attn', ''))].attn.register_forward_hook(hook_act2)
    else:
        hh = model.transformer.encoder.layer[int(args.layernum)].register_forward_hook(hook_act2)

    if args.edit_type == 'editing_local':
        for name, param in model.named_parameters():
            if f'transformer.encoder.layer.{args.layernum}' not in name:
                param.requires_grad = False
    elif args.edit_type == 'editing_global':
        layer = int(args.layernum.split('.')[0])
        for name, param in model.named_parameters():
            if f'transformer.encoder.layer' not in name:
                param.requires_grad = False
            else:
                if layer > int(name.split('transformer.encoder.layer.')[-1].split('.')[0]):
                    param.requires_grad = False
    elif args.edit_type == 'editing_scores':
        assert 'attn' in args.layernum, 'need to pass in an attention layer'
        for name, param in model.named_parameters():
            if (f'transformer.encoder.layer.{args.layernum}.query' not in name) and (f'transformer.encoder.layer.{args.layernum}.key' not in name):
                param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.l1:
        compute_loss = torch.nn.L1Loss()
    else:
        compute_loss = torch.nn.MSELoss()
    pbar = tqdm(range(args.niter))

    imgs = train_data['modified_imgs'][:args.ntrain].float()
    target_label = np.unique(train_data['labels'][:args.ntrain].numpy())
    assert len(target_label) == 1

    tgts = torch.tensor([target_label[0]] * len(imgs))

    with torch.enable_grad():
        for i in pbar:
            # loss = compute_loss(model(imgs.cuda())[0], tgts.long().cuda())
            model(imgs.cuda())
            loss = compute_loss(attns2['scores'], attns['scores'])
            optimizer.zero_grad()
            loss.backward()
            pbar.set_description(str(loss))
            optimizer.step()
            wandb.log({'loss': loss.item()})
    loss.detach()
    hh.remove()
elif args.edit_type == 'edit_mask':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.l1:
        compute_loss = torch.nn.L1Loss()
    else:
        compute_loss = torch.nn.MSELoss()
    pbar = tqdm(range(args.niter))

    def get_attn_mask(att_mats):
        ret = []
        for att_mat in att_mats:
            print(len(att_mats), att_mat.shape)
            att_mat = att_mat.squeeze(1)

            # Average the attention weights across all heads.
            att_mat = torch.mean(att_mat, dim=1)

            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mat.size(1)).cuda()
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
            joint_attentions[0] = aug_att_mat[0]

            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
            ret.append(joint_attentions[-1])
        return torch.stack(ret)

    model.eval()
    imgs = train_data['imgs'][:args.ntrain].float()
    target_label = np.unique(train_data['labels'][:args.ntrain].numpy())
    with torch.no_grad():
        # loss = compute_loss(model(imgs.cuda())[0], tgts.long().cuda())
        logits, att_mat = model(imgs.cuda())
        orig_masks = get_attn_mask(att_mat)
    
    # train 
    model.train()
    imgs = train_data['modified_imgs'][:args.ntrain].float()
    target_label = np.unique(train_data['labels'][:args.ntrain].numpy())
    assert len(target_label) == 1
    with torch.enable_grad():
        for i in pbar:
            # loss = compute_loss(model(imgs.cuda())[0], tgts.long().cuda())
            logits, att_mat = model(imgs.cuda())
            mask = get_attn_mask(att_mat)
            print(mask.shape)
            loss = compute_loss(mask, orig_masks)
            optimizer.zero_grad()
            loss.backward()
            pbar.set_description(str(loss))
            optimizer.step()
            wandb.log({'loss': loss.item()})
    loss.detach()
else:
    if args.edit_type == 'local_finetune':
        # fintune ONLY the layer specified 
        for name, param in model.named_parameters():
            if f'transformer.encoder.layer.{args.layernum}' not in name:
                param.requires_grad = False
    elif args.edit_type == 'global_finetune':
        # finetune all layers up to the layer specified
        layer = int(args.layernum.split('.')[0])
        for name, param in model.named_parameters():
            if f'transformer.encoder.layer' not in name:
                param.requires_grad = False
            else:
                if layer > int(name.split('transformer.encoder.layer.')[-1].split('.')[0]):
                    param.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    compute_loss = torch.nn.CrossEntropyLoss()
    pbar = tqdm(range(args.niter))

    imgs = train_data['modified_imgs'][:args.ntrain].float()
    target_label = np.unique(train_data['labels'][:args.ntrain].numpy())
    assert len(target_label) == 1

    tgts = torch.tensor([target_label[0]] * len(imgs))

    with torch.enable_grad():
        for i in pbar:
            loss = compute_loss(model(imgs.cuda())[0], tgts.long().cuda())
            optimizer.zero_grad()
            loss.backward()
            pbar.set_description(str(loss))
            optimizer.step()
            wandb.log({'loss': loss.item()})
    loss.detach()

model.eval()

print("Change in accuracy on test vehicles-on-snow data \n")
acc_changes = []
for c, x in test_data.items():
    with torch.no_grad():
        logits, att_mat = model(x.cuda())
        probs = torch.nn.Softmax(dim=-1)(logits)
        pred = probs.argmax(axis=1)
    correct = [p for p in pred if p == c]
    acc = 100 * len(correct) / len(x)
    print(f'Class: {c}/{CD[c]} \n Accuracy change: {RESULTS["pre"]["acc"][c]:.2f} -> {acc:.2f} \n',) 
    RESULTS['post']['acc'][c] = acc
    RESULTS['post']['preds'][c] = pred
    wandb.summary[f'{CD[c]} acc change'] = acc - RESULTS["pre"]["acc"][c]
    acc_changes.append(acc - RESULTS["pre"]["acc"][c])
wandb.summary['global acc change'] = np.mean(acc_changes)

print('------- SNOW CLASSES -------')
if 'Waterbirds' not in args.dataset:
    accuracy, pred, post_results = nt.test_imagenet_snow(model, vit=True)
    for c in post_results:
        print(f'Class: {c}/{CD[c]} | Accuracy: {post_results[c]:.2f}',) 
        wandb.log({f"{CD[c]} post acc": post_results[c]})
        wandb.summary[f"{CD[c]} acc diff"] = post_results[c] - pre_results[c]
    wandb.log({'pre acc': RESULTS['pre']['acc'], 'post acc': RESULTS['post']['acc'], 'imagenet snow pre': pre_results, 'imagenet snow post': post_results})
else:
    wandb.log({'pre acc': RESULTS['pre']['acc'], 'post acc': RESULTS['post']['acc']})

if args.eval_imagenet:
    RESULTS['post']['acc']['global'] = nt.test_imagenet_val(model, GLOBAL_RESULTS)
    print(f'Global \n Accuracy change: {RESULTS["pre"]["acc"]["global"]:.2f} -> {RESULTS["post"]["acc"]["global"]:.2f} \n',) 
    wandb.summary['imagenet acc change'] = RESULTS["post"]["acc"]["global"] - RESULTS["pre"]["acc"]["global"]

# wandb.log(RESULTS)
