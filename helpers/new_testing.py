import sys, os, warnings
from argparse import Namespace
warnings.filterwarnings("ignore")

import torch as ch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

from helpers import classifier_helpers
import helpers.data_helpers as dh
import helpers.context_helpers as coh
import helpers.rewrite_helpers as rh
import helpers.vis_helpers as vh

import matplotlib.pyplot as plt
import random

import torchvision
import torch
from torchvision import transforms

from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

import ast 
with open('/home/lisabdunlap/pytorch-cifar/notebooks/imagenet1000_clsidx_to_labels.txt') as f:
    data = f.read()
    
d = ast.literal_eval(data)
IMAGENET_CLASSES = {}
for k in d:
    IMAGENET_CLASSES[k] = d[k].split(', ')

def test_imagenet_val(model, GLOBAL_RESULTS, mode='pre'):

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    imagenet_testset = torchvision.datasets.ImageFolder('/shared/group/ilsvrc/val', transform=transform)
    imagenet_loader = torch.utils.data.DataLoader(imagenet_testset, num_workers=2,
                                                batch_size=50, shuffle=False)
    total, correct = 0, 0
    for i, (x, targets) in enumerate(imagenet_loader):
        with ch.no_grad():
            try:
                pred = model(x.cuda()).argmax(axis=1)
            except:
                logits, att_mat = model(x.cuda())
                probs = torch.nn.Softmax(dim=-1)(logits)
                pred = probs.argmax(axis=1)
            targets = targets.cuda()
            class_correct = pred.eq(targets).sum().item()
            correct += class_correct
            total += targets.size(0)
            GLOBAL_RESULTS[mode]['acc'][i] = 100 * class_correct / targets.size(0)
            GLOBAL_RESULTS[mode]['preds'][i] = pred
    return 100 * correct / total

def get_displ_img(img, norm=False):
    try:
        img = img.cpu().numpy().transpose((1, 2, 0))
    except:
        img = img.numpy().transpose((1, 2, 0))
    if norm:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
    displ_img = np.clip(img, 0, 1)
    displ_img /= np.max(displ_img)
    displ_img = displ_img
    displ_img = np.uint8(displ_img*255)
    return displ_img/np.max(displ_img)

def test_internet_car_wheel(model):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                          std=[0.229, 0.224, 0.225]),
        ])
    # internet_wheel_test = torchvision.datasets.ImageFolder('./car_wheels_in_snow', transform=transform)
    internet_wheel_test = InternetSnowDataset('./internet_snow', transform=transform)
    internet_wheel_loader = torch.utils.data.DataLoader(internet_wheel_test, num_workers=2,
                                                    batch_size=50, shuffle=False)

    imagenet_snow_dataset = ImagenetSnow(transforms=transform)
    imagenet_snow_loader = torch.utils.data.DataLoader(imagenet_snow_dataset, num_workers=2,
                                                    batch_size=50, shuffle=False)
    preds = []
    correct, total = 0, 0
    for i, (inp, target) in enumerate(internet_wheel_loader):
        with ch.no_grad():
            pred = model(inp.cuda()).argmax(axis=1)
        preds += [p for p in pred.cpu().numpy()]
        # print(target)
        correct += len([p for p,l in zip(pred, target) if p == l])
        total += target.size(0)
        
    acc = 100 * correct/total
        
    incorrect_imgs = []
    incorrect_preds = []
    for i, p in enumerate(preds):
        if p != 479:
            incorrect_imgs.append(get_displ_img(internet_wheel_test[i][0]))
            incorrect_preds.append(IMAGENET_CLASSES[p])
            
    return acc, preds, incorrect_imgs, incorrect_preds

def test_imagenet_snow(model, vit=False):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                          std=[0.229, 0.224, 0.225]),
        ])

    results = {}
    imagenet_snow_dataset = ImagenetSnow(transforms=transform)
    imagenet_snow_loader = torch.utils.data.DataLoader(imagenet_snow_dataset, batch_size=50, shuffle=False)
    preds = []
    correct, total = 0, 0
    for i, (inp, target) in enumerate(imagenet_snow_loader):
        assert(len(set(target.cpu().numpy())) == 1)
        with ch.no_grad():
            if vit:
                logits, att_mat = model(inp.cuda())
                probs = torch.nn.Softmax(dim=-1)(logits)
                pred = probs.argmax(axis=1)
            else:
                pred = model(inp.cuda()).argmax(axis=1)
        preds += [p for p in pred.cpu().numpy()]
        # print(target)
        correct += len([p for p,l in zip(pred, target) if p == l])
        total += target.size(0)
        results[int(target.cpu().numpy()[0])] = 100 * len([p for p,l in zip(pred, target) if p == l])/target.size(0)
        
    acc = 100 * correct/total
            
    return acc, preds, results

def save_examples(incorrect_imgs, incorrect_preds, show=False, save_file='post_incorrect_preds.png'):
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                     axes_pad=0.4,  # pad between axes in inch.
                     )

    for ax, im, lab in zip(grid, incorrect_imgs, incorrect_preds):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(lab)

    plt.axis('off')
    # plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight')
    if show:
        plt.show()

class ImagenetSnow(torchvision.datasets.ImageFolder):

    def __init__(self, root='data/imagenet_snow.csv', transforms=None):
        self.df = pd.read_csv(root)
        self.transforms = transforms
        self.samples = [(row['Filename'], row['Label']) for i, row in self.df.iterrows()]

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]['Filename']).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        label = self.df.iloc[idx]['Label']
        return img, label

    def __len__(self):
        return len(self.df)

# ['army tank in snow',
#  'car wheel in snow',
#  'fire truck in snow',
#  'motor scooter in snow'
#  'racecar in snow',
#  'school bus in snow',
#  'stop light in snow']
#  Class: 920/traffic light, traffic signal, stoplight 

# Class: 479/car wheel 

# Class: 670/motor scooter, scooter 

# Class: 751/racer, race car, racing car 

# Class: 555/fire engine, fire truck 

# Class: 847/tank, army tank, armored combat vehicle, armoured combat vehicle 

# Class: 779/school bus 
class InternetSnowDataset(torchvision.datasets.ImageFolder):

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        if label == 0:
            return img, 847
        elif label == 1:
            return img, 479
        elif label == 2:
            return img, 555
        elif label == 3:
            return img, 670
        elif label == 4:
            return img, 751
        elif label == 5:
            return img, 779
        elif label == 6:
            return img, 920
        