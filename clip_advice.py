import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import wandb
import argparse

from helpers import classifier_helpers
import helpers.data_helpers as dh
import helpers.context_helpers as coh
import helpers.rewrite_helpers as rh
import helpers.vis_helpers as vh

import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix

import helpers.new_testing as nst
from helpers.new_testing import IMAGENET_CLASSES
from helpers.waterbirds import Waterbirds

import helpers.clip_transformations as CLIPTransformations

from helpers.waterbirds import Waterbirds, WaterbirdsEditing

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default="WaterbirdsTiny", help='dataset')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--arch', default='resnet50', help='model')
parser.add_argument('--pattern-file', default='./data/waterbirds/forest_broadleaf.jpg', help='pattern file for waterbirds')
parser.add_argument('--proj', '-p', default='CLIPAdvice', type=str, help='project name')
parser.add_argument('--wandb-silent', action='store_true', help='turn off wandb logging')
parser.add_argument('--nsteps', type=int, help='num steps for training')
parser.add_argument('--binary', action='store_true')
parser.add_argument('--advice-method', default='Noop', help='technique to apply to clip embeddings')
parser.add_argument('--seed', default=0, type=int, help="random seed")
args = parser.parse_args()

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

run = wandb.init(project=args.proj, group=args.dataset, config=args)

DATASET_NAME = args.dataset

# load data
base_dataset, train_loader, val_loader = dh.get_dataset(DATASET_NAME, '/shared/group/ilsvrc',
                                                        batch_size=1, workers=8)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataset):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
text_prompts = ["an image of land", "an image of water"]
# land_prompts = ["a photo of land", "an image of land", "a photo of a forest", "an image of a forest"]
# water_prompts = [ "a photo of water", "an image of water", "a photo of a lake", "an image of a lake"]
# text_prompts = [land_prompts, water_prompts]
# text_prompts = [["an image of grass", "a photo of grass", "an image of a field", "a photo of a field"], ["an image of a road", "a photo of a road", "an image of pavement", "a photo of pavement"]]
# text_prompts = ["an image of grass", "an image of road"]
wandb.config.text_prompts = text_prompts
bias_correction = getattr(CLIPTransformations, args.advice_method)(text_prompts, model)
train_features, train_labels = get_features(train_loader)
# apply 'advice' module
train_features = bias_correction.apply(train_features)
test_features, test_labels = get_features(val_loader)
test_features = bias_correction.apply(test_features)

best_acc, best_c, best_class_acc = 0, 0, []
# Perform logistic regression
for c in [0.0001, 0.0005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5, 2.0, 3.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]:
    classifier = LogisticRegression(random_state=args.seed, C=c, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    cf_matrix = confusion_matrix(test_labels, predictions)
    class_accuracy=100*cf_matrix.diagonal()/cf_matrix.sum(1)
    # print("Class accuracy ", class_accuracy)
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    balanced_acc = class_accuracy.mean()
    # print(f"C = {c} \t Accuracy = {accuracy:.3f}")
    wandb.summary[f"C = {c} Acc"] = accuracy
    wandb.summary[f"C = {c} Balanced Class Acc"] = balanced_acc
    wandb.summary[f"C = {c} Class Acc"] = [round(c, 2) for c in class_accuracy]
    if balanced_acc > best_acc:
        best_acc, best_c, best_class_acc = balanced_acc, c, [round(c, 2) for c in class_accuracy]
wandb.summary['best acc'] = best_acc
wandb.summary['best c'] = best_c
wandb.summary['best class acc'] = best_class_acc