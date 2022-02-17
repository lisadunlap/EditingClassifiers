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

import helpers.clip_transformations as CLIPTransformations
from helpers.clip_transformations import evaluate

from omegaconf import OmegaConf
import omegaconf

parser = argparse.ArgumentParser(description='CLIP Advice')
parser.add_argument('--config', default='configs/Noop.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
flags = parser.parse_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/Noop.yaml')
args      = OmegaConf.merge(base, cfg, overrides)
args.yaml = flags.config

if args.EXP.WANDB_SILENT:
    os.environ['WANDB_SILENT']="true"

def flatten_config(dic, running_key=None, flattened_dict={}):
    for key, value in dic.items():
        if running_key is None:
            running_key_temp = key
        else:
            running_key_temp = '{}.{}'.format(running_key, key)
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            flatten_config(value, running_key_temp)
        else:
            #print(running_key_temp, value)
            flattened_dict[running_key_temp] = value
    return flattened_dict

run = wandb.init(project=args.EXP.PROJ, group=args.DATA.DATASET, config=flatten_config(args))
wandb.save(flags.config)
wandb.run.log_code(".")

DATASET_NAME = args.DATA.DATASET

# load data
if args.DATA.LOAD_FROM:
    data = torch.load(args.DATA.LOAD_FROM)
    train_features, train_labels = data['train_features'], data['train_labels']
    val_features, val_labels = data['val_features'], data['val_labels']
    test_features, test_labels = data['test_features'], data['test_labels']
else:
    base_dataset, train_loader, val_loader, test_loader = dh.get_dataset(DATASET_NAME, '/shared/group/ilsvrc',
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
prompts = list(args.EXP.TEXT_PROMPTS)
if type(prompts[0]) == omegaconf.listconfig.ListConfig:
    prompts = [list(p) for p in prompts]
bias_correction = getattr(CLIPTransformations, args.EXP.ADVICE_METHOD)(prompts, model, args)
if args.DATA.LOAD_FROM ==  None:
    train_features, train_labels = get_features(train_loader)
    val_features, val_labels = get_features(val_loader)
    test_features, test_labels = get_features(test_loader)
    data = {
        "train_features": train_features,
        "train_labels": train_labels,
        "val_features": val_features,
        "val_labels": val_labels,
        "test_features": test_features,
        "test_labels": test_labels
    }
    torch.save(data, "data/waterbirds/clip_embeddings.pth")
if args.EXP.ADVICE_METHOD == "MLPDebias":
    # train MLP with domain adaptation loss
    bias_correction.train_debias(train_features, train_labels, val_features, val_labels)
    predictions = bias_correction.eval(test_features)
    print(predictions.shape)
    accuracy, balanced_acc, class_accuracy = evaluate(predictions, test_labels)
    wandb.summary["test acc"] = accuracy
    wandb.summary["test blanced acc"] = balanced_acc
    wandb.summary["test class acc"] = class_accuracy
else:
    # apply 'advice' module
    train_features = bias_correction.apply(train_features)
    val_features = bias_correction.apply(val_features)
    test_features = bias_correction.apply(test_features)

    best_acc, best_c, best_class_acc = 0, 0, []
    # Perform logistic regression
    for c in [0.0001, 0.0005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5, 2.0, 3.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]:
        classifier = LogisticRegression(random_state=args.EXP.SEED, C=c, max_iter=1000, verbose=1)
        classifier.fit(train_features, train_labels)

        # Evaluate using the logistic regression classifier
        predictions = classifier.predict(val_features)
        accuracy, balanced_acc, class_accuracy = evaluate(predictions, val_labels)
        
        # print(f"C = {c} \t Accuracy = {accuracy:.3f}")
        wandb.summary[f"C = {c} Acc"] = accuracy
        wandb.summary[f"C = {c} Balanced Class Acc"] = balanced_acc
        wandb.summary[f"C = {c} Class Acc"] = [round(c, 2) for c in class_accuracy]
        if balanced_acc > best_acc:
            best_acc, best_c, best_class_acc = balanced_acc, c, [round(c, 2) for c in class_accuracy]
    wandb.summary['best acc'] = best_acc
    wandb.summary['best c'] = best_c
    wandb.summary['best class acc'] = best_class_acc

    # run on test set
    classifier = LogisticRegression(random_state=args.EXP.SEED, C=best_c, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy, balanced_acc, class_accuracy = evaluate(predictions, test_labels)
    wandb.summary["test acc"] = accuracy
    wandb.summary["test blanced acc"] = balanced_acc
    wandb.summary["test class acc"] = class_accuracy