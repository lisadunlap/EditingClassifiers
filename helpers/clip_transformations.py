import os

from click import pass_obj
import clip
import torch
import torch.nn as nn
import torch.nn.functional as nnf

import wandb

import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Tuple, Optional, Union
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA
from pytorch_revgrad import RevGrad
from utils import progress_bar
from omegaconf import OmegaConf

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(predictions, labels):
    cf_matrix = confusion_matrix(labels, predictions)
    class_accuracy=100*cf_matrix.diagonal()/cf_matrix.sum(1)
    # print("Class accuracy ", class_accuracy)
    accuracy = np.mean((labels == predictions).astype(np.float)) * 100.
    balanced_acc = class_accuracy.mean()
    return accuracy, balanced_acc, class_accuracy


class Noop:
    """
    Does nothing.
    """
    def __init__(self, text_prompts, model, cfg):
        self.text_prompts = text_prompts
        self.text_embeddings = []

    def apply(self, inputs):
        return inputs

class BiasCorrection(Noop):
    """
    Simply subtracts the embeddings of the text prompts
    """
    def __init__(self, text_prompts, model, cfg):
        self.text_prompts = text_prompts
        text_inputs = torch.cat([clip.tokenize(t) for t in text_prompts]).to(device)
        # Calculate features
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)

        self.text_embeddings = text_features.cpu().numpy()

    def apply(self, inputs):
        ret = []
        for i in range(inputs.shape[0]):
            altered = i - np.sum(self.text_embeddings, axis=0)
            ret.append(altered)
        return np.array(ret)

    def calc_dist(self, inputs, labels):
        ret = [["label", "text", "text id", "sim"]]
        for i in range(inputs.shape[0]):
            for j in range(len(self.text_prompts)):
                dist = 1 - distance.cosine(inputs[i], self.text_embeddings[j])
                ret.append([int(labels[i]), self.text_prompts[j], j, dist])
        return pd.DataFrame(ret[1:], columns=ret[0])

class Project(BiasCorrection):

    def apply(self, inputs):
        ret = []
        for i in range(inputs.shape[0]):
            cat = []
            for j in range(self.text_embeddings.shape[0]):
                a = inputs[i]
                b = self.text_embeddings[j]
                proj = (np.dot(a, b) / np.dot(b, b)) * b
                cat.append(proj)
            ret.append(np.mean(np.array(cat), axis=0))
        return np.array(ret)

class SentenceDebias:

    def __init__(self, prompts, model, cfg):
        # WARNING: prompts should be a list of lists
        assert type(prompts[0]) == list
        self.prompts = prompts
        self.model = model
        self.d = len(self.prompts[0])
        self.embs = []
        self.bias_vectors = []
        for i in range(len(self.prompts)):
            emb = self.get_embedding(self.prompts[i], model)
            self.embs.append(emb)
            self.bias_vectors += [v - np.mean(emb, axis=0) for v in emb]
        pca = PCA(n_components=cfg.EXP.K, svd_solver='full')
        pca.fit(self.bias_vectors)
        self.bias_space = pca.components_

    @staticmethod
    def get_embedding(prompts, model):
        text_inputs = torch.cat([clip.tokenize(t) for t in prompts]).cuda()
        # Calculate features
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        return text_features.cpu().numpy()

    def apply(self, inputs):
        ret = []
        for h in inputs:
            h_v = np.sum([(np.dot(h, b) / np.dot(b, b)) * b for b in self.bias_space], axis=0)
            ret.append(h - h_v)
        return np.array(ret)

def zeroshot_classifier(prompts, model):
    assert type(prompts[0]) == list, "prompts must be a list of lists"
    with torch.no_grad():
        zeroshot_weights = []
        for texts in tqdm(prompts):
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.num_layers = cfg.MODEL.NUM_LAYERS
        # assert num_layers in [1,2], 'Only one or two # layers supported'
        if self.num_layers == 1:
            self.fc = nn.Linear(cfg["in_dim"], cfg["out_dim"])
        else:
            self.fc1 = nn.Linear(cfg["in_dim"], int(cfg["in_dim"]/2))
            self.fc2 = nn.Linear(int(cfg["in_dim"]/2), cfg["out_dim"])

    def forward(self, x):
        if self.num_layers == 1:
            x = self.fc(x)
        else:
            x = nnf.relu(self.fc1(x))
            x = self.fc2(x)
        return x

class EmbeddingDebiasModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.mlp = MLP(cfg)
        self.classifier_head = nn.Linear(cfg["out_dim"], 1 if cfg["num_classes"]==2 else cfg["num_classes"])
        self.domain_head = nn.Sequential(nn.Linear(cfg["out_dim"], 1 if cfg["num_domains"]==2 else cfg["num_domains"]), RevGrad(alpha=cfg.MODEL.DOM_WEIGHT))

    def forward(self, x):
        x = self.mlp(x)
        c = self.classifier_head(x)
        d = self.domain_head(x)
        return c, d


class EmbeddingDataset:
    """
    Takes in CLIP embeddings (INPUTS), labels, and CLIP text embedding (TEXT_EMB of shape (num_domains, clip emb shape)).
    Weakly labels the domain using the text embeddings 
    TODO: try softlabels
    """
    def __init__(self, inputs, labels, text_emb):
        self.inputs, self.labels = inputs, labels
        self.text_emb = text_emb
        self.domain_labels = self.get_labels(self.text_emb, self.inputs)
        self.num_classes, self.num_domains = len(set(self.labels)), len(set(self.domain_labels))
        assert len(self.inputs) == len(self.labels) == len(self.domain_labels), "input, label, and domain label lengths don't match"

    @staticmethod
    def get_labels(text_emb, inputs):
        """ Gets weak domain labels given CLIP text embeddings """
        similarity = (100.0 * torch.Tensor(inputs).cuda().float() @ text_emb.T.float()).softmax(dim=-1)
        values, indices = similarity.topk(1)
        return [i[0].item() for i in indices]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.domain_labels[idx]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class MLPDebias(Noop):

    def __init__(self, text_prompts, model, cfg):
        super().__init__(text_prompts, model, cfg)
        self.text_emb = zeroshot_classifier(text_prompts, model).T # translates text-> embedding
        self.cfg = cfg

    def train_debias(self, inputs, labels, test_inputs, test_labels):
        B, W  = inputs.shape
        self.train_dataset = EmbeddingDataset(inputs, labels, self.text_emb)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=True)
        self.test_dataset = EmbeddingDataset(test_inputs, test_labels, self.text_emb)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=False)

        model_conf = OmegaConf.create({"in_dim": W, "h_dim": W, "out_dim": W, "num_classes": self.train_dataset.num_classes, "num_domains": self.train_dataset.num_domains})
        self.cfg = OmegaConf.merge(self.cfg, model_conf)
        self.net = EmbeddingDebiasModel(self.cfg)
        self.net = self.net.cuda()
        self.optimizer = AdamW(self.net.parameters(), lr=self.cfg.MODEL.LR)
        if model_conf.num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss()
            self.m = nn.Sigmoid()
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.m = nn.Softmax()

        self.best_acc, self.best_epoch = 0, 0
        for epoch in range(50):
            self.train(epoch)
            self.test(epoch)

    def train(self, epoch):
        self.net.train()
        train_cls_loss, train_dom_loss, cls_correct, dom_correct, total = 0, 0, 0, 0, 0
        for i, (inp, cls_target, dom_target) in enumerate(self.train_loader):
            inp, cls_target, dom_target = inp.cuda().float(), cls_target.cuda(), dom_target.cuda()
            self.optimizer.zero_grad()
            cls_outputs, dom_outputs = self.net(inp)
            cls_loss = self.criterion(cls_outputs, cls_target.reshape(cls_outputs.shape))
            dom_loss = self.criterion(dom_outputs, cls_target.reshape(dom_outputs.shape))
            loss = cls_loss + dom_loss
            loss.backward()
            self.optimizer.step()

            train_cls_loss += cls_loss.item()
            train_dom_loss += dom_loss.item()
            _, cls_predicted = self.m(cls_outputs).max(1)
            _, dom_predicted = self.m(dom_outputs).max(1)
            total += cls_target.size(0)
            cls_correct += cls_predicted.eq(cls_target).sum().item()
            dom_correct += dom_predicted.eq(dom_target).sum().item()

            progress_bar(i, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Domain Acc: %.3f%% (%d/%d)'
                         % (train_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total, 100.*dom_correct/total, dom_correct, total))

        wandb.log({"class loss": train_cls_loss, "dom loss": train_dom_loss, "train cls acc": 100.*cls_correct/total, "train dom acc": 100.*dom_correct/total})

    def test(self, epoch):
        self.net.eval()
        test_cls_loss, test_dom_loss, cls_correct, dom_correct, total = 0, 0, 0, 0, 0
        cls_true, cls_pred, dom_true, dom_pred = np.array([]), np.array([]), np.array([]), np.array([])
        with torch.no_grad():
            for i, (inp, cls_target, dom_target) in enumerate(self.test_loader):
                inp, cls_target, dom_target = inp.cuda().float(), cls_target.cuda(), dom_target.cuda()
                cls_outputs, dom_outputs = self.net(inp)
                cls_loss = self.criterion(cls_outputs, cls_target.reshape(cls_outputs.shape))
                dom_loss = self.criterion(dom_outputs, cls_target.reshape(dom_outputs.shape))
                loss = cls_loss + dom_loss

                test_cls_loss += cls_loss.item()
                test_dom_loss += dom_loss.item()
                _, cls_predicted = self.m(cls_outputs).max(1)
                _, dom_predicted = self.m(dom_outputs).max(1)
                total += cls_target.size(0)
                cls_correct += cls_predicted.eq(cls_target).sum().item()
                dom_correct += dom_predicted.eq(dom_target).sum().item()
                # this is for creating the confusion matrix
                cls_true = np.append(cls_true, cls_target.cpu().numpy())
                cls_pred = np.append(cls_pred, cls_predicted.cpu().numpy())
                dom_true = np.append(dom_true, dom_target.cpu().numpy())
                dom_pred = np.append(dom_pred, dom_predicted.cpu().numpy())

                progress_bar(i, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Domain Acc: %.3f%% (%d/%d)'
                            % (test_cls_loss/(i+1), 100.*cls_correct/total, cls_correct, total, 100.*dom_correct/total, dom_correct, total))
        
        accuracy, balanced_acc, class_accuracy =  evaluate(cls_pred, cls_true)
        dom_accuracy, dom_balanced_acc, dom_class_accuracy =  evaluate(dom_pred, dom_true)
        wandb.log({"val class loss": test_cls_loss, "val dom loss": test_dom_loss, "val cls acc": accuracy, "val balanced_cls_acc": balanced_acc, "val cls acc": class_accuracy, "val dom acc": dom_accuracy, "val balanced dom acc": dom_balanced_acc, "dom class acc": dom_class_accuracy})
        if balanced_acc > self.best_acc:
            self.best_acc, self.best_epoch = balanced_acc, epoch
            self.save_checkpoint(cls_correct/total, epoch)

    def eval(self, inputs):
        """ Farward pass for classification """
        generator = chunks(torch.tensor(inputs).cuda().float(), self.cfg.DATA.BATCH_SIZE)
        preds = np.array([])
        for i, inp in enumerate(generator):
            cls_outputs, dom_outputs = self.net(inp)
            _, cls_predicted = self.m(cls_outputs).max(1)
            preds = np.append(preds, cls_predicted.cpu().numpy())
        return preds

    def save_checkpoint(self, acc, epoch):
        print(f'Saving checkpoint to ./checkpoint/{self.cfg.EXP.get("CHECKPONT_NAME", "mlp")}.pth...')
        state = {
            "acc": acc,
            "epoch": epoch,
            "net": self.net.state_dict()
        }
        torch.save(state, f'./checkpoint/{self.cfg.EXP.get("CHECKPONT_NAME", "mlp")}.pth')
        wandb.save(f'./checkpoint/{self.cfg.EXP.get("CHECKPONT_NAME", "mlp")}.pth')