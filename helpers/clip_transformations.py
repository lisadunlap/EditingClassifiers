import os
import clip
import torch

import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import pandas as pd

from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

class Noop:
    """
    Does nothing.
    """
    def __init__(self, text_prompts, model):
        self.text_prompts = text_prompts
        self.text_embeddings = []

    def apply(self, inputs):
        return inputs

class BiasCorrection(Noop):
    """
    Simply subtracts the embeddings of the text prompts
    """
    def __init__(self, text_prompts, model):
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

    # def init(self, text_prompts, model, metric):
    #     super().__init__(text_prompts=text_prompts, model=model)
    #     self.metric = 'diff'

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

    def __init__(self, prompts, model, k=2):
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
        pca = PCA(n_components=k, svd_solver='full')
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