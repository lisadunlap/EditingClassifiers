import sys, os, dill, torch, numpy, itertools
sys.path.append('./CLIP')
import torch as ch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from PIL.Image import BICUBIC, new
from collections import OrderedDict
from robustness.model_utils import make_and_restore_model
from robustness.tools.label_maps import CLASS_DICT
from clip.clip import _download, _MODELS, _convert_image_to_rgb
import helpers.context_helpers as coh
from models.custom_vgg import vgg16_bn, vgg16
from models.custom_resnet import resnet18, resnet50
from models.custom_clip import build_model
from tools.places365_names import class_dict
from models.vit import VisionTransformer, CONFIGS

IMAGENET_PATH = '/path/to/imagenet'

def get_default_paths(dataset_name, arch='vgg16', binary=False):
    num_classes = 1 if binary else 2
    if dataset_name == 'ImageNet':
        data_path = '/shared/group/ilsvrc'
        label_map = CLASS_DICT['ImageNet']
        
        if arch.startswith('clip'):
            model_path = None
            model_class = None
            arch = arch
        elif arch == 'resnet50':
            model_path = './checkpoints/imagenet_resnet50.ckpt'
            model_class, arch = resnet50(), 'resnet50'
        elif arch == 'vgg16':
            model_path = './checkpoints/imagenet_vgg.pt.best'
            model_class, arch = vgg16_bn(), 'vgg16_bn'
        elif arch == 'ViT':
            model_path = "./attention_data/ViT-B_16-224.npz"
            config = CONFIGS["ViT-B_16"]
            model_class, arch = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True), 'ViT'
            model_class.load_from(np.load("./attention_data/ViT-B_16-224.npz"))
    elif dataset_name == 'Waterbirds':
        data_path = '/shared/lisabdunlap/vl-attention/data/waterbird_1.0_forest2water2'
        label_map = {0: 'landbird', 1: 'waterbird'}
        if arch == 'resnet50':
            model_path = './checkpoints/waterbirds.pth'
            model_class, arch = resnet50(num_classes=1000), 'resnet50'
            model_class = load_checkpoint(model_class, "ImageNet", './checkpoints/imagenet_resnet50.ckpt')
            model_class.layer17.fc = nn.Linear(2048, num_classes)
        elif arch == 'ViT':
            model_path = "./checkpoints/WaterbirdsSimple-vit.pth"
            config = CONFIGS["ViT-B_16"]
            model_class, arch = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True), 'ViT'
            model_class.load_from(np.load("./attention_data/ViT-B_16-224.npz"))
            model_class.head = nn.Linear(768, num_classes)
    elif dataset_name == 'WaterbirdsSimple':
        data_path = '/shared/lisabdunlap/vl-attention/data/waterbird_1.0_forest2water2'
        label_map = {0: 'landbird', 1: 'waterbird'}
        if arch == 'resnet50':
            model_path = './checkpoints/WaterbirdsSimple.pth'
            model_class, arch = resnet50(num_classes=1000), 'resnet50'
            model_class = load_checkpoint(model_class, "ImageNet", './checkpoints/imagenet_resnet50.ckpt')
            model_class.layer17.fc = nn.Linear(2048, num_classes)
        elif arch == 'ViT':
            model_path = "./checkpoints/WaterbirdsSimple-vit.pth"
            config = CONFIGS["ViT-B_16"]
            model_class, arch = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True), 'ViT'
            model_class.load_from(np.load("attention_data/ViT-B_16-224.npz"))
            model_class.head = nn.Linear(768, num_classes)
    elif dataset_name == 'Planes':
        data_path = '/shared/lisabdunlap/vl-attention/data/planes'
        label_map = {0: 'Airbus', 1: 'Boeing'}
        if arch == 'resnet50':
            model_path = './checkpoints/planes-binary.pth'
            model_class, arch = resnet50(num_classes=1000), 'resnet50'
            model_class = load_checkpoint(model_class, "ImageNet", './checkpoints/imagenet_resnet50.ckpt')
            model_class.layer17.fc = nn.Linear(2048, num_classes)
    elif dataset_name == 'PlanesBalanced':
        data_path = '/shared/lisabdunlap/vl-attention/data/planes'
        label_map = {0: 'Airbus', 1: 'Boeing'}
        if arch == 'resnet50':
            model_path = './checkpoint/planes-balanced.pth'
            model_class, arch = resnet50(num_classes=1000), 'resnet50'
            model_class = load_checkpoint(model_class, "ImageNet", './checkpoints/imagenet_resnet50.ckpt')
            model_class.layer17.fc = nn.Linear(2048, num_classes)
    else:
        NotImplementedError("Dataset not implemented")
 
    return data_path, model_path, model_class, arch, label_map

def eval_accuracy(model, loader, alt=False, normalize=None):
    labels, preds = [], []
    with ch.no_grad():
        for _, (im, targ) in enumerate(loader):
            if normalize:
                im = normalize(im.cuda())
            if alt:
                op, _ = model(im.cuda())
            else:
                op = model(im.cuda()) #model(normalizer(im))
            preds.append(op.argmax(dim=1).cpu())
            labels.append(targ)
    return ch.cat(preds), ch.cat(labels)

def load_clip(arch='clip_RN50'):
    
    def _transform_unnorm(n_px):
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor()
        ])
    
    arch = arch.split('_')[1]
    
    model_path = _download(_MODELS[arch], './cache')
    model = torch.jit.load(model_path, map_location="cpu").eval()    
    model = build_model(model.state_dict()).cuda()
    
    model.cache_dir = f'./models/clip/{arch}'
    if not os.path.exists(model.cache_dir): os.makedirs(model.cache_dir)
    model.zeroshot_classifier()
    return model, _transform_unnorm(model.visual.input_resolution)

def load_checkpoint(model, dataset, resume_path, 
                    new_code=True, arch='resnet50'):
    print(f"....loading checkpoint from {resume_path}")
    
    if arch.startswith('clip'):
        return load_clip(arch)

    
    checkpoint = ch.load(resume_path, pickle_module=dill)
    if isinstance(model, str):
        model = dataset.get_model(model, False).cuda()
        model.eval()
        pass
    
    # if 'Waterbirds' in dataset or 'Planes' in dataset:
    if 'net' in checkpoint.keys():
        state_dict= checkpoint['net']
        new_dict = {}
        for k in state_dict:
            if 'module.' in k[:7]:
                new_dict[k[7:]] = state_dict[k]
            else:
                new_dict[k] = state_dict[k]
        state_dict = new_dict
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    if 'model_state_dict' in checkpoint:
        state_dict= checkpoint['model_state_dict']
        new_dict = {}
        for k in state_dict:
            if 'module' in k:
                components = k.replace('module.', '').split('.')
                new_thing = components[:-1] + ['module'] + [components[-1]]
                new_dict['.'.join(new_thing)] = state_dict[k]
            else:
                new_dict[k] = state_dict[k]
        state_dict = new_dict
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict= checkpoint['state_dict']
    else:
        state_dict = checkpoint
    sd = {}
    
    for i, (k, v) in enumerate(state_dict.items()):
        if arch == 'resnet50':
            if k[len('module.'):].startswith('model'):
                kk = k[len('module.model.'):]
                if arch == 'resnet50':
                    if kk.startswith('conv') or kk.startswith('bn'):
                        kk = 'layer0.' + kk
                    elif kk.startswith('fc'):
                        kk = 'layer5.' + kk

                    if kk.startswith('layer0'):
                        rep_key = 'layer0'
                        kk = kk.replace(rep_key, model.sequence_dict[rep_key])
                    elif kk.startswith('layer5'):
                        rep_key = 'layer5'
                        kk = kk.replace(rep_key, model.sequence_dict[rep_key])
                    else:
                        rep_key = kk[:len('layer1.0')]
                        kk = kk.replace(rep_key, model.sequence_dict[rep_key])
                        if new_code:
                            if 'downsample' in kk:
                                kks = kk.split('.')
                                kks.insert(1, "residual")
                            else:
                                kks = kk.split('.')
                                kks.insert(-1, "module")
                            kk = '.'.join(kks) 
                        for rr in ['.conv3.',  '.bn3.', '.relu.', '.residual.']:
                            if rr in kk:
                                kk = kk.replace(rr, '.final'+rr)
                sd[kk] = v
            else:
                continue
        elif arch == 'resnet18':
            kk = k
            if kk.startswith('conv') or kk.startswith('bn'):
                kk = 'layer0.' + kk
            elif kk.startswith('fc'):
                kk = 'layer5.' + kk

            if kk.startswith('layer0'):
                rep_key = 'layer0'
                kk = kk.replace(rep_key, model.sequence_dict[rep_key])
            elif kk.startswith('layer5'):
                rep_key = 'layer5'
                kk = kk.replace(rep_key, model.sequence_dict[rep_key])
            else:
                rep_key = kk[:len('layer1.0')]
                kk = kk.replace(rep_key, model.sequence_dict[rep_key])
                if new_code:
                    if 'downsample' in kk:
                        kks = kk.split('.')
                        kks.insert(1, "residual")
                    else:
                        kks = kk.split('.')
                        kks.insert(-1, "module")
                    kk = '.'.join(kks) 
                for rr in ['.conv2.',  '.bn2.', '.relu.', '.residual.']:
                    if rr in kk:
                        kk = kk.replace(rr, '.final'+rr)
            sd[kk] = v
        elif arch == 'vgg16_bn':
            if k[len('module.model.'):].startswith('model'):
                kk = k[len('module.model.model.'):]
                kks = kk.split('.')
                p, s = '.'.join(kks[:-1]), kks[-1]
                if p in model.sequence_dict:
                    kk = f"{model.sequence_dict[p]}.{s}"
                else:
                    kk = f"{p}.{s}"
                sd[kk] = v
            else:
                continue
        elif arch == 'vgg16' and k.startswith('model'):
            kk = k[len('model.'):]
            kks = kk.split('.')
            p, s = '.'.join(kks[:-1]), kks[-1]
            if p in model.sequence_dict:
                kk = f"{model.sequence_dict[p]}.{s}"
            else:
                kk = f"{p}.{s}"
            sd[kk] = v
        elif arch == 'vgg16' and k.startswith('features'):
            suffix = k.split(".")[-1]
            kk = f'layer{i // 2}.conv.{suffix}'
            sd[kk] = v
        elif arch == 'vgg16' and k.startswith('classifier'):
            layer= {'fc6': 0,
                    'fc7': 3,
                    'fc8a': 6,
                   }[k.split('.')[1]]
            suffix = k.split(".")[-1]
            kk = f'classifier.{layer}.{suffix}'
            sd[kk] = v
        else: 
            raise ValueError('unrecognizable checkpoint')
            
    model.load_state_dict(sd)
    model.eval()
    pass

    return model

def load_classifier(model_path, model_class, arch, dataset, layernum):
    
    preprocess = None
    mod = load_checkpoint(model=model_class, 
                            dataset=dataset,
                            resume_path=model_path,
                            arch=arch)
    if arch.startswith('clip'):
        mod, preprocess = mod
    elif arch == 'ViT':
        return mod.cuda()
    mod = mod.cuda()
    
    con_mod, Nfeatures = coh.get_context_model(mod, layernum, arch)
    if arch.startswith('vgg'):
        targ_mod = mod[layernum + 1]
    elif arch.startswith('clip'):
        targ_mod = mod.visual[layernum + 1].final
    else:
        targ_mod = mod[layernum + 1].final
        
    if not arch.startswith('clip'):
        return mod, con_mod, targ_mod
    else:
        return mod, con_mod, targ_mod, preprocess