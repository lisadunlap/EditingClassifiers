import os, io
from tqdm import tqdm
import torch as ch
import torchvision
from torchvision import transforms
import numpy as np
from robustness import datasets
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from tools.custom_folder import ImageFolder
from omegaconf import OmegaConf

from datasets.waterbirds import Waterbirds, WaterbirdsBoring, WaterbirdsSimple
from datasets.planes import Planes, PlanesOrig, PlanesGroundSeg

def tile_image(img):
    tiled = Image.new('RGB', (800,800), "black")
    pixels = img.load()
    pixels_tiled = tiled.load()
    for i in range(800):
        for j in range(800):
            pixels_tiled[i,j] = pixels[i % 256,j % 256]
    return tiled

def get_config(name="Waterbirds"):
    base_cfg  = OmegaConf.load('configs/base.yaml')
    if name == "Waterbirds":
        cfg       = OmegaConf.load('configs/waterbirds.yaml')
    elif name == "WaterbirdsTiny":
        cfg       = OmegaConf.load('configs/waterbirds_tiny.yaml')
    elif name == "Waterbirds95":
        cfg       = OmegaConf.load('configs/waterbirds_95.yaml')
    elif "Planes" in name:
        cfg       = OmegaConf.load('configs/planes.yaml')
    else:
        raise ValueError("Dataset config not found")
    args      = OmegaConf.merge(base_cfg, cfg)
    return args

def get_dataset(dataset_name, dataset_path, 
                batch_size=32, workers=8):
    assert dataset_name in ['ImageNet', 'Places365', 'Waterbirds', 'Waterbirds95', 'WaterbirdsTiny', 'WaterbirdsSimple', 'Planes', 'PlanesBalanced']
    if dataset_name == 'Waterbirds':
        args = get_config('Waterbirds')
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = Waterbirds('/shared/lisabdunlap/vl-attention/data', args, transform=transform)
        valset = Waterbirds('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=transform)
        testset = Waterbirds('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=transform)
        return trainset, ch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), ch.utils.data.DataLoader(valset, batch_size=batch_size), ch.utils.data.DataLoader(testset, batch_size=batch_size)
    if dataset_name == 'Waterbirds95':
        args = get_config('Waterbirds95')
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = Waterbirds('/shared/lisabdunlap/vl-attention/data', args, transform=transform)
        valset = Waterbirds('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=transform)
        testset = Waterbirds('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=transform)
        return trainset, ch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), ch.utils.data.DataLoader(testset, batch_size=batch_size), ch.utils.data.DataLoader(testset, batch_size=batch_size)
    if dataset_name == 'WaterbirdsTiny':
        args = get_config('WaterbirdsTiny')
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = Waterbirds('/shared/lisabdunlap/vl-attention/data', args, transform=transform)
        valset = Waterbirds('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=transform)
        testset = Waterbirds('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=transform)
        return trainset, ch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), ch.utils.data.DataLoader(valset, batch_size=batch_size), ch.utils.data.DataLoader(testset, batch_size=batch_size)
    elif dataset_name == 'WaterbirdsSimple':
        args = get_config('Waterbirds')
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = WaterbirdsSimple('/shared/lisabdunlap/vl-attention/data', args, transform=transform)
        testset = WaterbirdsSimple('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=transform)
        return trainset, ch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), ch.utils.data.DataLoader(testset, batch_size=batch_size), ch.utils.data.DataLoader(testset, batch_size=batch_size)
    elif dataset_name == 'Planes':
        args = get_config('Planes')
        args.DATA.BIAS_TYPE = 'bias_A'
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = Planes('/shared/lisabdunlap/vl-attention/data', args, transform=transform)
        valset = Planes('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=transform)
        testset = Planes('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=transform)
        return trainset, ch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), ch.utils.data.DataLoader(valset, batch_size=batch_size), ch.utils.data.DataLoader(testset, batch_size=batch_size)
    elif dataset_name == 'PlanesBalanced':
        args = get_config('Planes')
        args.DATA.BIAS_TYPE = 'balanced'
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = Planes('/shared/lisabdunlap/vl-attention/data', args, transform=transform)
        valset = Planes('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=transform)
        testset = Planes('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=transform)
        return trainset, ch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), ch.utils.data.DataLoader(valset, batch_size=batch_size), ch.utils.data.DataLoader(testset, batch_size=batch_size)
    elif dataset_name == 'ImageNet':
        dataset = datasets.ImageNet(dataset_path)
    else:
        dataset = datasets.Places365(dataset_path)

    train_loader, val_loader = dataset.make_loaders(batch_size=batch_size, workers=workers)
    return dataset, train_loader, val_loader, None

def tile_image(img):
    tiled = Image.new('RGB', (800,800), "black")
    pixels = img.load()
    pixels_tiled = tiled.load()
    for i in range(800):
        for j in range(800):
            pixels_tiled[i,j] = pixels[i % 256,j % 256]
    return tiled

def get_vehicles_on_snow_data(dataset_name, class_dict, dataset_path='./data/'):
    
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256 if dataset_name == 'Places' else 224),
            transforms.ToTensor(),
        ])
    
    TRAIN_PATH = f'{dataset_path}/snow/train/road:15:05:2021_17:39:17.pt'
    train_data = ch.load(TRAIN_PATH)
    train_imgs, train_masks, train_labels = train_data['imgs'], train_data['masks'], train_data['labels']
    print(train_masks.shape)
    pattern_img_path = f'{dataset_path}/snow/train/snow_texture.jpg'
    pattern_img = transform(Image.open(pattern_img_path))[:3, :, :]
    modified_imgs = train_imgs * (1-train_masks) + pattern_img.unsqueeze(0) * train_masks
    
    train_data = {'imgs': train_imgs,
                 'modified_imgs': modified_imgs,
                 'masks': train_masks,
                 'labels': train_labels
                 }

    TEST_PATH = f'{dataset_path}/snow/test'
    test_data = get_scraped_data(TEST_PATH, dataset_name, class_dict, transform)
    TEST_EXT_PATH = f'{dataset_path}/snow/test_ext'
    test_ext_data = get_scraped_data(TEST_EXT_PATH, dataset_name, class_dict, transform)
    
    return train_data, test_data, test_ext_data

def get_waterbirds_data(dataset_path='./data/', pattern_img_path = './data/waterbirds/forest_broadleaf.jpg'):

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    
    TRAIN_PATH = f'{dataset_path}/waterbirds/data.pt'
    train_data = ch.load(TRAIN_PATH)
    train_imgs, train_masks, train_labels = train_data['imgs'], train_data['masks'], train_data['labels']
    train_masks_new = []
    for mask in train_masks:
        train_masks_new.append(ch.stack([mask[0], mask[0], mask[0]]))
    print(ch.stack(train_masks_new).shape)
    # pattern_img_path = f'{dataset_path}/waterbirds/forest_broadleaf.jpg'
    pattern_img = transform(Image.open(pattern_img_path))[:3, :, :]
    modified_imgs = train_imgs * (1-train_masks) + pattern_img.unsqueeze(0) * train_masks
    
    train_data = {'imgs': train_imgs,
                 'modified_imgs': modified_imgs,
                 'masks': train_masks,
                 'labels': train_labels
                 }

    test = ch.load('./data/waterbirds/data_test_normalized.pt')
    test_data = {0: ch.stack(test[0]), 1: ch.stack(test[1])}
    
    return train_data, test_data

def get_waterbirds_simple_data(dataset_path='./data/', pattern_img_path = './data/waterbirds/forest.jpg', nimgs=10):
    args = get_config("Waterbirds")
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = WaterbirdsBoring('/shared/lisabdunlap/vl-attention/data', args, transform=transform)
    mod_imgs, imgs, masks, labels = [], [], [], []
    for i, d in enumerate(dataset):
        if i >= nimgs:
            break
        pattern_img = transform(Image.open(pattern_img_path))[:3, :, :]
        modified_imgs = d['img'] * (1-d['mask']) + pattern_img * d['mask']
        imgs.append(d['img'])
        masks.append(d['mask'])
        labels.append(d['label'])
        mod_imgs.append(modified_imgs)
    train_data = {
        'imgs': ch.stack(imgs),
        'modified_imgs': ch.stack(mod_imgs),
        'masks': ch.stack(masks),
        'labels': ch.stack(labels)
    }
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    test_data = ch.load('data/waterbirds/simple_test_data.pt')
    # test_dataset = WaterbirdsBoring('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=transform)
    # test_data = {
    #     0:[],
    #     1:[]
    # }
    # for d in test_dataset:
    #     test_data[int(d['label'])].append(d['img'])

    # test_data[0] = ch.stack(test_data[0])
    # test_data[1] = ch.stack(test_data[1])
    # ch.save(test_data, 'data/waterbirds/simple_test_data.pt')

    return train_data, test_data

def get_planes_trian_data(dataset_path='./', pattern_img_path = './data/planes_ground_seg/grass.jpeg', nimgs=5, plane="Airbus", ground="road"):
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    args = get_config("Planes")
    dataset  = PlanesGroundSeg(dataset_path, transform=transform, plane=plane, ground=ground)
    mod_imgs, imgs, masks, labels = [], [], [], []
    for i, d in enumerate(dataset):
        if i >= nimgs:
            break
        pattern_img = transform(Image.open(pattern_img_path))[:3, :, :]
        modified_imgs = d['img'] * (1-d['mask']) + pattern_img * d['mask']
        imgs.append(d['img'])
        masks.append(d['mask'])
        labels.append(d['label'])
        mod_imgs.append(modified_imgs)
    train_data = {
        'imgs': ch.stack(imgs),
        'modified_imgs': ch.stack(mod_imgs),
        'masks': ch.stack(masks),
        'labels': ch.stack(labels)
    }
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    test_dataset = Planes('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=transform)
    test_data = {
        0:[],
        1:[]
    }
    for d in test_dataset:
        test_data[int(d[1])].append(d[0])

    test_data[0] = ch.stack(test_data[0])
    test_data[1] = ch.stack(test_data[1])
    
    return train_data, test_data

def get_typographic_attacks(dataset_path, preprocess, synthetic=True):
    def load_images(base_dir, filter='_clean'):
        PATHS = sorted([os.path.join(base_dir, p) for p in os.listdir(base_dir) if filter in p])
        images = ch.cat([preprocess(PIL.Image.open(p))[None] for p in PATHS])
        images = images.permute(0, 1, 3, 2)
        return ch.flip(images, dims=(3,))
    
    def fig2img(fig):
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)
        return img

    test_data = {}
    for filter in ['clean', 'white', 'ipod']:
        test_data[filter] = load_images(dataset_path, filter=filter)
        
    
    if synthetic:
        plt.text(0.5, .5, 'iPod', fontsize=16, alpha=1)
        plt.axis('off')
        fig = plt.gcf()
        image = fig2img(fig).convert("RGB")
        
        imgs = preprocess(image).unsqueeze(0)
        masks = ch.zeros_like(imgs)
        masks[0, :, 90:120, 110:150] = 1
        modified_imgs = ch.ones_like(imgs)
    else:
        imgs = test_data['ipod'][:1]
        masks = ch.zeros_like(imgs)
        masks[0, :, 160:180, 90:140] = 1
        pattern = ch.ones_like(imgs)
        modified_imgs = imgs * (1-masks) + pattern * masks

    train_data = {'modified_imgs': imgs, 'masks': masks, 'imgs': modified_imgs}
    
    return train_data, test_data  

def get_scraped_data(data_path, dataset_name, class_dict, transform):
    data = {}

    print("Test data stats...")
    for c in os.listdir(data_path):
        l = []
        for f in os.listdir(os.path.join(data_path, c)):
            img_path = os.path.join(data_path, c, f)
            img = Image.open(img_path)
            l.append(transform(img)[:3, :, :])
        valid_classes = [k for k, v in class_dict.items() if c.split('_')[0] in v.replace(' ', '')]
        assert len(valid_classes) == 1
        valid_classes = valid_classes[0]
        data[valid_classes] = ch.stack(l)
        
        print(f'ImageNet class: {class_dict[valid_classes]}; # Images: {len(data[valid_classes])} \n')
    return data

def interpolate(imgs, masks, stylized, batch_size=25):
    interpolated = []
    batch_count = int(np.ceil(len(imgs) / batch_size))
    
    for b in range(batch_count):
        interpolated.append(imgs[b*batch_size:(b+1)*batch_size] * (1 - masks[b*batch_size:(b+1)*batch_size]) + \
                            stylized[b*batch_size:(b+1)*batch_size] * masks[b*batch_size:(b+1)*batch_size])
    return ch.cat(interpolated)

def preprocess_imgs(imgs, preprocess):
    if preprocess is None:
        return imgs
    else:
        return ch.stack([preprocess(torchvision.transforms.ToPILImage()(im)) for im in imgs])

def load_style_images(args, style_name, idx_to_style, preprocess=None):
    style_path = os.path.join(args.style_dir, f'{args.dataset_name}', style_name) 
    
    dataset_style = ImageFolder(style_path, 
                                transform=torchvision.transforms.ToTensor(),
                                img_mapping=idx_to_style)

    loader_style = ch.utils.data.DataLoader(dataset_style, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=False)
        
    imgs_style, labels_style = [], []

    for _, style in tqdm(enumerate(loader_style), total=len(loader_style)):
        imgs_style.append(style[0])
        labels_style.append(style[1])

    return preprocess_imgs(ch.cat(imgs_style), preprocess=preprocess), ch.cat(labels_style)

def obtain_train_test_splits(args, concept, class_dict, style_name, preprocess=None, rng=None):
    
    all_imgs, all_labels = concept['imgs'], concept['labels']
    
    label_counter = Counter(all_labels.numpy())
    allowed_labels = [k for k, v in label_counter.items() if v >= args.ntrain]
    target_label = rng.choice(allowed_labels, 1)[0]
    
    print("Target label: ", target_label)
    print("Examples of relevant classes:")
    print([(k, class_dict[k], v) for ii, (k, v) in enumerate(label_counter.items()) if ii < 5])
    
    rel_idx = np.where(all_labels.numpy() == target_label)[0]
        
    idx_train = rel_idx[rng.choice(len(rel_idx), args.nconcept, replace=False)]
    idx_test = np.array(list(set(np.arange(len(all_labels))) - set(idx_train)))
    
    style_path = os.path.join(args.style_dir, f'{args.dataset_name}', style_name)
    Nstyles = len(os.listdir(style_path))
    train_style = rng.choice(Nstyles, 1)[0]
    assert train_style < Nstyles
    style_number_test = rng.choice(list(set(np.arange(Nstyles)) - set([train_style])), len(idx_test))
    assert train_style not in style_number_test
    
    idx_all = concept['idx'][np.concatenate([idx_train, idx_test])]
    style_number_same = np.concatenate([[train_style] * len(idx_train), [train_style] * len(idx_test)])
    style_number_diff = np.concatenate([[train_style] * len(idx_train), style_number_test])
    
    idx_to_style = {k: v for k, v in zip(idx_all, style_number_same)}
    stylized_img_same, style_labels_same = load_style_images(args, 
                                                             style_name, 
                                                             idx_to_style,
                                                             preprocess=preprocess)
    assert np.array_equal(style_labels_same.numpy(), np.array(list(idx_to_style.keys())))
    
    idx_to_style = {k: v for k, v in zip(idx_all, style_number_diff)}
    stylized_img_diff, style_labels_diff = load_style_images(args, 
                                                             style_name, 
                                                             idx_to_style,
                                                             preprocess=preprocess)
   
    
    assert np.array_equal(style_labels_diff.numpy(), np.array(list(idx_to_style.keys())))
    
    data_dict, data_info_dict = {}, {}

    # Training data
    data_dict['train_data'] = {k: v[idx_train] for k, v in concept.items()}
    
    check_labels = np.unique(data_dict['train_data']['labels'].numpy())
    assert len(check_labels) == 1
    assert check_labels[0] == target_label

        
    data_dict['train_data']['modified_imgs'] = interpolate(data_dict['train_data']['imgs'], 
                                                        data_dict['train_data']['masks'], 
                                                        stylized_img_same[:len(idx_train)])
    
    # Test data
    data_dict['test_data'] = {k: concept[k][idx_test] for k in data_dict['train_data']
                              if k != 'modified_imgs'}
    data_dict['test_data']['modified_imgs_same'] = interpolate(data_dict['test_data']['imgs'], 
                                                        data_dict['test_data']['masks'], 
                                                        stylized_img_same[len(idx_train):])
    data_dict['test_data']['modified_imgs_diff'] = interpolate(data_dict['test_data']['imgs'], 
                                                        data_dict['test_data']['masks'], 
                                                        stylized_img_diff[len(idx_train):])

    
    
    data_info_dict = {'target_label': target_label, 
                      'idx_train': data_dict['train_data']['idx'], 
                      'idx_test': data_dict['test_data']['idx'], 
                      'labels_train': data_dict['train_data']['labels'], 
                      'labels_test': data_dict['test_data']['labels'], 
                     }
    return data_dict, data_info_dict