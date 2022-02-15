from fnmatch import translate
import os
from shutil import SpecialFileError
import torch
from PIL import Image
import numpy as np
import pandas as pd
import random

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms

# from utils import attention_utils as au

GROUP_NAMES_AIR_GROUND = np.array(['Airbus_ground', 'Airbus_air', 'Boeing_ground', 'Boeing_air'])
GROUP_NAMES_GRASS_ROAD = np.array(['Airbus_grass', 'Airbus_road', 'Boeing_grass', 'Boeing_road'])

def get_label_mapping():
    return np.array(['Airbus', 'Boeing'])


class PlanesOrig(torch.utils.data.Dataset):
    def __init__(self, root, cfg, split='train', transform=None):
        dataset_type = cfg.DATA.BIAS_TYPE
        assert dataset_type in ['bias_A', 'bias_B', 'balanced', 'bias_A_05', 'bias_B_05', 'bias_A_10', 'bias_B_10'], '{} is not an available Planes dataset type'.format(dataset_type)
        assert split in ['train', 'val', 'test']
        self.cfg = cfg
        self.transform = transform
        if cfg.DATA.CROP:
            new_trans = [transforms.Lambda(self.crop_img)]
            if transform:
                for t in transform.transforms:
                    new_trans.append(t)
            self.transform = transforms.Compose(new_trans)
        self.split = split
        self.size  = cfg.DATA.SIZE
        self.orig_root = root.replace('/data', '')

        if cfg.DATA.BIAS_MODE == 'air_ground':
            self.root = os.path.join(root, 'planes/manufacturer/biased')
            group_names = GROUP_NAMES_AIR_GROUND
            print("GET HERE --------------")
        elif cfg.DATA.BIAS_MODE == 'grass_road':
            self.root = os.path.join(root, 'planes/manufacturer/biased_ground_type')
            group_names = GROUP_NAMES_GRASS_ROAD
        else:
            raise Exception('ERROR: bias mode {} not supported'.format(cfg.DATA.BIAS_MODE))
        self.return_attention = cfg.DATA.ATTENTION_DIR != "NONE"
        print("RETURN ATTENTION ", self.return_attention)
        self.mask_pixels = cfg.DATA.MASK_PIXELS
        if self.mask_pixels:
            assert self.return_attention, 'DATA.ATTENTION_DIR must not be NONE if masking pixels'
        self.add_rotations = cfg.DATA.ADD_ROTATIONS and split == 'train'
        if self.add_rotations:
            self.rotation_probability = float(cfg.DATA.ROTATION_PROBABILITY)
            self.max_rotation_angle = int(cfg.DATA.MAX_ROTATION_ANGLE)

        df_all = pd.read_csv(os.path.join(self.root, 'all_images.csv'))

        # All filenames, regardless of split & bias type. Useful for extracting attention on all images at once.
        self.data = np.array(df_all['Filename'])


        if split == 'train':
            if dataset_type == 'bias_B_05' or dataset_type == 'bias_A_05' or dataset_type == 'bias_B_10' or dataset_type == 'bias_A_10':
                if cfg.DATA.BIAS_MODE == 'grass_road':
                    df = pd.read_csv(os.path.join(root, 'train_{}_ground.csv'.format(dataset_type)))
                else:
                    df = pd.read_csv(os.path.join(root, 'train_{}.csv'.format(dataset_type)))
                    print(os.path.join(root, 'train_{}.csv'.format(dataset_type)))
            else:
                df = pd.read_csv(os.path.join(self.root, 'train_{}.csv'.format(dataset_type)))
        elif cfg.DATA.BIAS_MODE == 'grass_road':
            if split == 'val':
                df = pd.read_csv(os.path.join(self.root, 'val.csv'))
            else:
                df = pd.read_csv(os.path.join(self.root, 'test.csv'))
        else:
            df = pd.read_csv(os.path.join(self.root, 'all_images.csv'))
            df = df[df['Split'] == split]

        filenames = np.array(df['Filename'])
        labels = np.array(df['Label'])
        groups = np.array(df['Group'])
        self.image_filenames = filenames

        self.filenames = filenames
        self.labels    = torch.Tensor(labels)
        self.groups    = torch.Tensor(groups)
        self.classes = ['airbus', 'boeing']

        if self.return_attention:
            attention_root = os.path.join(
                root,
                'planes',
                cfg.DATA.ATTENTION_DIR
            )
            print("=====================")
            print("=====================")
            print(attention_root)
            print("=====================")
            print("=====================")

            self.attention_data = np.array(
                [os.path.join(attention_root, f.replace('.jpg', '.pth')) for f in self.filenames]
            )
        else:
            self.attention_data = None

        segmentation_root = os.path.join(
            root,
            'planes',
            'deeplabv3_attention'
        )
        self.segmentation_data = np.array(
            [os.path.join(segmentation_root, f.replace('.jpg', '.pth').replace('biased_ground_type', 'biased').replace('grass', 'ground').replace('road', 'ground')) for f in self.filenames]
        )
        self.seg_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.size,self.size)),
            transforms.ToTensor(),
        ])

        print('PLANES {}'.format(split.upper()))
        print('LEN DATASET: {}'.format(len(self.filenames)))
        print('# AIRBUS:    {}'.format(len(torch.where(self.labels == 0)[0])))
        print('# BOEING:    {}'.format(len(torch.where(self.labels == 1)[0])))


        for idx, group in enumerate(group_names):
            print('# {}: {}'.format(group, len(torch.where(self.groups == idx)[0])))

    def get_label(self, filename):
        return self.labels[np.where(self.filenames == filename)[0]]

    def __getitem__(self, index):

        path = self.filenames[index]
        label = self.labels[index]
        group = self.groups[index]
        seg_path = self.segmentation_data[index]
        
        img = Image.open(os.path.join(self.orig_root, path)).convert('RGB')
        # width, height = img.size

        # # crop bottom
        # left, top = 0, 0
        # right = width
        # bottom = height - 20
        # img = img.crop((left, top, right, bottom))
        
        seg = torch.load(seg_path)['mask'][0]
        # seg = torch.Tensor([-1])

        if self.transform is not None:
            img = self.transform(img)
            # seg = self.seg_transform(seg)

        if self.return_attention or self.mask_pixels:
            att = torch.load(self.attention_data[index].replace('biased_ground_trype', 'biased'))

            if 'deeplabv3_attention' in self.cfg.DATA.ATTENTION_DIR:
                att = att['mask']
            else:
                att = att['unnormalized_attentions']
            att = F.interpolate(att, size=(self.size, self.size), mode='bilinear',
                                align_corners=False)#[0]
            if self.mask_pixels:
                if self.cfg.DATA.MASK_PIXELS_SEG_THEN_ATTENTION:
                    # mask_att = self.mask_seg_then_attention(att, seg)
                    print("dont get here")
                else:
                    mask_att = self.get_attention_for_mask(att)
                if self.cfg.DATA.MASK_PIXELS_THRESHOLD != "NONE":
                    threshold = self.cfg.DATA.MASK_PIXELS_THRESHOLD
                    mask_att[mask_att >= threshold] = 1.0
                img = img * mask_att
            # print(att)
            if self.cfg.DATA.REMOVE_BACKGROUND and self.split == 'train':
                img = img * att[0][0]

        else:
            att = torch.Tensor([-1]) # So batches correctly

        if self.add_rotations:
            if random.random() < self.rotation_probability:
                angle = random.randint(
                    -1 * self.max_rotation_angle,
                    self.max_rotation_angle
                )
                img = TF.rotate(img, angle)
                seg = TF.rotate(seg, angle)
                if self.return_attention or self.mask_pixels:
                    att = TF.rotate(att, angle)

                # torch.save({
                #     'img': img,
                #     'seg': seg,
                #     'att': att
                # }, 'tmp_rots_{}.pth'.format(index))
                # print('SAVED {}'.format(index))

        return {
            'image_path': path,
            'image': img,
            'label': label,
            'seg':   seg,
            'group': group,
            #'bbox': NULL,
            'attention': att
        }

    @staticmethod
    def crop_img(img):
        width, height = img.size

        # crop bottom
        left, top = 0, 0
        right = width
        bottom = height - 20
        # img = img.crop((left, top, right, bottom))
        return img.crop((left, top, right, bottom))

    # def get_attention_for_mask(self, att):
    #     att, valid_inds = au.parse_attention(
    #         att.unsqueeze(0),
    #         shape=(self.size, self.size),
    #         mode=self.cfg.DATA.MASK_PIXELS_COMBINE_ATT_MODE
    #     )
    #     # For non-valid attention, replace w/ tensor of 1's -
    #     # image will stay the same
    #     out = torch.ones(
    #         1,
    #         1,
    #         self.size,
    #         self.size
    #     )
    #     if valid_inds.sum() > 0:
    #         out[valid_inds.bool()] = att
    #     return att[0]

    # def mask_seg_then_attention(self, att, seg):
    #     mask_att = self.get_attention_for_mask(att)
    #     combined_att = seg * mask_att
    #     return combined_att


    def __len__(self):
        return len(self.filenames)



class Planes(PlanesOrig):
    
    def __getitem__(self, idx):
        results = super().__getitem__(idx)
        return results['image'], results['label']


class PlanesGroundSeg(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, plane="Airbus", ground="road"):
        self.df = pd.read_csv(os.path.join(root, 'data/planes_ground_seg/images.csv'))
        self.df = self.df[self.df['Plane'] == plane]
        self.df = self.df[self.df['Ground'] == ground].reset_index()
        self.filenames = [os.path.join(root, f) for f in self.df['Filename']]
        self.labels = self.df['Label']
        self.groups = self.df['Group']
        self.seg = self.df['Seg']
        self.transform = transform
        self.size = 224
        self.seg_transform = transforms.Compose([
            transforms.Resize((self.size,self.size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filenames[idx]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        label = torch.Tensor(int(self.labels[idx]))
        group = torch.Tensor(int(self.groups[idx]))
        seg = self.seg_transform(Image.fromarray(np.uint8(np.load(self.seg[idx]))))
        return {
            'image_path': path,
            'img': img,
            'label': label,
            'mask':   seg,
            'group': group,
        }

