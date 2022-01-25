import torch
from torchvision import transforms
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from PIL import Image

GROUP_NAMES = np.array(['Land_on_Land', 'Land_on_Water', 'Water_on_Land', 'Water_on_Water'])

def get_label_mapping():
    return np.array(['Landbird', 'Waterbird'])

class WaterbirdsOrig(torch.utils.data.Dataset):
    def __init__(self, root, cfg, split='train', transform=None, metadata='/shared/lisabdunlap/vl-attention/data/waterbird_1.0_forest2water2/metadata.csv'):
        self.cfg = cfg
        # self.original_root       = os.path.expanduser(root)
        self.original_root = root
        self.transform  = transform
        self.split      = split
        self.root       = os.path.join(self.original_root, 'waterbird_1.0_forest2water2')
        self.return_seg = True
        self.return_bbox = False
        self.size       = cfg.DATA.SIZE
        self.remove_background = cfg.DATA.REMOVE_BACKGROUND
        self.draw_square = cfg.DATA.DRAW_SQUARE
        self.crop_bird   = cfg.DATA.CROP_BIRD
        self.return_attention = cfg.DATA.ATTENTION_DIR != "NONE"

        print('WATERBIRDS DIR: {}'.format(self.root))

        self.seg_transform = transforms.Compose([
            transforms.Resize((self.size,self.size)),
            transforms.ToTensor(),
        ])

        # metadata
        # self.metadata_df = pd.read_csv(
        #     os.path.join(self.root, 'metadata.csv'))
        self.metadata_df = pd.read_csv(metadata)

        # Get the y values
        self.labels = self.metadata_df['y'].values
        self.num_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.labels*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.seg_data  =  np.array([os.path.join(root, 'CUB_200_2011/segmentations',
                                                 path.replace('.jpg', '.png')) for path in self.filename_array])

        self.data = np.array([os.path.join(self.root, filename) for filename in self.filename_array])

        if self.return_attention:
            self.attention_data = np.array([os.path.join(self.root, cfg.DATA.ATTENTION_DIR,
                                                path.replace('.jpg', '.pth')) for path in self.filename_array])

        mask = self.split_array == self.split_dict[self.split]
        num_split = np.sum(mask)
        self.indices = np.where(mask)[0]

        self.labels = torch.Tensor(self.labels)
        self.group_array = torch.Tensor(self.group_array)

        # Arrays holding image filenames and labels for just the split, not all data.
        # Useful for detection approach to quickly access labels & filenames
        self.image_filenames     = []
        self.labels_split        = []
        self.group_labels_split  = []
        for idx in self.indices:
            self.image_filenames.append(self.data[idx])
            self.labels_split.append(self.labels[idx])
            self.group_labels_split.append(self.group_array[idx])
        self.image_filenames    = np.array(self.image_filenames)
        self.labels_split       = torch.Tensor(self.labels_split)
        self.group_labels_split = torch.Tensor(self.group_labels_split)

        if self.draw_square:
            draw_square_transforms_0 = []
            draw_square_transforms_1 = []
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    draw_square_transforms_0.append(t)
                    draw_square_transforms_1.append(t)
                    draw_square_transforms_0.append(DrawRect())
                    draw_square_transforms_1.append(DrawRect(color=(0,0,255), start_pos=(180,180)))
                else:
                    draw_square_transforms_0.append(t)
                    draw_square_transforms_1.append(t)
            self.draw_square_transforms_0 = transforms.Compose(draw_square_transforms_0)
            self.draw_square_transforms_1 = transforms.Compose(draw_square_transforms_1)

        if self.return_bbox:
            bboxes = pd.read_csv(os.path.join(self.original_root, 'CUB_200_2011', 'bounding_boxes.txt'), header=None)
            self.bbox_coords = np.zeros((self.data.shape[0], 4))
            for i, row in enumerate(bboxes.values):
                coords = row[0].split(' ')[1:]
                coords = np.array(coords).astype(float)
                self.bbox_coords[i] = coords


        print('NUMBER OF SAMPLES WITH LABEL {}: {}'.format(get_label_mapping()[0],
                                                           len(torch.where(self.labels[self.indices] == 0)[0]))
        )
        print('NUMBER OF SAMPLES WITH LABEL {}: {}'.format(get_label_mapping()[1],
                                                           len(torch.where(self.labels[self.indices] == 1)[0]))
        )

        for i in range(len(GROUP_NAMES)):
            print('NUMBER OF SAMPLES WITH GROUP {}: {}'.format(GROUP_NAMES[i],
                                                               len(torch.where(self.group_array[self.indices] == i)[0]))
            )

    def create_subset(self):
        subset_size = self.cfg.DATA.SUBSET_SIZE
        images_per_class = subset_size // 2
        inds = {
            'class_0': torch.where(self.labels[self.indices] == 0)[0],
            'class_1': torch.where(self.labels[self.indices] == 1)[0]
        }

    def get_filenames(self, indices):
        """
        Return list of filenames for requested indices.
        Need to access self.indices to map the requested indices to the right ones in self.data
        """
        filenames = []
        for i in indices:
            new_index = self.indices[i]
            filenames.append(self.data[new_index])
        return filenames

    def get_label(self, filename):
        return self.labels[np.where(self.data == filename)[0]]

    def __getitem__(self, index):

        original_index = index
        index = self.indices[index]

        path     = self.data[index]
        label    = self.labels[index]
        seg_path = self.seg_data[index]
        group    = self.group_array[index]
        place = self.confounder_array[index]

        group = torch.Tensor([group])

        img = Image.open(path).convert('RGB')
        if self.return_seg:
            seg = Image.open(seg_path)
        if self.return_bbox:
            bbox = self.bbox_coords[index]
            bbox = np.round(bbox).astype(int)

            arr = np.array(img)
            bbox_im = np.zeros(arr.shape[:2])
            bbox_im[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0]+bbox[2])] = 1
            bbox_im = torch.Tensor(bbox_im).unsqueeze(0).unsqueeze(0)
            bbox_im = F.interpolate(bbox_im, size=(self.size, self.size), mode='bilinear',
                                    align_corners=False)[0]
        else:
            bbox_im  = torch.Tensor([-1]) # So batches correctly

        if self.return_attention:
            att = torch.load(self.attention_data[index])
            if 'deeplab' in self.cfg.DATA.ATTENTION_DIR:
                att = att['mask']
            else:
                att = att['unnormalized_attentions']
            att = F.interpolate(att, size=(self.size, self.size), mode='bilinear',
                                align_corners=False)#[0]

        else:
            att  = torch.Tensor([-1]) # So batches correctly

        if self.crop_bird:
            bbox = self.bbox_coords[index]
            bbox = np.round(bbox).astype(int)
            img = np.array(img)
            img = img[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0]+bbox[2])]
            img = Image.fromarray(img)
            if self.return_seg:
                seg = np.array(seg)
                seg = seg[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0]+bbox[2])]
                seg = Image.fromarray(seg)
            # print('saving crop')
            # idx = np.random.rand()
            # img.save('crop_im_{}.jpg'.format(idx))
            # seg.save('crop_seg_{}.jpg'.format(idx))

        if self.transform is not None and self.draw_square:
            if label == 0:
                img = self.draw_square_transforms_0(img)
            else:
                img = self.draw_square_transforms_1(img)
        elif self.transform is not None:
            img = self.transform(img)

        if self.return_seg:
            seg = self.seg_transform(seg)
            if self.remove_background:
                img = img * seg
        else:
            seg = torch.Tensor([-1]) # So batches correctly

        # from datasets import normalizations
        # from utils import general_utils as gu
        # mean, std = normalizations.normalizations['imagenet']['mean'], \
        #         normalizations.normalizations['imagenet']['std']
        # unnorm = gu.create_unnormalize_transform(mean, std)
        # unnormed = gu.convert_to_numpy(img.unsqueeze(0), unnorm)[0]
        # unnormed = Image.fromarray(unnormed)
        # unnormed.save('{}.jpg'.format(index))
        # print('saved')

        return {
            'image_path': path,
            'image': img,
            'label': label,
            'seg':   seg,
            'group': group,
            'bbox': bbox_im,
            'attention': att,
            'index': original_index,
            'split': self.split,
            'place': place,
        }
        # return img, label


    def __len__(self):
        return len(self.indices)

class Waterbirds(WaterbirdsOrig):
    
    def __getitem__(self, idx):
        results = super().__getitem__(idx)
        return results['image'], results['label']

class WaterbirdsBoring(WaterbirdsOrig):

    def __init__(self, root, cfg, split='train', transform=None, 
                    metadata='/shared/lisabdunlap/vl-attention/data/waterbird_1.0_forest2water2/metadata.csv', 
                    land_background='/home/lisabdunlap/EditingClassifiers/data/waterbirds/forest.jpg', 
                    water_background='/home/lisabdunlap/EditingClassifiers/data/waterbirds/ocean_2.jpg'):
        super().__init__(root, cfg, split, transform, metadata)
        self.land_background = land_background
        self.water_background = water_background
    
    def __getitem__(self, idx):
        results = super().__getitem__(idx)
        new_results = results.copy()
        mask_copy = np.ones(results['seg'].shape)
        train_masks = np.logical_xor(mask_copy, results['seg'])
        pattern_img_path = self.land_background if results['place'] == 0 else self.water_background
        if self.transform:
            pattern_img = self.transform(Image.open(pattern_img_path))[:3, :, :]
            modified_img = results['image'] * (1-train_masks) + pattern_img * train_masks
        else:
            pattern_img = Image.open(pattern_img_path)
            modified_img = results['image'] * (1-train_masks) + pattern_img * train_masks
        
        return {
            'img': modified_img, 
            'mask': train_masks, 
            'label': results['label']
            }

class WaterbirdsSimple(WaterbirdsBoring):

    def __getitem__(self, idx):
        results = super().__getitem__(idx)
        return results['img'], results['label']


class DrawRect(object):
    def __init__(self, start_pos=(20,20), width=32, color=(255,0,0)):
        self.start_pos = start_pos
        self.end_pos   = (start_pos[0] + width, start_pos[1] + width)
        self.color     = color
    def __call__(self, img):
        """ Draw rectangle on PIL Image, return PIL Image"""
        img = np.array(img)
        img[self.start_pos[0]:self.end_pos[0], self.start_pos[0]:self.end_pos[1], :] = self.color
        #img = cv2.rectangle(np.array(img), self.start_pos, self.end_pos, self.color, -1)
        img = Image.fromarray(img)
        return img


def get_loss_upweights(bias_fraction=0.95, mode='per_class'):
    """
    For weighting training loss for imbalanced classes.

    Returns 1D tensor of length 2, with loss rescaling weights.

    weight w_c for class c in C is calculated as:
    (1 / num_samples_in_class_c) / (max(1/num_samples_in_c) for c in num_classes)

    """
    assert mode in ['per_class', 'per_group']

    # Map bias fraction to per-class and per-group stats.
    training_dataset_stats = {
        0.95: {
            'per_class': [3682, 1113],
            'per_group': [3498, 184, 56, 1057]
        },
        1.0: {
            'per_class': [3694, 1101]
        }
    }
    counts  = training_dataset_stats[bias_fraction][mode]
    counts  = torch.Tensor(counts)
    fracs   = 1 / counts
    weights = fracs / torch.max(fracs)

    return weights

class WaterbirdsEditing(WaterbirdsOrig):

    def __init__(self, root, cfg, split='train', transform=None):
        super().__init__(root, cfg, split, transform, metadata='/home/lisabdunlap/EditingClassifiers/data/waterbirds/land_birds.csv')
        # df = pd.read_csv('/home/lisabdunlap/EditingClassifiers/data/waterbirds/metadata.csv')
        # df['img_id']

    def __getitem__(self, index):

        results = super().__getitem__(index)
        return {
            'img': results['image'], 
            'mask': results['seg'], 
            'label': results['label']
            }