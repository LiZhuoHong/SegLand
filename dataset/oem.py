import os
import os.path as osp
import numpy as np
import random
import cv2
import rasterio
import torch
from collections import defaultdict

from .base_dataset import BaseDataset

class GFSSegTrain(BaseDataset):
    num_classes = 11
    def __init__(self, root, list_path, fold, shot=1, mode='train', crop_size=(512, 512),
             ignore_label=255, base_size=(1024, 1024), resize_label=False, filter=False, seed=123):
        super(GFSSegTrain, self).__init__(mode, crop_size, ignore_label, base_size=base_size)
        assert mode in ['train', 'val_supp']
        self.root = root
        self.list_path = list_path
        self.shot = shot
        self.mode = mode
        self.resize_label = resize_label
        self.img_dir = 'images'
        self.lbl_dir = 'labels'

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
    
        self.ratio_range = (0.5, 1)

        # base classes = all classes - novel classes
        self.base_classes = set(range(1, 8, 1))
        # novel classes
        self.novel_classes = set(range(8, self.num_classes + 1))

        list_dir = os.path.dirname(self.list_path)
        # list_dir = list_dir + '/fold%s'%fold
        list_saved = os.path.exists(os.path.join(list_dir, 'train.txt'))
        if list_saved:
            print('id files exist...')
            with open(os.path.join(list_dir, 'train.txt'), 'r') as f:
                self.data_list = f.read().splitlines()
        else:
            raise FileNotFoundError

    def __len__(self):
        if self.mode == 'val_supp':
            return len(self.novel_classes)
        else:
            return len(self.data_list)

    def __getitem__(self, index):
        return self._get_train_sample(index)

    def _get_train_sample(self, index):
        id = self.data_list[index]
        image = rasterio.open(osp.join(self.root, self.img_dir, '%s.tif'%id)).read()
        label = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id)).read()
        image = np.rollaxis(image, 0, 3)
        label = label[0]
        
        # if id in ['baybay_23', 'chiclayo_26', 'coxsbazar_63', 'coxsbazar_70', 'coxsbazar_80', 'daressalaam_51', 
        #           'dhaka_28', 'ica_41', 'khartoum_46', 'koeln_32', 'kyoto_8', 'kyoto_13', 'kyoto_30', 'kyoto_47', 
        #           'malopolskie_39', 'maputo_10', 'maputo_12', 'maputo_16', 'melbourne_67', 'monrovia_25', 'piura_7', 
        #           'santiago_63', 'svaneti_20', 'tyrolw_9', 'tyrolw_63', 'tyrolw_64', 'viru_24']:
        #     label = np.where(label == 6, 0, label)

        # date augmentation & preprocess
        image, label = self.crop(image, label)
        image, label = self.pad(self.crop_size, image, label)
        image, label = self.random_flip(image, label)
        image, label = self.fixed_random_rotate(image, label)
        image = self.normalize(image)
        image, label = self.totensor(image, label)

        return image, label, id

class GFSSegVal(BaseDataset):
    num_classes = 11
    def __init__(self, root, list_path, fold, crop_size=(512, 512),
             ignore_label=255, base_size=(1024, 1024), resize_label=False, use_novel=True, use_base=True):
        super(GFSSegVal, self).__init__('val', crop_size, ignore_label, base_size=base_size)
        self.root = root
        self.list_path = list_path
        self.fold = fold
        self.resize_label = resize_label
        self.use_novel = use_novel
        self.use_base = use_base
        self.img_dir = 'images'
        self.lbl_dir = 'labels'

        # base classes = all classes - novel classes
        self.base_classes = set(range(1, 8, 1))
        # novel classes
        self.novel_classes = set(range(8, self.num_classes + 1))

        with open(os.path.join(self.list_path), 'r') as f:
            self.ids = f.read().splitlines()
#         self.ids = ['2007_005273', '2011_003019']
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        image = rasterio.open(osp.join(self.root, self.img_dir, '%s.tif'%id)).read()
        image = np.rollaxis(image, 0, 3)

        if os.path.exists(osp.join(self.root, self.lbl_dir, '%s.tif'%id)):
            label = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id)).read()
            label = label[0]

            new_label = label.copy()
            label_class = np.unique(label).tolist()
            base_list = list(self.base_classes)
            novel_list = list(self.novel_classes)

            for c in label_class:
                if c in base_list:
                    if self.use_base:
                        new_label[label == c] = (base_list.index(c) + 1)    # 0 as background
                    else:
                        new_label[label == c] = 0
                elif c in novel_list:
                    if self.use_novel:
                        if self.use_base:
                            new_label[label == c] = (novel_list.index(c) + len(base_list) + 1)
                        else:
                            new_label[label == c] = (novel_list.index(c) + 1)
                    else:
                        new_label[label == c] = 0

            label = new_label.copy()
            # date augmentation & preprocess
            if self.resize_label:
                image, label = self.resize(image, label)
                image = self.normalize(image)
                image, label = self.pad(self.base_size, image, label)
            else:
                image = self.normalize(image)
            image, label = self.totensor(image, label)

            return image, label, id
        
        else:
            image = self.normalize(image)
            image = image.transpose((2, 0, 1)) # [H, W, C] -> [C, H, W]
            image = torch.from_numpy(image.copy()).float()
            return image, image, id