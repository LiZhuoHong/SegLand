import os
import os.path as osp
import numpy as np
import random
import cv2
from collections import defaultdict
import rasterio

from .base_dataset import BaseDataset

class GFSSegTrain(BaseDataset):
    num_classes = 11
    def __init__(self, root, list_path, fold, shot=1, mode='train', crop_size=(512, 512),
             ignore_label=255, base_size=(1024,1024), resize_label=False, seed=123, filter=False, use_base=True):
        super(GFSSegTrain, self).__init__(mode, crop_size, ignore_label, base_size=base_size)
        assert mode in ['train', 'val_supp']
        self.root = root
        self.list_path = list_path
        self.shot = shot
        self.mode = mode
        self.resize_label = resize_label
        self.use_base = use_base
        self.ratio_range = (0.8, 1.25)
        self.img_dir = 'images'
        self.lbl_dir = 'labels'

        # base classes = all classes - novel classes
        self.base_classes = set(range(1, 8, 1))
        # novel classes
        self.novel_classes = set(range(8, self.num_classes + 1))

        filter_flag = True if (self.mode == 'train' and filter) else False
        list_dir = os.path.dirname(self.list_path)
        if filter_flag:
            list_dir = list_dir + '_filter'
        list_saved = os.path.exists(os.path.join(list_dir, 'train_base_class%s.txt'%(list(self.base_classes)[0])))
        if list_saved:
            print('id files exist...')
            self.base_cls_to_ids = defaultdict(list)
            for cls in self.base_classes:
                with open(os.path.join(list_dir, 'train_base_class%s.txt'%cls), 'r') as f:
                    self.base_cls_to_ids[cls] = f.read().splitlines()
        else:
            '''
            fold0/train_fold0.txt: training images containing base classes (novel classes will be ignored during training)
            fold0/train_base_class[6-20].txt: training images containing base class [6-20]
            fold0/train_novel_class[1-5].txt: training images containing novel class [1-5]
            '''
            with open(os.path.join(self.list_path), 'r') as f:
                self.ids = f.read().splitlines()
            print('checking ids...')

            self.base_cls_to_ids, self.novel_cls_to_ids = self._filter_and_map_ids(filter_intersection=filter_flag)
            for cls in self.base_classes:
                with open(os.path.join(list_dir, 'train_base_class%s.txt'%cls), 'w') as f:
                    for id in self.base_cls_to_ids[cls]:
                        f.write(id+"\n")

        with open(os.path.join(list_dir, 'all_%sshot_seed%s.txt'%(shot, seed)), 'r') as f:
            self.novel_id_list = f.read().splitlines()
        if self.use_base:
            self.supp_cls_id_list, self.base_id_list = self._get_supp_list()
        else:
            self.supp_cls_id_list = self.novel_id_list

    def __len__(self):
        if self.mode == 'val_supp':
            return len(self.novel_classes) + len(self.base_classes) if self.use_base else len(self.novel_classes)
        else:
            return len(self.base_id_list)

    def update_base_list(self):
        base_list = list(self.base_classes)
        base_id_list = []
        id_s_list = []
        base_with_novel = 0
        for target_cls in base_list:
            file_class_chosen = self.base_cls_to_ids[target_cls]
            num_file = len(file_class_chosen)
            if num_file < self.shot:
                print('extend images with repeating')
                for i in range(num_file):
                    id_s = file_class_chosen[i]
                    base_id_list.append(id_s)
                    label = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id_s)).read()
                    label = label[0]
                    label_class = np.unique(label).tolist()
                    if 0 in label_class:
                        label_class.remove(0)
                    if set(label_class).issubset(self.base_classes):
                        pass
                    else:
                        base_with_novel += 1
                for i in range(self.shot-num_file):
                    id_s = file_class_chosen[random.randint(1, num_file) - 1]
                    base_id_list.append(id_s)
                    label = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id_s)).read()
                    label = label[0]
                    label_class = np.unique(label).tolist()
                    if 0 in label_class:
                        label_class.remove(0)
                    if set(label_class).issubset(self.base_classes):
                        pass
                    else:
                        base_with_novel += 1
            else:
                id_s_sampled = random.choices(list(range(num_file)), k=self.shot)
                for k in range(self.shot):
                    id_s = file_class_chosen[id_s_sampled[k]]
                    id_s_list.append(id_s)
                    base_id_list.append(id_s)
                    label = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id_s)).read()
                    label = label[0]
                    label_class = np.unique(label).tolist()
                    if 0 in label_class:
                        label_class.remove(0)
                    if set(label_class).issubset(self.base_classes):
                        pass
                    else:
                        base_with_novel += 1
        print('%s base images contain novel classes'%base_with_novel)
        
        self.supp_cls_id_list = self.novel_id_list + base_id_list
        self.base_id_list = base_id_list

    def _get_supp_list(self):
        base_list = list(self.base_classes)
        base_id_list = []
        novel_id_list = self.novel_id_list
        id_s_list = []
        for id in novel_id_list:
            id_s_list.append(id)

        base_with_novel = 0
        for target_cls in base_list:
            file_class_chosen = self.base_cls_to_ids[target_cls]
            num_file = len(file_class_chosen)
            if num_file < self.shot:
                print('extend images with repeating')
                for i in range(num_file):
                    id_s = file_class_chosen[i]
                    base_id_list.append(id_s)
                    label = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id_s)).read()
                    label = label[0]
                    label_class = np.unique(label).tolist()
                    if 0 in label_class:
                        label_class.remove(0)
                    if set(label_class).issubset(self.base_classes):
                        pass
                    else:
                        base_with_novel += 1
                for i in range(self.shot-num_file):
                    id_s = file_class_chosen[random.randint(1, num_file) - 1]
                    base_id_list.append(id_s)
                    label = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id_s)).read()
                    label = label[0]
                    label_class = np.unique(label).tolist()
                    if 0 in label_class:
                        label_class.remove(0)
                    if set(label_class).issubset(self.base_classes):
                        pass
                    else:
                        base_with_novel += 1
            else:
                id_s_sampled = random.choices(list(range(num_file)), k=self.shot)
                for k in range(self.shot):
                    id_s = file_class_chosen[id_s_sampled[k]]
                    id_s_list.append(id_s)
                    base_id_list.append(id_s)
                    label = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id_s)).read()
                    label = label[0]
                    label_class = np.unique(label).tolist()
                    if 0 in label_class:
                        label_class.remove(0)
                    if set(label_class).issubset(self.base_classes):
                        pass
                    else:
                        base_with_novel += 1
        print('%s base images contain novel classes'%base_with_novel)
        supp_cls_id_list = novel_id_list + base_id_list
        return supp_cls_id_list, base_id_list

    def __getitem__(self, index):
        if self.mode == 'val_supp':
            return self._get_val_support(index)
        else:
            return self._get_train_sample(index)

    def _get_train_sample(self, index):
        id_b = self.base_id_list[index]
        id = random.choice(self.novel_id_list)
        image = rasterio.open(osp.join(self.root, self.img_dir, '%s.tif'%id)).read()
        label = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id)).read()
        image = np.rollaxis(image, 0, 3)
        label = label[0]

        label = np.where(label == 0, self.ignore_label, label)

        image_b = rasterio.open(osp.join(self.root, self.img_dir, '%s.tif'%id_b)).read()
        label_b = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id_b)).read()
        image_b = np.rollaxis(image_b, 0, 3)
        label_b = label_b[0]

        # date augmentation & preprocess
        image, label = self.crop(image, label)
        image, label = self.pad(self.crop_size, image, label)
        image, label = self.random_flip(image, label)
        image, label = self.fixed_random_rotate(image, label)
        image = self.normalize(image)
        image, label = self.totensor(image, label)

        # date augmentation & preprocess
        image_b, label_b = self.crop(image_b, label_b)
        image_b, label_b = self.pad(self.crop_size, image_b, label_b)
        image_b, label_b = self.random_flip(image_b, label_b)
        image_b, label_b = self.fixed_random_rotate(image_b, label_b)
        image_b = self.normalize(image_b)
        image_b, label_b = self.totensor(image_b, label_b)

        return image, label, image_b, label_b, id 

    def _get_val_support(self, index):
        if self.use_base:
            if index < len(self.base_classes):
                cls_id_list = self.base_id_list
                cls_idx = index
                target_cls = list(self.base_classes)[cls_idx]
            else:
                cls_id_list = self.novel_id_list
                cls_idx = index - len(self.base_classes)
                target_cls = list(self.novel_classes)[cls_idx]
        else:
            cls_id_list = self.novel_id_list
            cls_idx = index
            target_cls = list(self.novel_classes)[cls_idx]

        id_s_list, image_s_list, label_s_list = [], [], []

        for k in range(self.shot):
            id_s = cls_id_list[cls_idx*self.shot+k]
            image = rasterio.open(osp.join(self.root, self.img_dir, '%s.tif'%id_s)).read()
            label = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%id_s)).read()
            image = np.rollaxis(image, 0, 3)
            label = label[0]

            new_label = label.copy()
            new_label[(label != target_cls) & (label != self.ignore_label)] = 0
            new_label[label == target_cls] = 1
            label = new_label.copy()
            # date augmentation & preprocess
            image, label = self.random_rotate(image, label)
            image, label = self.random_flip(image, label)
            image = self.normalize(image)
            image, label = self.totensor(image, label)
            id_s_list.append(id_s)
            image_s_list.append(image)
            label_s_list.append(label)
        # print(target_cls)
        # print(id_s_list)
        return image_s_list, label_s_list, id_s_list, target_cls
        
    def _filter_and_map_ids(self, filter_intersection=False):
        base_cls_to_ids = defaultdict(list)
        novel_cls_to_ids = defaultdict(list)
        for i in range(len(self.ids)):
            mask = rasterio.open(osp.join(self.root, self.lbl_dir, '%s.tif'%self.ids[i])).read()
            mask = mask[0]
            label_class = np.unique(mask).tolist()
            if 0 in label_class:
                label_class.remove(0)
            valid_base_classes = set(np.unique(mask)) & self.base_classes
            valid_novel_classes = set(np.unique(mask)) & self.novel_classes

            if valid_base_classes:
                new_label_class = []
                if filter_intersection:
                    if set(label_class).issubset(self.base_classes):
                        for cls in valid_base_classes:
                            new_label_class.append(cls)
                else:
                    for cls in valid_base_classes:
                        new_label_class.append(cls)

                if len(new_label_class) > 0:
                    # map each valid class to a list of image ids
                    for cls in new_label_class:
                        base_cls_to_ids[cls].append(self.ids[i])

            if valid_novel_classes:
            # remove images whose valid objects are all small (according to PFENet)
                new_label_class = []
                for cls in valid_novel_classes:
                    new_label_class.append(cls)

                if len(new_label_class) > 0:
                    # map each valid class to a list of image ids
                    for cls in new_label_class:
                        novel_cls_to_ids[cls].append(self.ids[i])

        return base_cls_to_ids, novel_cls_to_ids