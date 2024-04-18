import numpy as np
import random
import torch
import cv2
from torch.utils import data

class BaseDataset(data.Dataset):
    def __init__(self, mode='train', crop_size=(512, 512), 
            ignore_label=255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], base_size=(512, 512)):
        self.crop_size = crop_size

        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.padding = [val*255.0 for val in self.mean]
        
        self.ids = []
        self.mode = mode

        self.base_size = base_size
        self.ratio_range = (0.9, 1.1)
        self.blur_radius = 5
        self.rotate_range = (-10, 10)
        self.fg_remain_ratio = 0.85

    def __len__(self):
        return len(self.ids)

    def normalize(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def totensor(self, image, label=None):
        image = image.transpose((2, 0, 1)) # [H, W, C] -> [C, H, W]
        image = torch.from_numpy(image.copy()).float()
        if label is not None:
            label = torch.from_numpy(label.copy()).long()
            return image, label
        return image

    def resize(self, image, label=None, random_scale=False):
        if random_scale:
            min_ratio, max_ratio = self.ratio_range
            f_scale = random.random() * (max_ratio - min_ratio) + min_ratio
            dsize = int(image.shape[1] * f_scale + 0.5), int(image.shape[0] * f_scale + 0.5)
            image_scale = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
            label_scale = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)
            return image_scale, label_scale
        else:
            output_size = self.base_size[0], self.base_size[1]
            scale_factor = min(max(output_size) / max(image.shape[:2]), min(output_size) / min(image.shape[:2]))
            new_w = int(image.shape[1] * scale_factor + 0.5)
            new_h = int(image.shape[0] * scale_factor + 0.5)
            dsize = new_w, new_h
            image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
            if label is not None:
                label = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)
                return image, label
            else:
                return image
    
    def fixed_resize(self, image, label=None):
        output_size = self.base_size[0], self.base_size[1]
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, output_size, interpolation=cv2.INTER_NEAREST)
            return image, label
        else:
            return image

    def square_resize(self, image, label=None, random_scale=False):
        if random_scale:
            min_ratio, max_ratio = self.ratio_range
            f_scale = random.random() * (max_ratio - min_ratio) + min_ratio
            dsize = int(self.base_size[1] * f_scale + 0.5), int(self.base_size[0] * f_scale + 0.5)
        else:
            dsize = self.base_size[1], self.base_size[0]
        image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)
            return image, label
        else:
            return image

    def pad(self, output_size, image, label=None):
        pad_h = max(output_size[0] - image.shape[0],0)
        pad_w = max(output_size[1] - image.shape[1],0)
        if pad_h > 0 or pad_w > 0:
            image_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            if label is not None:
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                    pad_w, cv2.BORDER_CONSTANT,
                    value=(self.ignore_label,))
        else:
            image_pad, label_pad = image, label
        if label is not None:
            return image_pad, label_pad
        else:
            return image_pad

    def random_flip(self, image, label, p=0.5):
        if random.random() < p:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        return image, label

    def random_gaussian(self, image, p=0.5):
        if random.random() < p:
            image = cv2.GaussianBlur(image, (self.blur_radius, self.blur_radius), 0)
        return image

    def random_rotate(self, image, label, p=0.5):
        if random.random() < p:
            rotate_cnt = 0
            while (rotate_cnt < 5):
                angle = self.rotate_range[0] + (self.rotate_range[1] - self.rotate_range[0]) * random.random()
                h, w = label.shape
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                image_tmp = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
                label_tmp = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
                if np.sum(label_tmp == 1) > 0:
                    break
                rotate_cnt += 1
            if rotate_cnt < 5:
                image = image_tmp
                label = label_tmp
        return image, label
    
    def fixed_random_rotate(self, image, label):
        k = int(random.random() // 0.25)
        image = np.rot90(image, k, (0, 1))
        label = np.rot90(label, k, (0, 1))
        return image, label

    def crop(self, image, label, target_cls=1):
        img_h, img_w = label.shape
        crop_h, crop_w = self.crop_size
        margin_h = max(img_h - crop_h, 0)
        margin_w = max(img_w - crop_w, 0)
        if self.mode == 'train':
            h_off = np.random.randint(0, margin_h + 1)
            w_off = np.random.randint(0, margin_w + 1)
            label_c = label[h_off : h_off+crop_h, w_off : w_off+crop_w]
            l = np.unique(label_c).tolist()
            while len(l) == 1 and self.ignore_label in l:
                h_off = np.random.randint(0, margin_h + 1)
                w_off = np.random.randint(0, margin_w + 1)
                label_c = label[h_off : h_off+crop_h, w_off : w_off+crop_w]
                l = np.unique(label_c).tolist()
            # if self.fg_remain_ratio > 0:
            #     raw_pos_num = np.sum(label == target_cls)
            #     label_temp = label[h_off : h_off+crop_h, w_off : w_off+crop_w]
            #     pos_num = np.sum(label_temp == target_cls)
            #     crop_cnt = 0
            #     while (pos_num < self.fg_remain_ratio * raw_pos_num and crop_cnt <= 30):
            #         h_off = np.random.randint(0, margin_h + 1)
            #         w_off = np.random.randint(0, margin_w + 1)
            #         label_temp = label[h_off : h_off+crop_h, w_off : w_off+crop_w]
            #         pos_num = np.sum(label_temp == 1)
            #         crop_cnt += 1
            #     if pos_num < self.fg_remain_ratio * raw_pos_num:
            #         image = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            #         label = cv2.resize(label, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)            
            #         return image, label
        else:
            h_off = int(round(margin_h / 2.))
            w_off = int(round(margin_w / 2.))
        image = image[h_off : h_off+crop_h, w_off : w_off+crop_w]
        label = label[h_off : h_off+crop_h, w_off : w_off+crop_w]
        return image, label