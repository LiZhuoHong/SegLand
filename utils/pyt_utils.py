# encoding: utf-8
import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict
from datetime import datetime
import logging
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
from math import ceil
import cv2
import random
# from torch._six import inf
from torch import inf

def reduce_tensor(tensor, dst=0, norm=False, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, dst, op)
    if norm and dist.get_rank() == dst:
        tensor.div_(world_size)

    return tensor

def get_logger(prefix, output_dir, date_str):
    logger = logging.getLogger('Segmentation')

    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    filename = os.path.join(output_dir, prefix + '_' + date_str + '.log')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    # only rank 0 will add a FileHandler
    if rank == 0:
        file_handler = logging.FileHandler(filename, 'w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    return logger

def prep_experiment(args, need_writer=False):
    '''
    Make output directories, setup logging, Tensorboard, snapshot code.
    '''
    log_path = args.snapshot_dir + '/log'

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    args.log_path = log_path

    args.date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    logger = get_logger('', log_path, args.date_str)

    open(os.path.join(args.snapshot_dir, args.date_str + '.txt'), 'w').write(
        str(args) + '\n\n')
    if need_writer:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir=log_path)
        return writer, logger
    else:
        return logger

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    with torch.no_grad():
        tensor = tensor.detach()
        dist.all_reduce(tensor, op)
        if norm:
            tensor.div_(world_size)
    return tensor

def load_model(model, model_file, is_restore=False, backbone_only=False):
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        checkpoint = torch.load(model_file, map_location=device)
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict

    if backbone_only:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'backbone.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logging.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logging.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logging.info(
        "Load model from {}, Time usage:\n\tIO: {}, initialize parameters: {}".format(model_file,
            t_ioend - t_start, t_end - t_ioend))

    return model

def parse_devices(input_devices):
    if input_devices.endswith('*'):
        devices = list(range(torch.cuda.device_count()))
        return devices

    devices = []
    for d in input_devices.split(','):
        if '-' in d:
            start_device, end_device = d.split('-')[0], d.split('-')[1]
            assert start_device != ''
            assert end_device != ''
            start_device, end_device = int(start_device), int(end_device)
            assert start_device < end_device
            assert end_device < torch.cuda.device_count()
            for sd in range(start_device, end_device + 1):
                devices.append(sd)
        else:
            device = int(d)
            assert device < torch.cuda.device_count()
            devices.append(device)

    logging.info('using devices {}'.format(
        ', '.join([str(d) for d in devices])))

    return devices

def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix
    
def get_parameters_freeze_backbone(model):
    param_list = []
    key_list = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if 'backbone' not in key and value.requires_grad:
            param_list.append(value)
            key_list.append(key)
        else:
            value.requires_grad = False
    print('not frozen: ', key_list)
    params = [{'params': param_list}]
    return params

def get_parameters(model, lr, scale=10.0, freeze_backbone=False, fix_bn=False, logger=None):
    wd_0 = []
    lr_1 = []
    lr_10 = []
    key_list = []
    
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if value.requires_grad:
            if 'backbone' not in key:
                if 'bias' in key:
                    wd_0.append(value)
                else:
                    lr_10.append(value)
                key_list.append(key)
            else:
                if freeze_backbone:
                    value.requires_grad = False
                else:
                    if fix_bn and 'bn' in key:
                        value.requires_grad = False
                    else:
                        lr_1.append(value)
                        key_list.append(key)

    print('not frozen: ', key_list)
    if freeze_backbone:
        params = [{'params': wd_0, 'lr': lr * scale, 'weight_decay': 0.0},
                {'params': lr_10, 'lr': lr * scale}]
    else:
        params = [{'params': lr_1, 'lr': lr},
                    {'params': wd_0, 'lr': lr * scale, 'weight_decay': 0.0},
                    {'params': lr_10, 'lr': lr * scale}]
    return params

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)