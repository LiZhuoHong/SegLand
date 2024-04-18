import argparse
import cv2
from datetime import datetime
import numpy as np
import sys
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import networks
import dataset
import os
from math import ceil
from utils.pyt_utils import load_model, get_confusion_matrix, set_seed, get_logger, intersectionAndUnionGPU
import rasterio

from engine import Engine

DATA_DIRECTORY = '/data/wzx99/pascal-context'
VAL_LIST_PATH = './dataset/list/context/val.txt'
COLOR_PATH = './dataset/list/context/context_colors.txt'
BATCH_SIZE = 1
INPUT_SIZE = '512,512'
BASE_SIZE = '2048,512'
RESTORE_FROM = '/model/lsa1997/deeplabv3_20200106/snapshots/CS_scenes_40000.pth'
SAVE_PATH = '/output/predict'
RANDOM_SEED = '123,234,345'

colormap = {
    0:  (147, 147, 147),
    1:  (49, 139, 87),
    2: (0, 255, 0),
    3: (128, 0, 0),
    4: (75, 181, 73),
    5: (245, 245, 245),
    6: (35, 91, 200),
    7: (247, 142, 82),
    # 8:  (166, 166, 171),
    # 9:  (3, 7, 255),
    # 10: (255, 242, 0),
    # 11: (170, 255, 0),
    8:  (255, 0, 0),
    9:  (255, 0, 255),
    10: (0, 255, 255),
    11: (255, 255, 0),
}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='Dataset for training')
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--train-list", type=str, default='',
                        help="Path to the file listing the images in the training set.")
    parser.add_argument("--val-list", type=str, default=VAL_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--test-batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument('--backbone', type=str, default='resnet101',
                    help='backbone model, can be: resnet101 (default), resnet50')
    parser.add_argument("--base-size", type=str, default=BASE_SIZE,
                        help="Base size of images for resize.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--save", type=str2bool, default='False',
                        help="save predicted image.")
    parser.add_argument("--os", type=int, default=8, help="output stride")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="path to save results")
    parser.add_argument("--random-seed", type=str, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3], help='validation fold')
    parser.add_argument('--shot', type=int, default=1, help='number of support pairs')
    return parser

def main():
    """Create the model and start the evaluation process."""
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            save_path = args.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            logger = get_logger('', save_path, date_str)

        cudnn.benchmark = False
        cudnn.deterministic = True

        h, w = map(int, args.base_size.split(','))
        args.base_size = (h, w)

        testset = eval('dataset.' + args.dataset + '.GFSSegVal')(
            args.data_dir, args.val_list, args.fold, 
            base_size=args.base_size, resize_label=False, use_novel=True, use_base=True)

        test_loader, test_sampler = engine.get_test_loader(testset)
        args.ignore_label = testset.ignore_label
        args.base_classes = len(testset.base_classes)
        args.novel_classes = len(testset.novel_classes)
        args.num_classes = testset.num_classes + 1 # consider background as a class

        # base_classes = sorted(list(testset.base_classes))
        # novel_classes = sorted(list(testset.novel_classes))
        base_classes = list(testset.base_classes)
        novel_classes = list(testset.novel_classes)
        
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            logger.info('[Testset] base_cls_list: ' + str(base_classes))
            logger.info('[Testset] novel_cls_list: ' + str(novel_classes))
            logger.info('[Testset] {} valid images are loaded!'.format(len(testset)))

        if engine.distributed:
            test_sampler.set_epoch(0)

        stride = args.os
        assert stride in [8, 16, 32]
        dilated = False if stride == 32 else True 
        seg_model = eval('networks.' + args.model + '.GFSS_Model')(
            n_base=args.base_classes, backbone=args.backbone, 
            dilated=dilated, os=stride, n_novel=args.novel_classes, is_ft=True)
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            logger.info(seg_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model.to(device)
        model = engine.data_parallel(seg_model)

        seed_list = args.random_seed
        seeds = list(map(int, seed_list.split(',')))
        
        for seed in seeds:
            restore_model = args.restore_from[:-4] + '_' + str(seed) + '.pth'
            load_model(model, restore_model)

            model.eval()

            # generalized few-shot segmentation evaluation (base + novel)
            confusion_matrix = np.zeros((args.num_classes, args.num_classes))
            miou_array = np.zeros((args.num_classes))
            for idx, data in enumerate(test_loader):
                image, label, id = data
                image = image.cuda()

                with torch.no_grad():
                    output = model(image)
                    h, w = label.size(1), label.size(2)
                    longside = max(h, w)
                    output = F.interpolate(input=output, size=(longside, longside), mode='bilinear', align_corners=True)

                seg_pred = np.asarray(np.argmax(output.cpu().numpy(), axis=1), dtype=np.uint8)

                if len(label.shape) < 3:
                    seg_gt = np.asarray(label.numpy(), dtype=np.int)
                    pad_gt = np.ones((seg_gt.shape[0], longside, longside), dtype=np.int) * args.ignore_label
                    pad_gt[:, :h, :w] = seg_gt
                    seg_gt = pad_gt

                    ignore_index = seg_gt != args.ignore_label
                    seg_gt = seg_gt[ignore_index]
                    seg_pred = seg_pred[ignore_index]
                    confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)
                else:
                    input_profile_fn = os.path.join(args.data_dir, 'image', id[0] + ".tif")
                    output_profile = rasterio.open(input_profile_fn).profile.copy()
                    output_profile["driver"] = "GTiff"
                    output_profile["dtype"] = "uint8"
                    output_profile["count"] = 1
                    output_profile["nodata"] = 0
                    output_fn = os.path.join(save_path, id[0] + ".tif")
                    with rasterio.open(output_fn, "w", **output_profile) as f:
                        f.write(seg_pred[0], 1)
                        f.write_colormap(1, colormap)

            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            miou_array = (tp / (pos + res - tp))
            base_miou = np.nanmean(miou_array[:args.base_classes+1])
            novel_miou = np.nanmean(miou_array[args.base_classes+1:])
            total_miou = np.nanmean(miou_array)

            np.save(os.path.join(save_path, 'cmatrix_{}.npy'.format(seed)), confusion_matrix)

            if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                logger.info('>>>>>>> Current Seed {}: <<<<<<<'.format(seed))
                logger.info('meanIoU---base: mIoU {:.4f}.'.format(base_miou))
                logger.info('meanIoU---novel: mIoU {:.4f}.'.format(novel_miou))
                logger.info('meanIoU---total: mIoU {:.4f}.'.format(total_miou))

if __name__ == '__main__':
    main()
