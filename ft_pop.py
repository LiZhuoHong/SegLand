import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import networks
import dataset

import random
import time
import logging
import utils.pyt_utils as my_utils
from loss import get_loss
from engine import Engine

BATCH_SIZE = 8
DATA_DIRECTORY = '/data/wzx99/pascal-context'
DATA_LIST_PATH = './dataset/list/context/train.txt'
VAL_LIST_PATH = './dataset/list/context/val.txt'
INPUT_SIZE = '512,512'
TEST_SIZE = '512,512'
BASE_SIZE = '2048,512'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_STEPS = 100
POWER = 0.9
RANDOM_SEED = 321
RESTORE_FROM = '/home/lsao/model/resnet_backbone/resnet101-imagenet.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5
SNAPSHOT_DIR = '/home/lsao/output/20200706'

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
    parser = argparse.ArgumentParser(description="Few-shot Segmentation Framework")
    parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='Dataset for training')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--train-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the training set.")
    parser.add_argument("--base-size", type=str, default=BASE_SIZE,
                        help="Base size of images for resize.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Which epoch to start.")
    parser.add_argument("--num-epoch", type=int, default=NUM_STEPS,
                        help="Number of training epochs.")
    parser.add_argument("--random-seed", type=str, default='123,234',
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="choose the number of workers.")
    parser.add_argument('--backbone', type=str, default='resnet50',
                    help='backbone model, can be: resnet101, resnet50 (default)')
    parser.add_argument("--os", type=int, default=8, help="output stride")
    parser.add_argument("--print-frequency", type=int, default=100,
                        help="Number of training steps.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3], help='validation fold')
    parser.add_argument('--shot', type=int, default=1, help='number of support pairs')
    parser.add_argument("--val-list", type=str, default=VAL_LIST_PATH,
                        help="Path to the file listing the images in the val set.")
    parser.add_argument("--test-batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--fix-bn", action="store_true", default=True,
                        help="whether to fix batchnorm during training.")
    parser.add_argument("--freeze-backbone", action="store_true", default=False,
                        help="whether to freeze the backbone during training.")
    parser.add_argument("--update-base", action="store_true", default=False,
                        help="whether to update base list during finetuning.")
    parser.add_argument("--update-epoch", type=int, default=1,
                        help="Number of epochs to update base list.")
    parser.add_argument("--fix-lr", action="store_true", default=False,
                        help="whether to fix learning rate during finetuning.")
    parser.add_argument("--filter-novel", action="store_true", default=False,
                        help="whether to filter images containing novel classes during training.")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="whether to use mix precision training.")
    return parser

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
            
def adjust_learning_rate(optimizer, lr, index_split=-1, scale_lr=10.):
    for index in range(len(optimizer.param_groups)):
        if index <= index_split:
            optimizer.param_groups[index]['lr'] = lr
        else:
            optimizer.param_groups[index]['lr'] = lr * scale_lr
    return lr

def adjust_learning_rate_poly(optimizer, learning_rate, i_iter, max_iter, power, split=-1):
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    lr = adjust_learning_rate(optimizer, lr, index_split=split)
    return lr

def main_func():
    """Create the model and start the training."""
    # global logger
    parser = get_parser()
    torch_ver = torch.__version__[:3]
    use_val = True
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            logger = my_utils.prep_experiment(args, need_writer=False)
        cudnn.benchmark = True
        seed_list = args.random_seed
        seeds = list(map(int, seed_list.split(',')))
        for seed in seeds:
            my_utils.set_seed(seed)

            # data loader
            h1, w1 = map(int, args.input_size.split(','))
            input_size = (h1, w1)
            h2, w2 = map(int, args.base_size.split(','))
            base_size = (h2, w2)

            trainset = eval('dataset.' + args.dataset + '_ft.GFSSegTrain')(
                args.data_dir, args.train_list, args.fold, args.shot,
                crop_size=input_size, base_size=base_size, 
                mode='train', seed=seed, filter=args.filter_novel)
            train_loader, train_sampler = engine.get_train_loader(trainset)
            args.ignore_label = trainset.ignore_label
            args.base_classes = len(trainset.base_classes)

            testset = eval('dataset.' + args.dataset + '.GFSSegVal')(
                args.data_dir, args.val_list, args.fold, 
                base_size=base_size, resize_label=True, use_novel=True, use_base=True)
            test_loader, test_sampler = engine.get_test_loader(testset)
            args.novel_classes = len(testset.novel_classes)
            args.num_classes = testset.num_classes + 1 # consider background as a class

            if engine.distributed:
                test_sampler.set_epoch(0)

            if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                logger.info('>>>>>>> Current Seed {}: <<<<<<<'.format(seed))
                logger.info('[Trainset] base_cls_list: ' + str(trainset.base_classes))
                logger.info('[Trainset] {} valid images are loaded!'.format(len(trainset)))
                logger.info('[Trainset] novel_id_list: ' + str(trainset.novel_id_list))
                logger.info('[Testset] novel_cls_list: ' + str(testset.novel_classes))
                logger.info('[Testset] {} valid images are loaded!'.format(len(testset)))

            criterion = get_loss(args)
            if engine.distributed:
                BatchNorm = nn.SyncBatchNorm
            else:
                BatchNorm = nn.BatchNorm2d

            stride = args.os
            assert stride in [8, 16, 32]
            dilated = False if stride == 32 else True 
            seg_model = eval('networks.' + args.model + '.GFSS_Model')(
                n_base=args.base_classes, criterion=criterion, 
                backbone=args.backbone, norm_layer=BatchNorm,
                dilated=dilated, os=stride, is_ft=True, n_novel=args.novel_classes
                )

            my_utils.load_model(seg_model, args.restore_from, is_restore=True)

            try:
                seg_model.init_cls_n()
                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    logger.info('Initialize cls_n with cls')
            except:
                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    logger.info('random initialize')

            if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                logger.info(seg_model)

            try:
                params, split_index = seg_model.get_ft_params(args.learning_rate)
            except:
                params = my_utils.get_parameters(seg_model, lr=args.learning_rate, freeze_backbone=args.freeze_backbone)
                split_index = -1 if args.freeze_backbone else 0

            optimizer = optim.SGD(params, lr=args.learning_rate, 
                momentum=args.momentum, weight_decay=args.weight_decay)        
            optimizer.zero_grad()

            model = engine.data_parallel(seg_model.cuda())
            loss_scaler = my_utils.NativeScalerWithGradNormCount()

            if not os.path.exists(args.snapshot_dir) and engine.local_rank == 0:
                os.makedirs(args.snapshot_dir)

            global_iteration = args.start_epoch * len(train_loader)
            max_iteration = args.num_epoch * len(train_loader)
            loss_dict_memory = {}
            lr = args.learning_rate
            best_miou = 0
            best_biou = 0
            best_niou = 0
            best_epoch = 0
            for epoch in range(args.start_epoch, args.num_epoch):
                my_utils.set_seed(seed+epoch)

                epoch_log = epoch + 1
                if engine.distributed:
                    train_sampler.set_epoch(epoch)

                model.module.train_mode() # freeze backbone and BN

                for i, data in enumerate(train_loader):
                    global_iteration += 1
                    img, mask, img_b, mask_b, cls_idx = data
                    img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)
                    img_b, mask_b = img_b.cuda(non_blocking=True), mask_b.cuda(non_blocking=True)

                    if not args.fix_lr:
                        lr = adjust_learning_rate_poly(optimizer, args.learning_rate, global_iteration-1, max_iteration, args.power, split_index)
                    
                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        loss_dict = model(img, mask, img_b, mask_b)

                    total_loss = loss_dict['total_loss']
                    grad_norm = loss_scaler(total_loss, optimizer, clip_grad=5.0, parameters=model.parameters())
                    optimizer.zero_grad()

                    for loss_name, loss in loss_dict.items():
                        loss_dict_memory[loss_name] = engine.all_reduce_tensor(loss).item()

                    if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                        if i % args.print_frequency == 0:
                            print_str = 'Epoch{}/Iters{}'.format(epoch_log, global_iteration) \
                                + ' Iter{}/{}:'.format(i + 1, len(train_loader)) \
                                + ' lr=%.2e' % lr \
                                + ' grad_norm=%.4f' % grad_norm 
                            for loss_name, loss_value in loss_dict_memory.items():
                                print_str += ' %s=%.4f' % (loss_name, loss_value)
                            logger.info(print_str)

                if args.update_base and epoch_log % args.update_epoch == 0:
                    trainset.update_base_list()

                if use_val and (epoch % args.update_epoch == 0 or epoch == args.num_epoch-1):
                    inter, union = validate(model, test_loader, args)
                    inter =  engine.all_reduce_tensor(inter, norm=False)
                    union =  engine.all_reduce_tensor(union, norm=False)
                    miou_array = inter / union
                    miou_array = miou_array.cpu().numpy()
                    base_miou = np.nanmean(miou_array[:args.base_classes+1])
                    novel_miou = np.nanmean(miou_array[args.base_classes+1:])
                    total_miou = np.nanmean(miou_array)

                    if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                        if total_miou >= best_miou:
                            # if (total_miou-best_miou > 0.001) or (novel_miou >= best_niou):
                            if (base_miou - best_biou > 0.001):
                                print('taking snapshot ...')
                                if torch_ver < '1.6':
                                    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'best_{}.pth'.format(seed)))
                                else:
                                    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'best_{}.pth'.format(seed)), _use_new_zipfile_serialization=False)
                                best_miou = total_miou
                                best_biou = base_miou
                                best_niou = novel_miou
                                best_epoch = epoch_log
                        logger.info('>>>>>>> Evaluation Results: <<<<<<<')
                        logger.info('meanIU: {:.2%}, baseIU: {:.2%}, novelIU: {:.2%}, best_IU: {:.2%}, best_bIU: {:.2%}, best_nIU: {:.2%}, best_epoch: {}'.format(total_miou,base_miou,novel_miou,best_miou,best_biou,best_niou,best_epoch))
                        logger.info('>>>>>>> ------------------- <<<<<<<')
                        if epoch % 50 == 0 or epoch == args.num_epoch-1:
                            print('taking snapshot ...')
                            if torch_ver < '1.6':
                                torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'epoch_{}_{}.pth'.format(epoch, seed)))
                            else:
                                torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'epoch_{}_{}.pth'.format(epoch, seed)), _use_new_zipfile_serialization=False)

                    # if (epoch_log - best_epoch) > args.num_epoch // 4:
                    #     if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    #         logger.info('Early stopped at epoch {}, best epoch: {}'.format(epoch_log, best_epoch))
                    #     break

def validate(model, dataloader, args):
    '''
        Validation on base classes (only for training)
    '''
    model.eval()
    num_classes = args.num_classes # 0 for background
    inter_meter = torch.zeros(num_classes).cuda()
    union_meter = torch.zeros(num_classes).cuda()

    for idx, data in enumerate(dataloader):
        img, mask, id = data
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.fp16):
                output = model(img)
            h, w = mask.size(1), mask.size(2)
            output = F.interpolate(input=output, size=(h, w), mode='bilinear', align_corners=True)

        output = output.max(1)[1]
        intersection, union, _ = my_utils.intersectionAndUnionGPU(output, mask, num_classes, args.ignore_label)
        inter_meter += intersection
        union_meter += union

    return inter_meter, union_meter

if __name__ == '__main__':
    main_func()