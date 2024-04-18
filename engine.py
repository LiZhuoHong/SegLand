import os
import os.path as osp
import time
import argparse

import torch
import torch.distributed as dist
import numpy as np
import random
from utils.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, extant_file, reduce_tensor

# try:
#     from apex.parallel import DistributedDataParallel, SyncBatchNorm
# except ImportError:
#     raise ImportError(
#         "Please install apex from https://www.github.com/nvidia/apex .")


logger = get_logger()


class Engine(object):
    def __init__(self, custom_parser=None):
        logger.info(
            "PyTorch Version {}".format(torch.__version__))
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1

        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
            self.devices = [i for i in range(self.world_size)]
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            self.devices =  [i for i in range(len(gpus.split(',')))]
        else:
            self.devices = [0]
            # gpus = os.environ["NVIDIA_VISIBLE_DEVICES"]
            # self.devices =  [i for i in range(len(gpus.split(',')))]

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        p.add_argument('--local_rank', default=0, type=int,
                       help='process rank on node')

    def data_parallel(self, model):
        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model.cuda(),device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        else:
            model = torch.nn.DataParallel(model)
        return model


    def get_train_loader(self, train_dataset):
        train_sampler = None
        is_shuffle = True
        batch_size = self.args.batch_size
        num_workers = self.args.num_workers

        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
            batch_size = self.args.batch_size // self.world_size
            num_workers = self.args.num_workers // self.world_size
            is_shuffle = False

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       drop_last=True,
                                       shuffle=is_shuffle,
                                       pin_memory=True,
                                       sampler=train_sampler)

        return train_loader, train_sampler

    def get_test_loader(self, test_dataset):
        test_sampler = None
        is_shuffle = False
        batch_size = self.args.test_batch_size

        if self.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset)
            batch_size = self.args.test_batch_size // self.world_size

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                       batch_size=batch_size,
                                       num_workers=self.args.num_workers,
                                       drop_last=False,
                                       shuffle=is_shuffle,
                                       pin_memory=True,
                                       sampler=test_sampler)

        return test_loader, test_sampler

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return all_reduce_tensor(tensor, world_size=self.world_size, norm=norm)
        else:
            return tensor
            # return torch.mean(tensor)

    def reduce_tensor(self, tensor, dst=0):
        if self.distributed:
            return reduce_tensor(tensor, dst=dst, world_size=self.world_size)
        else:
            return torch.mean(tensor)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False

    