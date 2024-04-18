import os
import os.path as osp
import numpy as np
import random
from collections import defaultdict


def gen_list(dataset,fold,shot,seed):
    random.seed(seed)
    if dataset == 'voc':
        list_path = 'dataset/list/voc/trainaug.txt'
        num_classes = 20
        interval = num_classes // 4
        # base classes = all classes - novel classes
        base_classes = set(range(1, num_classes + 1)) - set(range(interval * fold + 1, interval * (fold + 1) + 1))
        # novel classes
        novel_classes = set(range(interval * fold + 1, interval * (fold + 1) + 1))
    else:
        list_path = 'dataset/list/coco/train.txt'
        num_classes = 80
        base_classes = set(range(1, num_classes + 1)) - set(range(fold + 1, fold + 78, 4))
        # novel classes
        novel_classes = set(range(fold + 1, fold + 78, 4))

    list_dir = os.path.dirname(list_path)
    list_dir = list_dir + '/fold%s'%fold

    novel_cls_to_ids = defaultdict(list)
    for cls in novel_classes:
        with open(os.path.join(list_dir, 'train_novel_class%s.txt'%cls), 'r') as f:
            novel_cls_to_ids[cls] = f.read().splitlines()

    novel_list = list(novel_classes)
    novel_id_list = []
    id_s_list = []
    for target_cls in novel_list:
        # id_s_list = []
        file_class_chosen = novel_cls_to_ids[target_cls]
        num_file = len(file_class_chosen)
        if num_file < shot:
            print('extend images with repeating')
            for i in range(num_file):
                novel_id_list.append(file_class_chosen[i])
            for i in range(shot-num_file):
                novel_id_list.append(file_class_chosen[random.randint(1, num_file) - 1])
            # num_file = shot
        else:
            for k in range(shot):
                id_s = ' '
                while((id_s == ' ') or id_s in id_s_list):
                    support_idx = random.randint(1, num_file) - 1
                    id_s = file_class_chosen[support_idx]
                id_s_list.append(id_s)
                novel_id_list.append(id_s)
    print(len(novel_id_list))

    with open(os.path.join(list_dir, 'fold%s_%sshot_seed%s.txt'%(fold, shot, seed)), 'w') as f:
        for id in novel_id_list:
            f.write(id+"\n")

if __name__ == '__main__':
    dataset = 'coco'
    folds = [0,1,2,3]
    shots = [1,5,10]
    seeds = [123]
    for fold in folds:
        for shot in shots:
            for seed in seeds:
                gen_list(dataset,fold,shot,seed)

