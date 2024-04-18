#!/bin/bash
uname -a
#date
#env
date

DATASET=oem
DATA_PATH=/home/lufangxiao/OEM_data/testset
TRAIN_LIST=/home/lufangxiao/POP-main/dataset/list/oem_ft/train.txt
VAL_LIST=/home/lufangxiao/POP-main/dataset/list/oem_ft/test.txt
FOLD=0
SHOT=5
MODEL=swin_pop
BACKBONE=swin-s
RESTORE_PATH=/home/lufangxiao/POP-main/model_saved_ft/epoch_499.pth
BS=1
BASE_SIZE=1024,1024
OS=8
SAVE=0
SAVE_DIR=/home/lufangxiao/POP-main/output
SEED=123

cd /home/lufangxiao/POP-main
python eval_ft.py  --dataset ${DATASET} --data-dir ${DATA_PATH} \
                --train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --test-batch-size ${BS} \
                --model ${MODEL} --restore-from ${RESTORE_PATH} --backbone ${BACKBONE} \
                --base-size ${BASE_SIZE} --save-path ${SAVE_DIR} --save ${SAVE}\
                --fold ${FOLD} --shot ${SHOT} --os ${OS} --random-seed ${SEED}