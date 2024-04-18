#!/bin/bash
uname -a
#date
#env
date

DATASET=oem
DATA_PATH=YOUR_PATH_FOR_OEM_TEST_DATA
TRAIN_LIST=YOUR_PROJECT_ROOT/dataset/list/oem/train.txt
VAL_LIST=YOUR_PROJECT_ROOT/dataset/list/oem/test.txt
FOLD=0
SHOT=5
MODEL=swin_pop
BACKBONE=swin-s
RESTORE_PATH=YOUR_PROJECT_ROOT/model_saved/epoch_50.pth
BS=1
BASE_SIZE=1024,1024
OS=8
SAVE=0
SAVE_DIR=YOUR_PROJECT_ROOT/output
SEED=123

cd YOUR_PROJECT_ROOT
python eval_base.py  --dataset ${DATASET} --data-dir ${DATA_PATH} \
                --train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --test-batch-size ${BS} \
                --model ${MODEL} --restore-from ${RESTORE_PATH} --backbone ${BACKBONE} \
                --base-size ${BASE_SIZE} --save-path ${SAVE_DIR} --save ${SAVE}\
                --fold ${FOLD} --shot ${SHOT} --os ${OS} --random-seed ${SEED}