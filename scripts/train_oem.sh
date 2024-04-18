#!/bin/bash
uname -a
#date
#env
date

DATASET=oem
DATA_PATH=YOUR_PATH_FOR_OEM_TRAIN_DATA
TRAIN_LIST=YOUR_PROJECT_ROOT/dataset/list/oem/train.txt
VAL_LIST=YOUR_PROJECT_ROOT/dataset/list/oem/val.txt
FOLD=0
MODEL=seghr_pop
BACKBONE=hr-w32
RESTORE_PATH=YOUR_PROJECT_ROOT/pretrained_backbones/hrnetv2_w32_imagenet_pretrained.pth
LR=1e-3
WD=1e-4
BS=4
BS_TEST=1
START=0
STEPS=200
BASE_SIZE=1024,1024
INPUT_SIZE=768,768
OS=8
SEED=123
SAVE_DIR=YOUR_PROJECT_ROOT/model_saved

cd YOUR_PROJECT_ROOT
python train_base.py --dataset ${DATASET} --data-dir ${DATA_PATH} \
			--train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --random-seed ${SEED}\
			--model ${MODEL} --backbone ${BACKBONE} --restore-from ${RESTORE_PATH} \
			--input-size ${INPUT_SIZE} --base-size ${BASE_SIZE} \
			--learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --test-batch-size ${BS_TEST}\
			--start-epoch ${START} --num-epoch ${STEPS}\
			--os ${OS} --snapshot-dir ${SAVE_DIR} --save-pred-every 50\
			--fold ${FOLD}