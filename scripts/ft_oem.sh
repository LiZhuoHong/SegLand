#!/bin/bash
uname -a
#date
#env
date

DATASET=oem
DATA_PATH=YOUR_PATH_FOR_OEM_TRAIN_DATA
TRAIN_LIST=YOUR_PROJECT_ROOT/dataset/list/oem_ft/train.txt
VAL_LIST=YOUR_PROJECT_ROOT/dataset/list/oem_ft/val.txt
FOLD=0
SHOT=5
MODEL=swin_pop
BACKBONE=swin-s
RESTORE_PATH=YOUR_PROJECT_ROOT/model_saved/epoch_50.pth
LR=1e-4
WD=1e-4
BS=1
BS_TEST=1
START=0
STEPS=500
BASE_SIZE=1024,1024
INPUT_SIZE=1024,1024
OS=8
SEED=123
SAVE_DIR=YOUR_PROJECT_ROOT/model_saved_ft

cd YOUR_PROJECT_ROOT
python ft_pop.py --dataset ${DATASET} --data-dir ${DATA_PATH} \
			--train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --random-seed ${SEED}\
			--model ${MODEL} --backbone ${BACKBONE} --restore-from ${RESTORE_PATH} \
			--input-size ${INPUT_SIZE} --base-size ${BASE_SIZE} \
			--learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --test-batch-size ${BS_TEST}\
			--start-epoch ${START} --num-epoch ${STEPS}\
			--os ${OS} --snapshot-dir ${SAVE_DIR} --save-pred-every 50\
			--fold ${FOLD} --shot ${SHOT} --freeze-backbone --fix-lr --update-base --update-epoch 1
