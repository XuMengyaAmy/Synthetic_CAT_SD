#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=0,1,2
NB_GPU=3

DATA_ROOT="/home/ren2/data2/mengya/mengya_dataset/EndoVis/"

DATASET=endo_instru
TASK=17-18
NAME=FT
METHOD=FT
MODEL=UNet11

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

# If you already trained the model for the first step, you can re-use those weights
# in order to skip this initial step --> faster iteration on your model
# Set this variable with the weights path
# FIRSTMODEL=/path/to/my/first/weights
FIRSTMODEL=./existing_model_EndoVis17_checkpoints/models/unet11_instruments_20/model_1.pt
# Then, for the first step, append those options:
# --ckpt ${FIRSTMODEL} --test
# And for the second step, this option:
# --step_ckpt ${FIRSTMODEL}

CHECKPOINT=./checkpoints_UNet11/step

# python -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name FT --task 17-18 --step 0 --lr 0.01 --epochs 100 --method FT

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --batch_size 32 --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.01 --epochs 100 --method ${METHOD} --ckpt ${FIRSTMODEL} --test --model ${MODEL} --crop_val --file_name_list train_17.txt test_17_18.txt train_18.txt test_17_18.txt
# CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --batch_size 32 --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr 0.001 --epochs 100 --method ${METHOD} --step_ckpt ${FIRSTMODEL} --model ${MODEL} --checkpoint ${CHECKPOINT} --test --crop_val --file_name_list train_17.txt test_17_18.txt train_18.txt test_17_18.txt


python3 average_csv.py ${RESULTSFILE}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"



# image size 也不一样
# RuntimeError: CUDA error: an illegal memory access was encountered
# CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
# 问题大多数是网络中的label和网络输出的维度大小不一样，