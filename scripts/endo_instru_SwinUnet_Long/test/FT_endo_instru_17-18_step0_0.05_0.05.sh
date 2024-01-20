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
MODEL=SwinUnet

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

# If you already trained the model for the first step, you can re-use those weights
# in order to skip this initial step --> faster iteration on your model
# Set this variable with the weights path
# FIRSTMODEL=/path/to/my/first/weights
# Then, for the first step, append those options:
# --ckpt ${FIRSTMODEL} --test
# And for the second step, this option:
# --step_ckpt ${FIRSTMODEL}

# python -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name FT --task 17-18 --step 0 --lr 0.01 --epochs 100 --method FT

# Use different lr and seed to make the class iou of unique instruments are not 0
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --batch_size 32 --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.05 --epochs 100 --method ${METHOD} --model ${MODEL} --checkpoint=./checkpoints_Long_SwinUnet/step_0.05_0.05 --test
# CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --batch_size 32 --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr 0.05 --epochs 100 --method ${METHOD} --model ${MODEL} --checkpoint=./checkpoints_Long_SwinUnet/step_0.05_0.05

python3 average_csv.py ${RESULTSFILE}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"

# CUDA_VISIBLE_DEVICES=0,1,2 python3 -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name FT --task 17-18 --step 1 --lr 0.01 --epochs 100 --method FT --step_ckpt /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/checkpoints_DeepLab/step_0_lr0.01_1_lr0.01/17-18-endo_instru_FT_0_best.pth --ckpt /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/checkpoints_DeepLab/step_0_lr0.01_1_lr0.01/17-18-endo_instru_FT_1_best.pth --test
# CUDA_VISIBLE_DEVICES=0,1,2 python3 -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name FT --task 17-18 --step 0 --lr 0.01 --epochs 100 --method FT --ckpt /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/checkpoints_DeepLab/step_0_lr0.01_1_lr0.01/17-18-endo_instru_FT_0_best.pth --test

# For the Swin-Unet model:
# CUDA_VISIBLE_DEVICES=0,1,2 python3 -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name FT --task 17-18 --step 0 --lr 0.01 --epochs 100 --method FT --ckpt /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/checkpoints_SwinUnet/step/17-18-endo_instru_FT_0_best.pth --test --model SwinUnet