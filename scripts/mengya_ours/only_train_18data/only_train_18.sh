#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=0,1,2
NB_GPU=3

DATA_ROOT="/home/ren2/data2/mengya/mengya_dataset/EndoVis/"

DATASET=endo_instru
TASK=17-18 # 18-17
NAME=FT
METHOD=FT


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
SAVED=./checkpoints/step_only_train_18/deeplab_on_18

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --batch_size 32 --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.01 --epochs 100 --method ${METHOD} --file_name_list train_18.txt test_18.txt train_17.tx test_17.txt
# train_17.txt', 'test_17.txt', 'train_18.txt', 'test_17_18.txt

python3 average_csv.py ${RESULTSFILE}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"

# INFO:rank0: *** End of Test, Total Loss=0.17127007246017456, Class Loss=0.17127007246017456, Reg Loss=0.0
# INFO:rank0: 
# Total samples: 597.000000
# Overall Acc: 0.955443
# Mean Acc: 0.573658
# FreqW Acc: 0.922830
# Mean IoU: 0.472780
# Class IoU:
#         class 0: 0.9653201546706753
#         class 1: 0.6200298410753351
#         class 2: 0.20538262340065339
#         class 3: 0.15745326658499545
#         class 4: X
#         class 5: X
#         class 6: 0.8884962810082592
#         class 7: 0.0
# Class Acc:
#         class 0: 0.9871910029577796
#         class 1: 0.7783956347923315
#         class 2: 0.5707833813676515
#         class 3: 0.1592131437409649
#         class 4: X
#         class 5: X
#         class 6: 0.9463670399338479
#         class 7: 0.0

# INFO:rank0: Closing the Logger.
# End of step !!!!!!!!!!!!!!!!!!!!
# best_epoch 91
# best_iou 0.47003382252389986
# Last Step: 0
# Final Mean IoU 47.28
# Average Mean IoU 47.28
# Mean IoU first 0.0
# Mean IoU last 0.0
# endo_instru_17-18_FT On GPUs 0,1,2
# Run in 3519s






