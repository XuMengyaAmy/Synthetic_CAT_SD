python -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name FT --task 17-18 --step 0 --lr 0.01 --epochs 100 --method FT
python -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name FT --task 17-18 --step 1 --lr 0.001 --epochs 100 --method FT
python -m torch.distributed.launch --nproc_per_node=3 --master_port 29501 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name FT --task 17-18 --step 1 --lr 0.01 --epochs 60 --method FT

python -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name LWF --task 17-18 --step 0 --lr 0.01 --epochs 100 --method LWF
python -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name LWF --task 17-18 --step 1 --lr 0.001 --epochs 100 --method LWF


python -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name EWC --task 17-18 --step 0 --lr 0.01 --epochs 100 --method EWC
python -m torch.distributed.launch --nproc_per_node=3 run.py --data_root /home/ren2/data2/mengya/mengya_dataset/EndoVis/ --batch_size 32 --dataset endo_instru --name EWC --task 17-18 --step 1 --lr 0.001 --epochs 100 --method EWC