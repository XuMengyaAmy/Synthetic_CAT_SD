
import numpy as np
import os 
import random



import os
import random
import numpy as np
import torch
def seed_everything(seed=42):
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
seed_everything()

Dataset_name = 'Harmonized_Blended_GAN_Endo17_addGraspingRetractor'
train = True # Then please switch it to False

# blended_root = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended'
blended_root = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended_grasping_tractor'

blended_dir = os.path.join(blended_root, 'harmonized_blended_2_50')

annotation_folder = os.path.join(blended_dir, 'annotations', 'multi')
image_folder = os.path.join(blended_dir, 'images')

fnames = sorted(os.listdir(image_folder))
random.shuffle(fnames)
N = 500   # GAN-based memory size (can try 300, 500, )
fnames = random.sample(fnames, N)

print("Size in GAN_17_memory:", len(fnames))
if train:
    f=open("train_harmonized_blended_18_FTplusGANimages.txt","a")
    for line in fnames:
        f.write(Dataset_name+'\t'+line+'\n') # 两列
    f.close()

