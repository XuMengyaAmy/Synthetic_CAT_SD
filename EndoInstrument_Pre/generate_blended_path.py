
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

Dataset_name = 'Blended_Endo18'
train = True # Then please switch it to False
if (os.path.exists('train_blended_18.txt')):
    os.remove('train_blended_18.txt')

if (os.path.exists('train_harmonized_blended_18.txt')):
    os.remove('train_harmonized_blended_18.txt')

blended_root = '/home/ren2/data2/mengya/mengya_code/Synthesis_18/Synthesis/data_gen/blended'
blended_dir = os.path.join(blended_root, 'blended_2_50')

annotation_folder = os.path.join(blended_dir, 'annotations', 'multi')
image_folder = os.path.join(blended_dir, 'images')

fnames = sorted(os.listdir(image_folder))
random.shuffle(fnames)
N = 1639
fnames = random.sample(fnames, N)

print("Size in Blended_Endo18:", len(fnames))
if train:
    f=open("train_blended_18.txt","w")
    for line in fnames:
        f.write(Dataset_name+'\t'+line+'\n') # 两列
    f.close()

Dataset_name_2 = 'Harmonized_Blended_Endo18'
# harmonized_blended_dir = os.path.join(blended_root, 'harmonized_blended_2_50')
# harmonized_image_folder = os.path.join(blended_dir, 'images')
# fnames = sorted(os.listdir(harmonized_image_folder))
# random.shuffle(fnames)

print("Size in Harmonized_Blended_Endo18:", len(fnames))
if train:
    f=open("train_harmonized_blended_18.txt","w")
    for line in fnames:
        f.write(Dataset_name_2+'\t'+line+'\n') # 两列
    f.close()
