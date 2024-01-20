
import numpy as np
import os 
import random



import os
import random
import numpy as np
import torch
from PIL import Image

def seed_everything(seed=42):
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
seed_everything()

Dataset_name = 'Harmonized_Blended_GAN_Endo17'
train = True # Then please switch it to False

blended_root = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended'
blended_dir = os.path.join(blended_root, 'harmonized_blended_2_50')

annotation_folder = os.path.join(blended_dir, 'annotations', 'multi')
image_folder = os.path.join(blended_dir, 'images')

fnames = sorted(os.listdir(image_folder))
random.shuffle(fnames)
N = 500   # GAN-based memory size (can try 300, 500, )
fnames = random.sample(fnames, N) # fnames is a list

print("Size in GAN_17_memory:", len(fnames))


# Count class frequency
Bipolar_Forceps_sum1 = 0
Prograsp_Forceps_sum2 = 0
Large_Needle_Driver_sum3 = 0
Vessel_Sealer_sum4 = 0
Grasping_Retractor_sum5 = 0
Monopolar_Curved_Scissors_sum6 = 0
Ultrasound_Probe_sum7 = 0


mask1 = 32
mask2 = 64
mask3 = 96
mask4 = 128
mask5 = 160
mask6 = 192
mask7 = 224

mask_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended/blended_2_50'
for fname in fnames:
    image_path = fname
    annotation = os.path.join(mask_dir, 'annotations', 'multi', image_path)
    mask = Image.open(annotation)
    mask_np = np.array(mask)
    print('unqiue value', np.unique(mask_np))
    if mask1 in np.unique(mask_np):
        Bipolar_Forceps_sum1 += 1

    if mask2 in np.unique(mask_np):
        Prograsp_Forceps_sum2 += 1

    if mask3 in np.unique(mask_np):
        Large_Needle_Driver_sum3 += 1

    if mask4 in np.unique(mask_np):
        Vessel_Sealer_sum4 += 1

    if mask5 in np.unique(mask_np):
        Grasping_Retractor_sum5 += 1

    if mask6 in np.unique(mask_np):
        Monopolar_Curved_Scissors_sum6 += 1

    if mask7 in np.unique(mask_np):
        Ultrasound_Probe_sum7 += 1

print('Bipolar_Forceps_sum1', Bipolar_Forceps_sum1)
print('Prograsp_Forceps_sum2', Prograsp_Forceps_sum2)
print('Large_Needle_Driver_sum3', Large_Needle_Driver_sum3)
print('Vessel_Sealer_sum4', Vessel_Sealer_sum4)
print('Grasping_Retractor_sum5', Grasping_Retractor_sum5)
print('Monopolar_Curved_Scissors_sum6', Monopolar_Curved_Scissors_sum6)
print('Ultrasound_Probe_sum7', Ultrasound_Probe_sum7)
    

# Bipolar_Forceps_sum1 174
# Prograsp_Forceps_sum2 128
# Large_Needle_Driver_sum3 96
# Vessel_Sealer_sum4 127
# Grasping_Retractor_sum5 113
# Monopolar_Curved_Scissors_sum6 88
# Ultrasound_Probe_sum7 125
    

# if train:
#     f=open("train_harmonized_blended_18_GAN_17_memory500.txt","a")
#     for line in fnames:
#         f.write(Dataset_name+'\t'+line+'\n') # 两列
#     f.close()

