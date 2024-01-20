
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


blended_root = '/home/ren2/data2/mengya/mengya_code/Synthesis_18/Synthesis/data_gen/blended'
blended_dir = os.path.join(blended_root, 'blended_2_50')

annotation_folder = os.path.join(blended_dir, 'annotations', 'multi')
image_folder = os.path.join(blended_dir, 'images')

fnames = sorted(os.listdir(image_folder))
random.shuffle(fnames)
N = 1639
fnames = random.sample(fnames, N)

# Count class frequency
Bipolar_Forceps_sum1 = 0
Prograsp_Forceps_sum2 = 0
Large_Needle_Driver_sum3 = 0
Monopolar_Curved_Scissors_sum4 = 0
Ultrasound_Probe_sum5 = 0
Suction_sum6 = 0
Clip_Applier_sum7 = 0


# "Bipolar_Forceps": 1 
# "Prograsp_Forceps": 2
# "Large_Needle_Driver": 3 
# "Monopolar_Curved_Scissors": 4 
# "Ultrasound_Probe": 5 
# "Suction_Instrument": 6  # unique
# "Clip_Applier": 7 # unique


mask1 = 1
mask2 = 2
mask3 = 3
mask4 = 4
mask5 = 5
mask6 = 6
mask7 = 7

mask_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_18/Synthesis/data_gen/blended/blended_2_50'
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
        Monopolar_Curved_Scissors_sum4 += 1

    if mask5 in np.unique(mask_np):
        Ultrasound_Probe_sum5 += 1

    if mask6 in np.unique(mask_np):
        Suction_sum6 += 1

    if mask7 in np.unique(mask_np):
       Clip_Applier_sum7 += 1

print('Bipolar_Forceps_sum1', Bipolar_Forceps_sum1)
print('Prograsp_Forceps_sum2', Prograsp_Forceps_sum2)
print('Large_Needle_Driver_sum3', Large_Needle_Driver_sum3)
print('Monopolar_Curved_Scissors_sum4', Monopolar_Curved_Scissors_sum4)
print('Ultrasound_Probe_sum5', Ultrasound_Probe_sum5)
print('Suction_sum6', Suction_sum6)
print('Clip_Applier_sum7', Clip_Applier_sum7)



# Bipolar_Forceps_sum1 519
# Prograsp_Forceps_sum2 391
# Large_Needle_Driver_sum3 353
# Monopolar_Curved_Scissors_sum4 405
# Ultrasound_Probe_sum5 351
# Suction_sum6 337
# Clip_Applier_sum7 397
