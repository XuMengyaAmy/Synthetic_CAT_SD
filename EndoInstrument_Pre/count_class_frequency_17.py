
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

# Dataset_name = 'Harmonized_Blended_GAN_Endo17'
# train = True # Then please switch it to False
train = False

# Dataset_name = 'Endo17'
# Endo17_dir = os.path.join(root, Endo17_directory)

# print('train', train)
# if train:
#     split_list = [1,3,4,6,7,8] 
# else:
#     split_list = [2,5,9,10]


dataset_dir = '/home/ren2/data2/mengya/mengya_dataset/EndoVis/2017_RoboticInstrumentSegmentation/cropped_train_test_no_1_2/'

if train:
  print('train set')
  image_folder = ['instrument_dataset_1', 'instrument_dataset_3', 'instrument_dataset_4', 'instrument_dataset_6', 'instrument_dataset_7', 'instrument_dataset_8']
else:
  print('val set')
  image_folder = ['instrument_dataset_2', 'instrument_dataset_5', 'instrument_dataset_9', 'instrument_dataset_10']


mask_dir = []

for i in image_folder:
    image_fodler_dir = os.path.join(dataset_dir, i, 'instruments_masks')
    # print('image_fodler_dir', image_fodler_dir)
    mask_dir.append(image_fodler_dir)

# print('mask_dir', mask_dir)


frame_path_all = []
for m in mask_dir:
    # print('m', m)
    fnames = sorted(os.listdir(m))
    # print('fnames', fnames)
    for n in fnames:
        frame_path = os.path.join(m, n)
        frame_path_all.append(frame_path)

print('frames_path_all', frame_path_all)
print('Number of frames', len(frame_path_all))  # Number of frames 2250

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

# mask_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended/blended_2_50'
for fname_path in frame_path_all:
    # image_path = fname
    # annotation = os.path.join(mask_dir, 'annotations', 'multi', image_path)
    print('fname', fname_path)
    mask = Image.open(fname_path)
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

    

######### 17 train
# Bipolar_Forceps_sum1 871
# Prograsp_Forceps_sum2 887
# Large_Needle_Driver_sum3 897
# Vessel_Sealer_sum4 500
# Grasping_Retractor_sum5 255
# Monopolar_Curved_Scissors_sum6 430
# Ultrasound_Probe_sum7 452


########### 17 val
# Bipolar_Forceps_sum1 503
# Prograsp_Forceps_sum2 479
# Large_Needle_Driver_sum3 300
# Vessel_Sealer_sum4 271
# Grasping_Retractor_sum5 54
# Monopolar_Curved_Scissors_sum6 333
# Ultrasound_Probe_sum7 227