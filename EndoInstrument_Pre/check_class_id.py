from PIL import Image
# import cv2
import numpy as np

Endo17_path = '/home/ren2/data2/mengya/mengya_dataset/EndoVis/2017_RoboticInstrumentSegmentation/cropped_train/instrument_dataset_2/instruments_masks/frame000.png'
Endo18_path ='/home/ren2/data2/mengya/mengya_dataset/EndoVis/2018_RoboticSceneSegmentation/ISINet_Train_Val/train/annotations/instrument/seq_3_frame000.png'


Endo17_target = Image.open(Endo17_path).convert('L')
Endo17_target = np.array(Endo17_target)
print('Endo17_target', Endo17_target.shape) # Endo17_target (1024, 1280)
print('Endo17_target', np.unique(Endo17_target)) # Endo17_target [  0  32  64 224]

Endo18_target = Image.open(Endo18_path).convert('L')
Endo18_target = np.array(Endo18_target)
print('Endo18_target', Endo18_target.shape) # Endo18_target (1024, 1280)
print('Endo18_target', np.unique(Endo18_target)) # Endo17_target [0 1 2 4]

# Endo17_target (1024, 1280)
# Endo17_target [  0  64 224]
# Endo18_target (1024, 1280)
# Endo18_target [0 4]

ade_path = '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/data/ADEChallengeData2016/annotations/training/ADE_train_00000001.png'
ade_target = Image.open(ade_path).convert('L')
ade_target = np.array(ade_target)
print('ade_target', ade_target.shape)
print('ade_target', np.unique(ade_target))
# ade_target (512, 683)
# ade_target [  0   1   4   5   6  13  18  32  33  43  44  88  97 105 126 139 150]

Grasping_Retractor = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended_grasping_tractor/blended_2_50/annotations/multi/5-instrument_dataset_5_frame085_0-fakes002845_5.png'
Grasping_Retractor_target = Image.open(Grasping_Retractor).convert('L')
Grasping_Retractor_target = np.array(Grasping_Retractor_target)
print('Endo17_target', np.unique(Grasping_Retractor_target)) # Endo17_target [  0  32  64 224]