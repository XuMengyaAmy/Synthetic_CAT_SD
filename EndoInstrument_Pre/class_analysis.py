# ===== 2017 ======== #
#    Background: 0*32 = 0

# 	"Bipolar Forceps": 1*32 = 32

# 	"Prograsp Forceps": 2*32 = 64
	
# 	"Large Needle Driver": 3*32 = 96

# 	"Vessel Sealer": 4*32 = 128       # unique

# 	"Grasping Retractor": 5*32 = 160  # unique

#   "Monopolar Curved Scissors": 6*32 = 192

#   "Ultrasound_Probe": 7*32 = 224

# class_dic = 

import numpy as np
import os 

root = '/home/ren2/data2/mengya/mengya_dataset/EndoVis/'

# cropped_train in Endo17 will be splited into train/val
Endo17_directory = '2017_RoboticInstrumentSegmentation/cropped_train/'







Dataset_name = 'Endo17'
Endo17_dir = os.path.join(root, Endo17_directory)

print('train', train)
if train:
    split_list = ['instrument_dataset_1', 'instrument_dataset_2', 'instrument_dataset_3', 'instrument_dataset_4','instrument_dataset_5', 'instrument_dataset_6']
else:
    split_list = ['instrument_dataset_7', 'instrument_dataset_8']


for folder in split_list:
    annotation_folder = os.path.join(Endo17_dir, folder, 'instruments_masks')
    print('annotation_folder', annotation_folder)



