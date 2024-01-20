
import numpy as np
import os 

root = '/home/ren2/data2/mengya/mengya_dataset/EndoVis/'

# cropped_train in Endo17 will be splited into train/val
Endo17_directory = '2017_RoboticInstrumentSegmentation/cropped_train/'
Endo18_directory ='2018_RoboticSceneSegmentation/ISINet_Train_Val/'




# train = True # Then please switch it to False
# if (os.path.exists('train.txt')):
#     os.remove('train.txt')

train = False
if (os.path.exists('test.txt')):
    os.remove('test.txt')
# =================================== #
Dataset_name = 'Endo17'
Endo17_dir = os.path.join(root, Endo17_directory)

if train:
    split_list = ['instrument_dataset_1', 'instrument_dataset_2', 'instrument_dataset_3', 'instrument_dataset_4','instrument_dataset_5', 'instrument_dataset_6']
else:
    split_list = ['instrument_dataset_7', 'instrument_dataset_8']

num = 0
for folder in split_list:
    annotation_folder = os.path.join(Endo17_dir, folder, 'instruments_masks')
    print('annotation_folder', annotation_folder)
    image_folder = os.path.join(Endo17_dir, folder, 'images')
    fnames = sorted(os.listdir(annotation_folder))
    print('Number of samples', len(fnames))
    num += len(fnames)
    if train:
        f=open("train.txt","a")
        for line in fnames:
            # f.write(line+'\n') # 一列
            # f.write(Dataset_name+'\t'+line+'\n') # 两列

            path = os.path.join(folder, line)
            f.write(Dataset_name+'\t'+path+'\n')
        f.close()
    else:
        f=open("test.txt","a")
        for line in fnames:
            # f.write(line+'\n') # 一列
            # f.write(Dataset_name+'\t'+line+'\n')# 两列

            path = os.path.join(folder, line)
            f.write(Dataset_name+'\t'+path+'\n') 
            
        f.close()


print("Size in Endo17:", num)

#====================================== #

Dataset_name = 'Endo18'
Endo18_dir = os.path.join(root, Endo18_directory)
if train:
    split = 'train'
else:
    split = 'val'

annotation_folder = os.path.join(Endo18_dir, split, 'annotations', 'instrument')
image_folder = os.path.join(Endo18_dir, split, 'images')

# because in Endo18 dataset, 2 images don't have the annotation masks. 
# num_frames should be based on annotation masks
fnames = sorted(os.listdir(annotation_folder))
# print('fnames', fnames)
print("Size in Endo18:", len(fnames))
if train:
    # f=open("train.txt","w")
    f=open("train.txt","a")
    for line in fnames:
        # f.write(line+'\n') # 一列
        f.write(Dataset_name+'\t'+line+'\n') # 两列
    f.close()
else:
    f=open("test.txt","a")
    for line in fnames:
        # f.write(line+'\n') # 一列
        f.write(Dataset_name+'\t'+line+'\n') # 两列
    f.close()


# (CSS) ren2@Ren2:~/data2/mengya/mengya_code/CVPR2021_PLOP/EndoInstrument_Pre$ python generate_train_path.py 
# annotation_folder /home/ren2/data2/mengya/mengya_dataset/EndoVis/2017_RoboticInstrumentSegmentation/cropped_train/instrument_dataset_1/instruments_masks
# Number of samples 225
# annotation_folder /home/ren2/data2/mengya/mengya_dataset/EndoVis/2017_RoboticInstrumentSegmentation/cropped_train/instrument_dataset_2/instruments_masks
# Number of samples 225
# annotation_folder /home/ren2/data2/mengya/mengya_dataset/EndoVis/2017_RoboticInstrumentSegmentation/cropped_train/instrument_dataset_3/instruments_masks
# Number of samples 225
# annotation_folder /home/ren2/data2/mengya/mengya_dataset/EndoVis/2017_RoboticInstrumentSegmentation/cropped_train/instrument_dataset_4/instruments_masks
# Number of samples 225
# Size in Endo17: 900
# Size in Endo18: 1639
# (CSS) ren2@Ren2:~/data2/mengya/mengya_code/CVPR2021_PLOP/EndoInstrument_Pre$ python generate_train_path.py 
# annotation_folder /home/ren2/data2/mengya/mengya_dataset/EndoVis/2017_RoboticInstrumentSegmentation/cropped_train/instrument_dataset_5/instruments_masks
# Number of samples 225
# annotation_folder /home/ren2/data2/mengya/mengya_dataset/EndoVis/2017_RoboticInstrumentSegmentation/cropped_train/instrument_dataset_6/instruments_masks
# Number of samples 225
# annotation_folder /home/ren2/data2/mengya/mengya_dataset/EndoVis/2017_RoboticInstrumentSegmentation/cropped_train/instrument_dataset_7/instruments_masks
# Number of samples 225
# annotation_folder /home/ren2/data2/mengya/mengya_dataset/EndoVis/2017_RoboticInstrumentSegmentation/cropped_train/instrument_dataset_8/instruments_masks
# Number of samples 225
# Size in Endo17: 900
# Size in Endo18: 596
