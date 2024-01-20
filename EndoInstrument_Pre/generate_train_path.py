
import numpy as np
import os
import random

root = '/home/ren2/data2/mengya/mengya_dataset/EndoVis/'

# cropped_train in Endo17 will be splited into train/val
Endo17_directory = '2017_RoboticInstrumentSegmentation/cropped_train_test_no_1_2/'


Endo18_directory ='2018_RoboticSceneSegmentation/ISINet_Train_Val/'


# train = True # Then please switch it to False
# if (os.path.exists('train.txt')):
#     os.remove('train.txt')

train = False
# if (os.path.exists('test.txt')):
#     os.remove('test.txt')
# =================================== #


Dataset_name = 'Endo17'
Endo17_dir = os.path.join(root, Endo17_directory)

print('train', train)
if train:
    split_list = [1,3,4,6,7,8] 
else:
    split_list = [2,5,9,10]

num = 0
for folder_id in split_list:
    annotation_folder = os.path.join(Endo17_dir, 'instrument_dataset_' + str(folder_id), 'instruments_masks') 
    print('annotation_folder', annotation_folder)
    image_folder = os.path.join(Endo17_dir, 'instrument_dataset_' + str(folder_id), 'images')
    fnames = sorted(os.listdir(annotation_folder))
    random.shuffle(fnames)
    print('Number of samples', len(fnames))
    num += len(fnames)
    if train:
        f=open("train_17.txt","a")
        for line in fnames:
            # f.write(line+'\n') # 一列
            # f.write(Dataset_name+'\t'+line+'\n') # 两列

            path = os.path.join('instrument_dataset_' + str(folder_id), line)
            f.write(Dataset_name+'\t'+path+'\n')
        f.close()
    else:
        # f=open("test_17.txt","a")
        f=open("test_17_18.txt","a") # You need to switch between test_17.txt and test_17_18.txt
        for line in fnames:
            # f.write(line+'\n') # 一列
            # f.write(Dataset_name+'\t'+line+'\n')# 两列

            path = os.path.join('instrument_dataset_' + str(folder_id), line)
            f.write(Dataset_name+'\t'+path+'\n') 
            
        f.close()


print("Size in Endo17:", num)

#=========== the following part should be disable when we get the txt file for just 2017 dataset ============== #

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
random.shuffle(fnames)
# print('fnames', fnames)
print("Size in Endo18:", len(fnames))
if train:
    f=open("train_18.txt","a")
    for line in fnames:
        # f.write(line+'\n') # 一列
        f.write(Dataset_name+'\t'+line+'\n') # 两列
    f.close()
else:
    # f=open("test_18.txt","a")
    f=open("test_17_18.txt","a")
    for line in fnames:
        # f.write(line+'\n') # 一列
        f.write(Dataset_name+'\t'+line+'\n') # 两列
    f.close()


