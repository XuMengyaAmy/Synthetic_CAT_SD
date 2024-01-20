
import numpy as np
import os
import random

# # Noise
# d['Gaussian Noise'] = gaussian_noise
# d['Shot Noise'] = shot_noise
# d['Impulse Noise'] = impulse_noise
# d['Speckle Noise'] = speckle_noise


# # Blur
# d['Defocus Blur'] = defocus_blur
# d['Glass Blur'] = glass_blur
# d['Motion Blur'] = motion_blur
# d['Zoom Blur'] = zoom_blur
# d['Gaussian Blur'] = gaussian_blur

# # Weather
# d['Snow'] = snow
# d['Frost'] = frost
# d['Fog'] = fog #error
# d['Brightness'] = brightness


# # Digital
# d['Contrast'] = contrast
# d['Elastic'] = elastic_transform
# d['Pixelate'] = pixelate
# d['JPEG'] = jpeg_compression

# # Others
# d['Spatter'] = spatter
# d['Saturate'] = saturate
# d['Gamma Correction'] = gamma_correction




root = '/home/ren2/data2/mengya/mengya_dataset/EndoVis/'

# cropped_train in Endo17 will be splited into train/val
Endo17_directory = '2017_RoboticInstrumentSegmentation/cropped_train_test_no_1_2/'

Endo18_directory ='2018_RoboticSceneSegmentation/ISINet_Train_Val/'

train = False
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
    # image_folder = os.path.join(Endo17_dir, 'instrument_dataset_' + str(folder_id), 'images')

    # ====robustness related code ========#
    image_folder = os.path.join(Endo17_dir, 'instrument_dataset_' + str(folder_id), 'images_c')
    test_17_corruption_dir = os.listdir(image_folder)
    print('test_17_corruption_dir', test_17_corruption_dir)
    print('number of corruption type', len(test_17_corruption_dir)) # number of corruption type 20
    severity_level = ['1', '2', '3', '4', '5']
    
    
    for c in test_17_corruption_dir:
        for d in severity_level:
            inter_dir = os.path.join(c, d)
            full_dir = os.path.join(image_folder, inter_dir)
            print('full_dir', full_dir)
            fnames = sorted(os.listdir(full_dir))
            random.shuffle(fnames)
            
            file_name = str('test_17_18_robustness'+'_'+ c +'.txt')
            f=open(file_name,"a")
            for line in fnames:
                path = os.path.join('instrument_dataset_' + str(folder_id), 'images_c')
                f.write(Dataset_name+'_'+'robustness'+'\t'+path+'/'+inter_dir+'/'+line+'\n') # 两列
            f.close()

#=====================================================================#
Dataset_name = 'Endo18'
Endo18_dir = os.path.join(root, Endo18_directory)
if train:
    split = 'train'
else:
    split = 'val'

annotation_folder = os.path.join(Endo18_dir, split, 'annotations', 'instrument')
# image_folder = os.path.join(Endo18_dir, split, 'images')

# === robustness related code ===== #
image_folder = os.path.join(Endo18_dir, split, 'images_c') 
test_18_corruption_dir = os.listdir(image_folder)
print('test_18_corruption_dir', test_18_corruption_dir)
print('number of corruption type', len(test_18_corruption_dir)) # number of corruption type 20
severity_level = ['1', '2', '3', '4', '5']


for a in test_18_corruption_dir:
    for b in severity_level:
        inter_dir = os.path.join(a, b)
        full_dir = os.path.join(image_folder, inter_dir)
        print('full_dir', full_dir)
        fnames = sorted(os.listdir(full_dir))
        random.shuffle(fnames)
        
        file_name = str('test_17_18_robustness'+'_'+ a +'.txt')
        f=open(file_name,"a")
        for line in fnames:
            f.write(Dataset_name+'_'+'robustness'+'\t'+inter_dir+'/'+line+'\n') # 两列
        f.close()

        
        
        




