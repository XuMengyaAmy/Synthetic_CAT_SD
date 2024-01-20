import copy
import glob
import os

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from .utils import Subset, filter_images, group_images
import torch

# 6 common classes: 
# Background, Bipolar Forceps, Prograsp Forceps, Large Needle Driver, Monopolar Curved Scissors, Ultrasound Probe
# ‌
# 2 unique classes in 2017:
# Vessel Sealer, Grasping Retractor
# ‌
# 2 unique classes in 2018:
# Suction_Instrument, Clip_Applier

# ===== 2017 ======== #
#    Background: 0*32 = 0

# 	"Bipolar Forceps": 1*32 = 32

# 	"Prograsp Forceps": 2*32 = 64
	
# 	"Large Needle Driver": 3*32 = 96

# 	"Vessel Sealer": 4*32 = 128       # unique

# 	"Grasping Retractor": 5*32 = 160  # unique

#   "Monopolar Curved Scissors": 6*32 = 192

#   "Ultrasound_Probe": 7*32 = 224



# ===== 2018 ======== #	\
# Background: 0

# "Bipolar_Forceps": 1 

# "Prograsp_Forceps": 2

# "Large_Needle_Driver": 3 

# "Monopolar_Curved_Scissors": 4 

# "Ultrasound_Probe": 5 

# "Suction_Instrument": 6  # unique

# "Clip_Applier": 7 # unique

id_to_trainid = {
    0:    0,

    32:   1,
    64:   2,
    96:   3,

    128:  4,
    160:  5,

    192:  6,
    224:  7,


    1:    1,
    2:    2,
    3:    3,

    4:    6,
    5:    7,

    6:    8,
    7:    9
}

# instrument_to_id = {
#     "Background": 0, "BipolarForceps": 1, "PrograspForceps": 2, "LargeNeedleDriver": 3, 
#     "Vessel Sealer": 4, "Grasping Retractor": 5,
#     "MonopolarCurvedScissors": 6, "Ultrasound_Probe": 7,
#     "Suction_Instrument": 8, "Clip_Applier": 9
# }

# https://github.com/ternaus/robot-surgery-segmentation/blob/de12469172239a32c0dfe7c39832ac5e52ee2baf/dataset.py
class EndoInstrumentSegmentation(data.Dataset):

    def __init__(self, root, file_path, train=True, transform=None):

        root = root # '/home/ren2/data2/mengya/mengya_dataset/EndoVis/'
        Endo17_directory = '2017_RoboticInstrumentSegmentation/cropped_train_test_no_1_2/'

        image_jpg_folder = ['instrument_dataset_1', 'instrument_dataset_2', 'instrument_dataset_3', 'instrument_dataset_4',
        'instrument_dataset_5', 'instrument_dataset_6', 'instrument_dataset_7', 'instrument_dataset_8']
        # image_png_folder = ['instrument_dataset_9', 'instrument_dataset_10']

        Endo18_directory ='2018_RoboticSceneSegmentation/ISINet_Train_Val/'
        if train:
            # file_path = '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/EndoInstrument_Pre/train.txt'
            split = 'train'
        else:
            # file_path = '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/EndoInstrument_Pre/test.txt'
            split = 'val'

        fp=open(file_path)

        self.images = []
        for line in fp.readlines():
            line=line.strip('\n')
            Dataset_name = line.split("\t")[0]
            image_path = line.split("\t")[-1]
            
            if Dataset_name == 'Endo17':
                dataset_dir = os.path.join(root, Endo17_directory)
                folder = image_path.split('/')[0] # # instrument_dataset_1/frame008.png
                image_name = image_path.split('/')[-1]


                annotation = os.path.join(dataset_dir, folder, 'instruments_masks', image_name)
                
                if folder in image_jpg_folder:
                    image = os.path.join(dataset_dir, folder, 'images', image_name.split('.')[0]+'.jpg')
                else:
                    image = os.path.join(dataset_dir, folder, 'images', image_name)
                
            elif Dataset_name == 'Endo18':
                dataset_dir = os.path.join(root, Endo18_directory)
                
                annotation = os.path.join(dataset_dir, split, 'annotations', 'instrument', image_path)
                image = os.path.join(dataset_dir, split,'images', image_path)
            
            elif Dataset_name == 'Blended_Endo18':               
                dataset_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_18/Synthesis/data_gen/blended/blended_2_50'
                annotation = os.path.join(dataset_dir, 'annotations', 'multi', image_path)
                image = os.path.join(dataset_dir, 'images', image_path)
            
            elif Dataset_name == 'Harmonized_Blended_Endo18':
                #  Harmonized_Blended_Endo18 只有 images 会发生变化，但是 mask 不会变化          
                dataset_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_18/Synthesis/data_gen/blended/harmonized_blended_2_50'
                image = os.path.join(dataset_dir, 'images', image_path)
                
                mask_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_18/Synthesis/data_gen/blended/blended_2_50'
                annotation = os.path.join(mask_dir, 'annotations', 'multi', image_path)
        
            # =========== GAN 17 =============== #
            elif Dataset_name == 'Harmonized_Blended_GAN_Endo17': # add GAN-based memory
                dataset_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended/harmonized_blended_2_50'
                image = os.path.join(dataset_dir, 'images', image_path)

                mask_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended/blended_2_50'
                annotation = os.path.join(mask_dir, 'annotations', 'multi', image_path)

            elif Dataset_name == 'Harmonized_Blended_GAN_Endo17_addGraspingRetractor':
                dataset_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended_grasping_tractor/harmonized_blended_2_50'
                image = os.path.join(dataset_dir, 'images', image_path)

                mask_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended_grasping_tractor/blended_2_50'
                annotation = os.path.join(mask_dir, 'annotations', 'multi', image_path)

            elif Dataset_name == 'Harmonized_Blended_GAN_Endo17_addVesselSealer':
                dataset_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended_Vessel_Sealer/harmonized_blended_2_50'
                image = os.path.join(dataset_dir, 'images', image_path)

                mask_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended_Vessel_Sealer/blended_2_50'
                annotation = os.path.join(mask_dir, 'annotations', 'multi', image_path)

            elif Dataset_name == 'Harmonized_Blended_GAN_tmibg_Endo17':
                dataset_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended/harmonized_blended_2_50_tmi_bg'
                image = os.path.join(dataset_dir, 'images', image_path)

                mask_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended/blended_2_50_tmi_bg'
                annotation = os.path.join(mask_dir, 'annotations', 'multi', image_path)

            elif Dataset_name == 'Harmonized_Blended_DDPM_Endo17':
                dataset_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended/harmonized_blended_2_50_ddpm'
                image = os.path.join(dataset_dir, 'images', image_path)

                mask_dir = '/home/ren2/data2/mengya/mengya_code/Synthesis_17/Synthesis/data_gen/blended/blended_2_50_ddpm'
                annotation = os.path.join(mask_dir, 'annotations', 'multi', image_path)
            # ========================== #


            # ========== Robustness (corrupted test set) =========== #
            elif Dataset_name == 'Endo18_robustness':
                dataset_dir = os.path.join(root, Endo18_directory)
                image_name = image_path.split('/')[-1]
                image = os.path.join(dataset_dir, split,'images_c', image_path)
                annotation = os.path.join(dataset_dir, split, 'annotations', 'instrument', image_name)
            
            elif Dataset_name == 'Endo17_robustness':
                dataset_dir = os.path.join(root, Endo17_directory)
                image = os.path.join(dataset_dir, image_path)

                folder = image_path.split('/')[0] # # instrument_dataset_2/images_c/fog/1/frame084.jpg
                image_name = image_path.split('/')[-1]
                annotation_name = image_name.split('.')[0]+'.png'
                annotation = os.path.join(dataset_dir, folder, 'instruments_masks', annotation_name)
            # ====================================================== #

            image_annotation_pair = (image, annotation)
            self.images.append(image_annotation_pair)

        fp.close()
        
        self.transform = transform

        # print('self.images', self.images)

        # # ================ make the target id in Endo18 same as Endo17 ========== #
        # # https://www.cnblogs.com/wanghui-garcia/p/11248416.html
        # https://blog.csdn.net/qq_35037684/article/details/121638114
        # Lambda Transforms https://zhuanlan.zhihu.com/p/401135685
        self.target_transform = tv.transforms.Lambda(lambda t: t.
                                                    apply_(lambda x: id_to_trainid.get(x))
                                                    )
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])

        if self.transform is not None:
            img, target = self.transform(img, target)
        
        # print('img', img.shape) # torch.Size([3, 224, 224])
        # print('target', target.shape)
        # print('target before', torch.unique(target)) # target before tensor([  0,  32, 224], dtype=torch.uint8)
        target = self.target_transform(target)
        # print('target after', torch.unique(target)) # target after tensor([0, 1, 7], dtype=torch.uint8)
        

        return img, target

    def __len__(self):
        return len(self.images)

class EndoInstrumentSegmentationIncremental(data.Dataset):

    def __init__(
        self,
        root,
        step,
        train,
        transform,
        file_root = '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/EndoInstrument_Pre/',
        file_name_list = ['train_17.txt', 'test_17.txt', 'train_18.txt', 'test_17_18.txt'],
    ):
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # print('step in dataset', step)
        # if step == 0:
        #     if train:
        #         file_path = os.path.join(file_root, 'train_17.txt')
        #     else:
        #         file_path = os.path.join(file_root, 'test_17.txt')
        # elif step == 1:
        #     if train:
        #         file_path = os.path.join(file_root, 'train_18.txt')
        #     else:
        #         file_path = os.path.join(file_root, 'test_17_18.txt')

        if step == 0:
            if train:
                file_path = os.path.join(file_root, file_name_list[0])
            else:
                file_path = os.path.join(file_root, file_name_list[1])
        elif step == 1:
            if train:
                file_path = os.path.join(file_root, file_name_list[2])
            else:
                file_path = os.path.join(file_root, file_name_list[3])


        print('%'*100)
        print('file_path', file_path)
        print('%'*100)

        self.dataset = EndoInstrumentSegmentation(root, file_path, train=train, transform=transform)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)


# class EndoInstrumentSegmentation(data.Dataset):

#     def __init__(self, root, train=True, transform=None):

#         root = os.path.expanduser(root)
#         base_dir = "ISINet_Train_Val"
#         ade_root = os.path.join(root, base_dir)
#         if train:
#             split = 'train'
#         else:
#             split = 'val'

#         annotation_folder = os.path.join(ade_root, split, 'annotations')
#         image_folder = os.path.join(ade_root, split, 'images')

#         self.images = []
#         # because in Endo18 dataset, 2 images don't have the annotation masks. 
#         # num_frames should be based on annotation masks
#         fnames = sorted(os.listdir(annotation_folder))

#         # self.images is a list
#         # each element is a tuple store the path to raw image and corresponding annotation mask
#         self.images = [
#             (os.path.join(image_folder, x), os.path.join(annotation_folder, x))
#             for x in fnames
#         ]

#         self.transform = transform

#         # ================ make the target id in Endo18 same as Endo17 ========== #
#         # https://www.cnblogs.com/wanghui-garcia/p/11248416.html
#         self.target_transform = tv.transforms.Lambda(lambda t: t.
#                                                     apply_(lambda x: id_to_trainid.get(x, 255))
#                                                     )

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is the image segmentation.
#         """
#         img = Image.open(self.images[index][0]).convert('RGB')
#         target = Image.open(self.images[index][1])

#         if self.transform is not None:
#             img, target = self.transform(img, target)

#         # ================ make the target id in Endo18 same as Endo17 ========== #
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self.images)








# import cv2
# from albumentations.pytorch.transforms import img_to_tensor
# import torch

# def load_mask(path, factor=32):
#     mask = cv2.imread(str(path), 0)
#     return (mask / factor).astype(np.uint8)

# def load_image(path):
#     img = cv2.imread(str(path))
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # https://github.com/ternaus/robot-surgery-segmentation/blob/de12469172239a32c0dfe7c39832ac5e52ee2baf/dataset.py
# class EndoInstrumentSegmentation(data.Dataset):

#     def __init__(self, root, file_path, train=True, transform=None):

#         self.instrument_factor = 32

#         root = root
#         root = '/home/ren2/data2/mengya/mengya_dataset/EndoVis/'
#         Endo17_directory = '2017_RoboticInstrumentSegmentation/cropped_train/'
#         Endo18_directory ='2018_RoboticSceneSegmentation/ISINet_Train_Val/'

#         if train:
#             # file_path = '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/EndoInstrument_Pre/train.txt'
#             split = 'train'
#         else:
#             # file_path = '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/EndoInstrument_Pre/test.txt'
#             split = 'val'

#         fp=open(file_path)

#         self.images = []
#         for line in fp.readlines():
#             line=line.strip('\n')
#             Dataset_name = line.split("\t")[0]
#             image_path = line.split("\t")[-1]
            
#             if Dataset_name == 'Endo17':
#                 dataset_dir = os.path.join(root, Endo17_directory)
#                 folder = image_path.split('/')[0]
#                 image_name = image_path.split('/')[-1]

#                 annotation = os.path.join(dataset_dir, folder, 'instruments_masks', image_name)
#                 image = os.path.join(dataset_dir, folder, 'images', image_name.split('.')[0]+'.jpg')
                
#             elif Dataset_name == 'Endo18':
#                 dataset_dir = os.path.join(root, Endo18_directory)
                
#                 annotation = os.path.join(dataset_dir, split, 'annotations', 'instrument', image_path)
#                 image = os.path.join(dataset_dir, split,'images', image_path)
            
#             image_annotation_pair = (image, annotation)
#             self.images.append(image_annotation_pair)

#         fp.close()
        
#         self.transform = transform

#         # # ================ make the target id in Endo18 same as Endo17 ========== #
#         # # https://www.cnblogs.com/wanghui-garcia/p/11248416.html
#         # self.target_transform = tv.transforms.Lambda(lambda t: t.
#         #                                             apply_(lambda x: id_to_trainid.get(x))
#         #                                             )
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is the image segmentation.
#         """

#         img_file_name = self.images[index][0]
#         img = load_image(img_file_name)

#         target_file_name = self.images[index][1]
#         target = load_mask(target_file_name, self.instrument_factor)
#         print('target', np.unique(target))

#         # data = {"image": img, "mask": target}
#         # augmented = self.transform(**data) # self.transform does not include transform.ToTensor()
#         # img, target = augmented["image"], augmented["mask"]
#         img = img_to_tensor(img)
#         target = torch.from_numpy(target).long()

#         return img, target

#     def __len__(self):
#         return len(self.images)
