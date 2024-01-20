import PIL.Image
import numpy as np
from skimage import io, data, color
from skimage import img_as_ubyte
from tqdm import tqdm
import os
import cv2
import pathlib



# # ===== 2018 ======== #	\
# # Background: 0


# # "Bipolar_Forceps": 1 

# # "Prograsp_Forceps": 2

# # "Large_Needle_Driver": 3 

# # "Monopolar_Curved_Scissors": 4 

# # "Ultrasound_Probe": 5 


# # "Suction_Instrument": 6  # unique

# # "Clip_Applier": 7 # unique

# id_to_trainid = {
#     0:    0,

#     32:   1,
#     64:   2,
#     96:   3,

#     128:  4,
#     160:  5,

#     192:  6,
#     224:  7,


#     1:    1,
#     2:    2,
#     3:    3,

#     4:    6,
#     5:    7,

#     6:    8,
#     7:    9
# }


# for 18 dataset
factor = 1

color  = {
        0*factor: [0, 0, 0],
        1*factor: [144, 238, 144], # light green
        2*factor: [153, 255, 255], # light blue
        3*factor: [0, 102, 255], # dark blue

        # 4*factor: [255, 55, 0], # red
        # 5*factor: [0, 153, 51], # dark green

        4*factor: [187, 155, 25], # khaki
        5*factor: [255, 204, 255], # pink
        6*factor: [255, 255, 125], # light yellow
        7*factor: [123, 15, 175], # purple
        # 10*factor: [124, 155, 5],
        # 11*factor: [125, 255, 12],
}


def visual_mask(src_path, dst_path):
    mask = PIL.Image.open(src_path)
    mask = np.asarray(mask)
    # print('mask', mask.shape) # (224, 224)  # mask (1024, 1280)
    # print('unique', np.unique(mask))
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color_mask[i, j] = color[mask[i, j]] # # if the mask does not have  RGB channels
            # color_mask[i, j] = color[mask[i, j, 0]] # if the mask has RGB channelss
            
            # if mask[i, j,0] == 11*factor:
            #     print("predict the tumor")

    mask = PIL.Image.fromarray(color_mask.astype(np.uint8))

    # # ==== Resize it into 1024*1280 === #
    # newsize = (1280, 1024) # (1280, 1024), (1024, 1280)
    # mask = mask.resize(newsize)
    # # ================================= #

    mask.save(dst_path)


# # 17_gt
# src_base_path = [
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/gt_mask_17/instrument_dataset_2/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/gt_mask_17/instrument_dataset_5/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/gt_mask_17/instrument_dataset_9/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/gt_mask_17/instrument_dataset_10/instruments_masks/',
# ]

# dst_base_path = [
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/gt_mask_17_RGB/instrument_dataset_2/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/gt_mask_17_RGB/instrument_dataset_5/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/gt_mask_17_RGB/instrument_dataset_9/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/gt_mask_17_RGB/instrument_dataset_10/instruments_masks/',
# ]


# # 18_gt
# src_base_path = [
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/gt_mask_18/',
# ]

# dst_base_path = [
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/gt_mask_18_RGB/',
# ]


# # 17 predict
# src_base_path = [
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/predicted_mask_17/instrument_dataset_2/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/predicted_mask_17/instrument_dataset_5/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/predicted_mask_17/instrument_dataset_9/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/predicted_mask_17/instrument_dataset_10/instruments_masks/',
# ]

# dst_base_path = [
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/predicted_mask_17_RGB/instrument_dataset_2/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/predicted_mask_17_RGB/instrument_dataset_5/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/predicted_mask_17_RGB/instrument_dataset_9/instruments_masks/',
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/predicted_mask_17_RGB/instrument_dataset_10/instruments_masks/',
# ]


# # 18 predict
# src_base_path = [
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/predicted_mask_18/',
# ]

# dst_base_path = [
#                 '/home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/MASK/predicted_mask_18_RGB/',
# ]


# # Blended 18
# src_base_path = [
#                 '/home/ren2/data2/mengya/mengya_code/Synthesis_18/Synthesis/data_gen/blended/blended_2_50/annotations/multi/',
# ]

# dst_base_path = ['/home/ren2/data2/mengya/mengya_code/Synthesis_18/Synthesis/data_gen/blended/blended_2_50/annotations/multi_RGB/',
# ]



# Single tool
src_base_path = [
                '/home/ren2/data2/mengya/mengya_code/Synthesis_18/Synthesis/data_gen/foreground/3-Large_Needle_Driver/mask_pos_2/',
]

dst_base_path = ['/home/ren2/data2/mengya/mengya_code/Synthesis_18/Synthesis/data_gen/foreground/3-Large_Needle_Driver/mask_pos_2_RGB/',
]

print("Start🙂")
for i in range(len(dst_base_path)):
    if not os.path.isdir(dst_base_path[i]):
        pathlib.Path(dst_base_path[i]).mkdir(parents=True, exist_ok=True)
    mask_names = os.listdir(src_base_path[i])
    print(len(mask_names))
    for j in tqdm(range(len(mask_names))):
        src_mask = src_base_path[i] + mask_names[j]
        dst_mask = dst_base_path[i] + mask_names[j]
        visual_mask(src_mask, dst_mask)
print("Done😀")

