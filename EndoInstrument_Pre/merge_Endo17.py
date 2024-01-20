import os
import shutil
import random

train = False

root = '/home/ren2/data2/mengya/mengya_dataset/EndoVis/'
Endo17_train_directory = '2017_RoboticInstrumentSegmentation/cropped_train/'
Endo17_test_directory = '2017_RoboticInstrumentSegmentation/cropped_test'
Endo17_merged_directory = '2017_RoboticInstrumentSegmentation/cropped_train_test_no_1_2/'

"""
we did not add seq 1-2 from official test set because the official release did not label the Ultrasound Probe, while in train set of seq 1-2, they do release the label of Ultrasound Probe. 
Ultrasound Probe is also a common instrument in 17 & 18
"""


merged_folder = [3,4,5,6,7,8] # skip 1 and 2 in cropped_test 

if train:
    Endo17_directory = Endo17_train_directory
else:
    Endo17_directory = Endo17_test_directory

for folder_id in merged_folder:
    annotation_folder_dir = os.path.join('instrument_dataset_' + str(folder_id), 'instruments_masks') 
    image_folder_dir = os.path.join('instrument_dataset_' + str(folder_id), 'images')

    annotation_folder = os.path.join(root, Endo17_directory, annotation_folder_dir) 
    # print('annotation_folder', annotation_folder)
    image_folder = os.path.join(root, Endo17_directory, image_folder_dir)
    
    new_annotation_folder = os.path.join(root, Endo17_merged_directory, annotation_folder_dir)
    print('new_annotation_folder', new_annotation_folder)
    new_image_folder = os.path.join(root, Endo17_merged_directory, image_folder_dir)

    
    annotation_listing = os.listdir(annotation_folder)
    # print('annotation_listing', annotation_listing) # ['frame137.png', 'frame189.png'..]
    annotation_files = []

    for annotation in annotation_listing:
        annotation_files.append(annotation)

    for annotation_index in range(0,len(annotation_files)):
        shutil.copy(os.path.join(annotation_folder,annotation_files[annotation_index]), new_annotation_folder)

    image_listing = os.listdir(image_folder)
    image_files = []

    for image in image_listing:
        image_files.append(image)

    for image_index in range(0,len(image_files)):
        shutil.copy(os.path.join(image_folder, image_files[image_index]), new_image_folder)

