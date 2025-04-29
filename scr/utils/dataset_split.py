from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import shutil
import numpy as np
import random
import json
from tqdm import tqdm
import sys
sys.path.append("../../scr/krn")
import config as config

def dataset_split_for_yolo(img_folder_path="../../data_3072px/trajectories_images_no_labels",
                  labels_folder_path="../../data_3072px/labels/seg_yolo",
                  output_split_dir ="../../data_splitted/data_seg",
                  num_train=18000, num_val=2000, num_test=1530) -> None:
  

    image_path_list = [image for image in sorted(Path(img_folder_path).rglob('*.jpg'), key=lambda x: int(x.stem.split('_')[-1]))]
    labels_path_list = [label for label in sorted(Path(labels_folder_path).rglob('*.txt'), key=lambda x: int(x.stem.split('_')[-1]))]

    # print(len(image_path_list))
    # print(len(labels_path_list))
    # exit()
    
    data_splitted = Path(output_split_dir)
    train_dir = data_splitted / 'train'
    val_dir = data_splitted / 'val'
    test_dir = data_splitted / 'test'

    folder_list = [train_dir, val_dir, test_dir]
    for dir in folder_list:
        if dir.is_dir():
            shutil.rmtree(dir)
            dir.mkdir(parents=True)
        else:
            dir.mkdir(parents=True)
        # For each dir create images and labels subdirs
        image_folder, label_folder = dir / 'images', dir / 'labels'
        image_folder.mkdir(), label_folder.mkdir()
    
    num_images = len(image_path_list)
    assert num_train + num_val + num_test == num_images, "The sum of num_train, num_val, and num_test must be equal to the total number of images"

    for i, (img, label) in tqdm(enumerate(zip(image_path_list, labels_path_list)),
                                total=num_images,
                                desc="images processed",
                                ncols=80):
        if i < num_train:
            shutil.copy(img, train_dir/'images')
            shutil.copy(label, train_dir/'labels')
        elif i < num_train + num_val:
            shutil.copy(img, val_dir/'images')
            shutil.copy(label, val_dir/'labels')
        else:
            shutil.copy(img, test_dir/'images')
            shutil.copy(label, test_dir/'labels')

    # Print number of trained files
    new_train_im_list = [im_new for im_new in sorted((train_dir/'images').rglob('*.jpg')) ]
    new_train_label_list = [lb_new for lb_new in sorted((train_dir/'labels').rglob('*.txt'))] 
    print(f"Number of train images and labels: {len(new_train_im_list)}, {len(new_train_label_list)}")

    # Print number of val files
    new_val_im_list = [im_new for im_new in sorted((val_dir/'images').rglob('*.jpg')) ]
    new_val_label_list = [lb_new for lb_new in sorted((val_dir/'labels').rglob('*.txt'))] 
    print(f"Number of val images and labels: {len(new_val_im_list)}, {len(new_val_label_list)}")

    # Print number of test files
    new_test_im_list = [im_new for im_new in sorted((test_dir/'images').rglob('*.jpg')) ]
    new_test_label_list = [lb_new for lb_new in sorted((test_dir/'labels').rglob('*.txt'))] 
    print(f"Number of test images and labels: {len(new_test_im_list)}, {len(new_test_label_list)}")



if __name__ == "__main__":
    img_folder_path = config.IMG_DIR
    labels_folder_path = "../../data_3072px/labels/seg_yolo"
    output_split_dir ="../../data_splitted/data_seg"

    dataset_split_for_yolo(img_folder_path, labels_folder_path, output_split_dir,
                           num_train=21600, num_val=2700, num_test=2700)
