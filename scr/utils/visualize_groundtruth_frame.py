from pathlib import Path
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


def visualize_kpts_from_img_folder(image_folder_path="../../data_3072px/images",
                                 json_data_path="../../data_3072px/labels/labels_sat_36kimgs.json",
                                 camera_sat_json="../../data_platform/labels/cam_plat.json",
                                 idx=8):                           
    
    with open(json_data_path, 'r') as f:
        annotations = json.load(f)

    image_path_list = sorted([image for image in Path(image_folder_path).rglob('*.png')])
    image = cv2.imread(str(image_path_list[idx]), cv2.IMREAD_GRAYSCALE)
    labels = annotations[idx]


    image_points = np.array(labels['keypoints']).reshape(-1, 2)

    # print(image_path_list[idx])
    # print(labels)
    # exit()

    plt.imshow(image, cmap='gray')
    plt.scatter(image_points[:, 0], image_points[:, 1], s=10)
    plt.show()

visualize_kpts_from_img_folder()