from pathlib import Path
import cv2
import numpy as np
import json
from utils_pnp import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def project_kpts_from_img_folder(image_folder_path="../../data/images/trajectories_images",
                                 json_data_path="../../data/labels/meta_keypoints.json",
                                 camera_sat_json="../../data/labels/cam_sat.json",
                                 idx=0):                           
    
    with open(json_data_path, 'r') as f:
        annotations = json.load(f)

    with open(camera_sat_json, 'r') as json_file:
        data = json.load(json_file)
    sat_model, cmt = np.array(data['sat_model'])*(-1), np.array(data['camera_matrix']) 
    image_path_list = sorted([image for image in image_folder_path.rglob('*.png')])

    image = cv2.imread(str(image_path_list[idx]), cv2.IMREAD_GRAYSCALE)
    labels = annotations[idx] 

    # Ground truth data
    q_gt =np.array(labels['pose'])
    t_gt = labels['translation']

    distCoeffs = np.zeros((5, 1), dtype=np.float32)
    image_points = project_keypoints(q_gt, t_gt, cmt, distCoeffs, sat_model)
    image_points = image_points.T

    plt.imshow(image, cmap='gray')
    plt.scatter(image_points[:, 0], image_points[:, 1], s=10)
    plt.savefig("projected_keypoints.png")
    # plt.show()
