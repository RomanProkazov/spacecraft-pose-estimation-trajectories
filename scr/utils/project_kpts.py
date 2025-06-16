from pathlib import Path
import cv2
import numpy as np
import json
from utils_pnp import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scr.krn import config as config
from scr.utils.utils_pnp import *


def project_kpts_from_img_folder(image_folder_path,
                                 json_data_path,
                                 camera_sat_json,
                                 idx=0):                           
    
    with open(json_data_path, 'r') as f:
        annotations = json.load(f)

    with open(camera_sat_json, 'r') as json_file:
        data = json.load(json_file)
    sat_model, cmt = np.array(data['sat_model']), np.array(data['camera_matrix']) 
    image_path_list = sorted([image for image in Path(image_folder_path).rglob('*.jpg')], key=lambda x: int(x.stem.split('_')[-1]))


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
    plt.show()


def visualize_kpts_from_img_folder(image_folder_path,
                                 json_data_path,
                                 camera_sat_json,
                                 idx=0):                           
    
    with open(json_data_path, 'r') as f:
        annotations = json.load(f)

    image_path_list = sorted([image for image in Path(image_folder_path).rglob('*.jpg')], key=lambda x: int(x.stem.split('_')[-1]))

    image = cv2.imread(str(image_path_list[idx]), cv2.IMREAD_GRAYSCALE)
    labels = annotations[idx] 

    print(image_path_list[idx])
    # print(labels)

    image_points = np.array(labels['keypoints']).reshape(-1, 2)

    plt.imshow(image, cmap='gray')
    plt.scatter(image_points[:, 0], image_points[:, 1], s=10)
    plt.show()


def visualize_kpts_from_img(image_path,
                                 json_data_path,
                                 camera_sat_json,
                                 idx=0):                           
    
    with open(json_data_path, 'r') as f:
        annotations = json.load(f)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    labels = annotations[idx] 

    image_points = np.array(labels['keypoints']).reshape(-1, 2)
    print(image_points)
    image_points = image_points # for 3072px images 
    print(image_points)

    plt.imshow(image, cmap='gray')
    plt.scatter(image_points[:, 0], image_points[:, 1], s=10)
    plt.show()


def project_kpts_from_image(image_path,
                                 json_data_path,
                                 camera_sat_json,
                                 bl_kpts=None,
                                 idx=0):                           
    
    with open(json_data_path, 'r') as f:
        annotations = json.load(f)

    with open(camera_sat_json, 'r') as json_file:
        data = json.load(json_file)
    sat_model, cmt = np.array(data['sat_model']), np.array(data['camera_matrix']) 
    sat_model[:, 0] *= -1
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    labels = annotations[idx]

    # Ground truth data
    q_gt =np.array(labels['pose'])
    t_gt = np.array(labels['translation'])*400000

    distCoeffs = np.zeros((5, 1), dtype=np.float32)
    # distCoeffs = np.array(([ -0.029736043469149088, -0.0022844622995838583, -0.001739466580238918, -0.0006001510606988042, 0.07342321343596342
    #     ]), dtype=np.float32)
    image_points = project_keypoints(q_gt, t_gt, cmt, distCoeffs, sat_model)
    image_points = image_points.T

    plt.imshow(image, cmap='gray')
    plt.scatter(image_points[:, 0], image_points[:, 1], s=10)
    plt.show()


if __name__ == "__main__":
    camera_sat_model = config.SAT_CAM_JSON
    image_folder_path = Path(config.IMG_DIR)
    json_path = config.LABELS_JSON
    json_path = "/home/roman/spacecraft-pose-estimation-trajectories/data/labels/labels_sat_1280px_20kimgs_leo_2.json"
    image_path = "/home/roman/spacecraft-pose-estimation-trajectories/data/images/image_00900.jpg"


    # old data
    # image_path = "../../data_3072px/images/img_26986.jpg"
    # model_path = "../../data_3072px/labels/labels_sat_27kimgs.json"

    # # # check ok
    # image_path = "/home/roman/Desktop/LUXEMBOURG PROJECT/blender-related/files/data/images/blender_kpts_test/Image0142.png"
    # json_path = "/home/roman/Desktop/LUXEMBOURG PROJECT/blender-related/files/data/labels/meta_keypoints.json"
    # camera_sat_model = "../../data_512px_5kimgs/labels/cam_sat.json"

    
    # project_kpts_from_img_folder(image_folder_path=image_folder_path,
    #                                json_data_path=json_path,
    #                                camera_sat_json=camera_sat_model,
    #                                idx=199)

    # visualize_kpts_from_img_folder(image_folder_path=image_folder_path,
    #                                json_data_path=json_path,
    #                                camera_sat_json=camera_sat_model,
    #                                idx=1799)

    # project_kpts_from_image(image_path,
    #                             json_data_path=json_path,
    #                             camera_sat_json=camera_sat_model,
    #                             idx=899)

    visualize_kpts_from_img(image_path=image_path,
                                json_data_path=json_path,
                                camera_sat_json=camera_sat_model,
                                idx=899)