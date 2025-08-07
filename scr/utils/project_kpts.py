from pathlib import Path
import cv2
import numpy as np
from utils_pnp import *
import matplotlib.pyplot as plt
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scr.krn import config as config
from scr.utils.utils_pnp import *
from scr.utils.general_utils import(load_camera_matrix_sat_model,
                                    load_images,
                                    load_labels)


def project_kpts_from_img_folder(image_folder_path,
                                 json_data_path,
                                 camera_sat_json,
                                 visualize=False,
                                 unit_scale=1,
                                 visualize_points_name='keypoints',
                                 idx=0):
                                
    image_path_list = load_images(image_folder_path)
  
    annotations = load_labels(json_data_path)
    cmt, sat_model = load_camera_matrix_sat_model(camera_sat_json)
 
    image = cv2.imread(str(image_path_list[idx]), cv2.IMREAD_GRAYSCALE)
    labels = annotations[idx]
    if visualize:
        image_points = np.array(labels[visualize_points_name]).reshape(-1, 2)
    else:
        # Ground truth data
        q_gt =np.array(labels['pose'])
        t_gt = np.array(labels['translation'])*unit_scale

        distCoeffs = np.zeros((5, 1), dtype=np.float32)
        image_points = project_keypoints(q_gt, t_gt, cmt, distCoeffs, sat_model)
        image_points = image_points.T

    plt.imshow(image, cmap='gray')
    plt.scatter(image_points[:, 0], image_points[:, 1], s=10)
    plt.show()


def project_kpts_from_image(image_path,
                                 json_data_path,
                                 camera_sat_json,
                                 bl_kpts=None,
                                 unit_scale=1,
                                 visualize=False,
                                 visualize_points_name='keypoints',
                                 idx=998):                           
    
    annotations = load_labels(json_data_path)
    cmt, sat_model = load_camera_matrix_sat_model(camera_sat_json)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    labels = annotations[idx]

    if visualize:
        image_points = np.array(labels[visualize_points_name]).reshape(-1, 2)
    else:
        # Ground truth data
        q_gt =np.array(labels['pose'])
        t_gt = np.array(labels['translation'])*unit_scale

        distCoeffs = np.zeros((5, 1), dtype=np.float32)
        image_points = project_keypoints(q_gt, t_gt, cmt, distCoeffs, sat_model)
        image_points = image_points.T

    plt.imshow(image, cmap='gray')
    plt.scatter(image_points[:, 0], image_points[:, 1], s=10)
    plt.show()


if __name__ == "__main__":
    camera_sat_model = config.SAT_CAM_JSON
    image_folder_path = Path(config.IMG_DIR)
    json_path = config.LABELS_JSON
    image_path = "../../data/images/image_00000.jpg"

    project_kpts_from_img_folder(image_folder_path=image_folder_path,
                                   json_data_path=json_path,
                                   camera_sat_json=camera_sat_model, unit_scale=1, visualize=True,
                                   visualize_points_name='keypoints',
                                   idx=1999)


    # project_kpts_from_image(image_path,
    #                             json_data_path=json_path,
    #                             camera_sat_json=camera_sat_model, unit_scale=1, visualize=True,
    #                             visualize_points_name='keypoints',
    #                             idx=903)

