from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import json
from utils_pnp import *
import matplotlib.pyplot as plt
import sys
sys.path.append("../../scr/krn")
import config as config



def add_kpts_to_json(image_folder_path,
                     json_data_path,
                     camera_sat_json,
                     output_json,
                     res=(640, 480)):

    image_path_list = sorted([image for image in Path(image_folder_path).rglob('*.jpg')], key=lambda x: int(x.stem.split('_')[-1]))
    with open(json_data_path, 'r') as f:
        annotations = json.load(f)
    # Clean data from blender unused info
    keys_to_remove = ["position", "distance", "background", "lighting"] # "offset"
    annotations = [
        {key: value for key, value in element.items() if key not in keys_to_remove}
        for element in annotations
    ]
    with open(camera_sat_json, 'r') as json_file:
        data = json.load(json_file)
    sat_model, cmt = np.array(data['sat_model']), np.array(data['camera_matrix'])
    sat_model[:, 0] *= -1


    kpts_list = []
    # annotations = annotations[:14000]  # Limit to 14k images for processing
    num_images = len(image_path_list)
    t_error_sum, r_error_sum, speed_score_sum  = 0, 0, 0
    for idx in tqdm(range(num_images),
                    total=num_images,
                    desc="Images processed",
                    ncols=80):

        image = cv2.imread(str(image_path_list[idx]), cv2.IMREAD_GRAYSCALE)
        labels = annotations[idx] 

        # Ground truth data
        q_gt =np.array(labels['pose'])
        t_gt = np.array(labels['translation'])*400000

        distCoeffs = np.zeros((5, 1), dtype=np.float32)
        image_points = project_keypoints(q_gt, t_gt, cmt, distCoeffs, sat_model)
        image_points = image_points.T
        image_points_marker = image_points[-4:]

        # Calculate bbox coords from keypoints    
        x_min = np.min(image_points[:, 0])
        y_min = np.min(image_points[:, 1])
        x_max = np.max(image_points[:, 0])
        y_max = np.max(image_points[:, 1])

        # Clip bounding box coordinates to the frame
        x_min = max(0, min(x_min, res[0]))
        y_min = max(0, min(y_min, res[1]))
        x_max = max(0, min(x_max, res[0]))
        y_max = max(0, min(y_max, res[1]))

        # Marker bounding box  
        x_min_marker = np.min(image_points_marker[:, 0])
        y_min_marker = np.min(image_points_marker[:, 1])
        x_max_marker = np.max(image_points_marker[:, 0])
        y_max_marker = np.max(image_points_marker[:, 1])

        bbox = [x_min, y_min, x_max, y_max]
        bbox_array = np.array(bbox).reshape(2, 2)

        bbox_marker = [x_min_marker, y_min_marker, x_max_marker, y_max_marker]
        bbox_array_marker= np.array(bbox_marker).reshape(2, 2)

        labels['keypoints'] = image_points.tolist()
        labels['filename'] = image_path_list[idx].name
        labels['translation'] = t_gt.tolist()
        labels['bbox_xyxy'] = bbox
        labels['bbox_marker'] = bbox_marker

        # print(image_points)
        # plt.imshow(image, cmap='gray')
        # plt.scatter(image_points[:, 0], image_points[:, 1], s=10)
        # # plt.scatter(bbox_array[:, 0], bbox_array[:, 1], s=10)
        # plt.show()


    with open(output_json, "w") as json_file:
        json.dump(annotations, json_file, indent=4)


if __name__ == "__main__":
    output_json = "../../data/labels/labels_sat_1280px_20kimgs_leo_2.json"
    add_kpts_to_json(image_folder_path=config.IMG_DIR,
                     json_data_path=config.LABELS_JSON,
                     camera_sat_json=config.SAT_CAM_JSON,
                     output_json=output_json,
                     res=(1280, 720))