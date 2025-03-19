from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import json
from utils_pnp import *
import matplotlib.pyplot as plt


# Load images     
image_folder_path = Path("../../data/images/trajectories")
json_data_path = "../../data/labels/meta_keypoints.json"
camera_sat_json = "../../data/labels/cam_sat.json"

# Output path for upadated json file
output_json = "../../data/labels/kpts_added.json"

image_path_list = sorted([image for image in image_folder_path.rglob('*.png')])


# Load json data
with open(json_data_path, 'r') as f:
    annotations = json.load(f)

# # Clean data from unused info
# keys_to_remove = ["position", "distance", "lightning", "offset", "background", "lighting"]
# annotations = [
#     {key: value for key, value in element.items() if key not in keys_to_remove}
#     for element in annotations
# ]

# Load math model of the satellite and camera matrix
with open(camera_sat_json, 'r') as json_file:
    data = json.load(json_file)
sat_model, cmt = np.array(data['sat_model'])*(-1), np.array(data['camera_matrix'])


kpts_list = []
num_images = len(image_path_list)
t_error_sum, r_error_sum, speed_score_sum  = 0, 0, 0
for idx in tqdm(range(num_images),
                total=num_images,
                desc="Images processed",
                ncols=80):

    image = cv2.imread(str(image_path_list[idx]), cv2.IMREAD_GRAYSCALE)
    labels = annotations[idx] 

    # Ground truth data
    q_gt =np.array(labels['rotation'])
    t_gt = np.array(labels['distance'])

    # PROJECT KEYPOINTS SPEED SOLUTION
    distCoeffs = np.zeros((5, 1), dtype=np.float32)
    image_points = project_keypoints(q_gt, t_gt, cmt, distCoeffs, sat_model)
    image_points = image_points.T

    # Calculate bbox coords from keypoints    
    x_min = np.min(image_points[:, 0])
    y_min = np.min(image_points[:, 1])
    x_max = np.max(image_points[:, 0])
    y_max = np.max(image_points[:, 1])
    bbox = [x_min, y_min, x_max, y_max]

    labels['keypoints'] = image_points.tolist()
    labels['filename'] = image_path_list[idx].name
    labels['translation'] = t_gt.tolist()
    labels['bbox_xyxy'] = bbox

    print(image_points)

    plt.imshow(image, cmap='gray')
    plt.scatter(image_points[:, 0], image_points[:, 1], s=10)
    plt.show()
    # exit()


with open(output_json, "w") as json_file:
    json.dump(annotations, json_file, indent=4)