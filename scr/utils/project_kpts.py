from pathlib import Path
import cv2
import numpy as np
import json
from utils_pnp import *
import matplotlib.pyplot as plt


# Load images     
image_folder_path = Path("../../data/images/trajectories")
json_data_path = "../../data/labels/meta_keypoints.json"
camera_sat_json = "../../data/labels/cam_sat.json"


# Load json data
with open(json_data_path, 'r') as f:
    annotations = json.load(f)

# Load math model of the satellite and camera matrix
with open(camera_sat_json, 'r') as json_file:
    data = json.load(json_file)
sat_model, cmt = np.array(data['sat_model'])*(-1), np.array(data['camera_matrix'])

idx = 0

image_path_list = sorted([image for image in image_folder_path.rglob('*.png')])
image = cv2.imread(str(image_path_list[idx]), cv2.IMREAD_GRAYSCALE)
labels = annotations[idx] 

# Ground truth data
q_gt =np.array(labels['rotation'])
t_gt = np.array(labels['distance'])

# PROJECT KEYPOINTS SPEED SOLUTION
distCoeffs = np.zeros((5, 1), dtype=np.float32)
image_points = project_keypoints(q_gt, t_gt, cmt, distCoeffs, sat_model)
image_points = image_points.T



# Define 3D object points (e.g., a cube corner or keypoints in 3D space)
objectPoints = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)

# Camera intrinsic matrix (example values)
cameraMatrix = np.array([[1000, 0, 640],  # fx, 0, cx
                         [0, 1000, 360],  # 0, fy, cy
                         [0, 0, 1]], dtype=np.float32)

# No lens distortion
distCoeffs = np.zeros(5)  

# Rotation vector (Rodrigues format)
rvec = np.array([0, 0, 0], dtype=np.float32)  # No rotation

# Translation vector
tvec = np.array([0, 0, 10], dtype=np.float32)  # Move 10 units along Z-axis

# Project 3D points to 2D
imagePoints, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)

print("Projected 2D Points:\n", imagePoints)


print(q_gt, t_gt)
plt.imshow(image, cmap='gray')
plt.show()
