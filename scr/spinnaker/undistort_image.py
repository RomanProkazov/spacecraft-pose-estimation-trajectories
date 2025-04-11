import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import json

cmt_path = Path('camera_calibration_final_mono.json')
img_path = Path("calibration_images4/img_1744115043782.png")
with open(cmt_path, "r") as f:
    data = json.load(f)
camera_matrix = np.array(data["camera_matrix"])
dist_coefs = np.array(data["distortion_coefficients"])
img = cv2.imread(str(img_path))
h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

# Finding the new optical camera matrix
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
im_undistorted = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

x, y, w_2, h_2 = roi
im_undistorted = im_undistorted[y:y+h_2, x:x+w_2]

plt.imshow(cv2.cvtColor(im_undistorted, cv2.COLOR_BGR2RGB))
plt.show()