import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import json

image_dir = Path('checkerboard_images')
output_json = Path('camera_calibration.json')
img_names = [image for image in image_dir.glob('*.png')]
# img_names = sorted(img_names, key=lambda x: int(x.stem.split('_')[-1]))
img_names = sorted(img_names)

square_size = 25  # mm
pattern_size = (10, 7)  # number of inner corners

# Building 3D points
indices = np.indices(pattern_size, dtype=np.float32)
indices *= square_size
pattern_points = np.zeros([pattern_size[0] * pattern_size[1], 3], np.float32)
coords_3D = indices.T.reshape(-1, 2)
pattern_points[:, :2] = coords_3D


def processImage(fn):
    print('processing {}'.format(fn))
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Failed to load", fn)
        return None

    found, corners = cv2.findChessboardCorners(img, pattern_size)
    if found:
        # Refining corner position
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    else:
        print('chessboard not found')
        return None

    print('           %s... OK' % fn)
    return corners.reshape(-1, 2)


# Building 2D-3D correspondences
chessboards = [processImage(str(fn)) for fn in img_names]
chessboards = [x for x in chessboards if x is not None]

obj_points = []  # 3D points
img_points = []  # 2D points

for corners in chessboards:
    img_points.append(corners)
    obj_points.append(pattern_points)

# print(f"object points: {obj_points}")

h, w = cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE).shape[:2]
print(f"Image size: {w}x{h}")

# Calibrating Camera
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

# Calculate reprojection error
mean_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
    imgpoints2 = imgpoints2.reshape(-1, 2)  # Ensure imgpoints2 has shape (N, 2)
    #plot projection
    # plt.imshow(cv2.imread(img_names[i]))   
    # plt.scatter(imgpoints2[:, 0], imgpoints2[:, 1], s=10)
    # plt.scatter(img_points[i][:, 0], img_points[i][:, 1], s=10)
    # plt.show()
    # Calculate reprojection error
    error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
    
    error = cv2.norm(img_points[i], imgpoints2.reshape(-1, 2), cv2.NORM_L2) / len(imgpoints2)
    print(f"Image {i} {img_names[i].name}: {error:.2f} px")

print("Total reprojection error: {}".format(mean_error / len(obj_points)))

print('RMS:', rms)
print('Camera matrix:\n', camera_matrix)
print('Distortion coefficients:\n', dist_coefs)

# Save camera matrix and distortion coefficients to a JSON file
calibration_data = {
    "camera_matrix": camera_matrix.tolist(),
    "distortion_coefficients": dist_coefs.tolist(),
    "rvecs": [rvec.tolist() for rvec in rvecs],
    "tvecs": [tvec.tolist() for tvec in tvecs],
    "reprojection_error": mean_error / len(obj_points),
    "rms": rms
}

with open(output_json, 'w') as f:
    json.dump(calibration_data, f, indent=4)

print(f"Calibration data saved to {output_json}")