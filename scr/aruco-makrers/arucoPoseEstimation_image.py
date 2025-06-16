import cv2
import numpy as np
from cv2 import aruco
import json
import screeninfo

def load_camera_parameters(camera_matrix_path):
    """Load camera calibration data from JSON file"""
    with open(camera_matrix_path, 'r') as f:
        cam_dist = json.load(f)
    camera_matrix = np.array(cam_dist['camera_matrix'])
    distortion_coeffs = np.array(cam_dist['dist'])
    return camera_matrix, distortion_coeffs


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def estimate_pose(image_path, camera_matrix, distortion_coefficients, marker_length):
    """Estimate pose of ArUco markers in an image"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize ArUco detector with modern API
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detector_params)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        # Draw detected markers
        image = aruco.drawDetectedMarkers(image, corners, ids)

        # Estimate pose for each marker
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
            corners, 
            marker_length, 
            camera_matrix, 
            distortion_coefficients
        )

        # Draw axis and display pose for each marker
        for i in range(len(ids)):
            image = cv2.drawFrameAxes(
                image, 
                camera_matrix, 
                distortion_coefficients, 
                rvecs[i], 
                tvecs[i], 
                marker_length * 0.5
            )
            
            print(f"Marker ID: {ids[i][0]}")
            print(f"Rotation Vector (rvec):\n{rvecs[i]}")
            print(f"Translation Vector (tvec):\n{tvecs[i]}")
            print("-" * 40)
    else:
        print("No markers detected.")

    
    cv2.imwrite("output_pose_estimation.png", image)

if __name__ == "__main__":
    # Configuration
    image_path = "/home/roman/spacecraft-pose-estimation-trajectories/data_real/lux_sat_data_real_v1_nobck/frame_1511.jpg"  # Update this path
    camera_matrix_path = "../../data_1280px/labels/cam_sat.json"     # Update this path
    marker_length = 0.08                   # Physical size of marker in meters

    # Run pose estimation
    camera_matrix, distortion_coefficients = load_camera_parameters(camera_matrix_path)
    estimate_pose(image_path, camera_matrix, distortion_coefficients, marker_length)