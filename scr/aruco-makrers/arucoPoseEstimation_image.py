import cv2
import numpy as np
from cv2 import aruco
import json

def load_camera_parameters(camera_matrix_path):
    """Load camera calibration data from JSON file"""
    with open(camera_matrix_path, 'r') as f:
        cam_dist = json.load(f)
    camera_matrix = np.array(cam_dist['camera_matrix'])
    distortion_coeffs = np.array(cam_dist['distortion_coefficients'])
    return camera_matrix, distortion_coeffs

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
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
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

    # Display results
    cv2.imshow("ArUco Pose Estimation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Configuration
    image_path = "frame_1745928396935.png"  # Update this path
    camera_matrix_path = "/home/roman/spacecraft-pose-estimation-trajectories/data_3072px/labels/cam_sat.json"     # Update this path
    marker_length = 0.064                   # Physical size of marker in meters

    # Run pose estimation
    camera_matrix, distortion_coefficients = load_camera_parameters(camera_matrix_path)
    estimate_pose(image_path, camera_matrix, distortion_coefficients, marker_length)