import cv2
import numpy as np
from cv2 import aruco
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def load_camera_parameters(camera_matrix_path, dist=False):
    """Load camera calibration data from JSON file"""
    with open(camera_matrix_path, 'r') as f:
        cam_dist = json.load(f)
    camera_matrix = np.array(cam_dist['camera_matrix'], dtype=np.float32)
    if dist:
        distortion_coeffs = np.array(cam_dist['dist'], dtype=np.float32)
    else:
        distortion_coeffs = np.zeros((5, 1), dtype=np.float32)
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
    """Estimate pose of ArUco markers with top-left corner info display"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        image = aruco.drawDetectedMarkers(image.copy(), corners, ids)
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, distortion_coefficients
        )

        # Initialize top-left corner text
        info_text = []
        
        for i in range(len(ids)):
            # Draw axes and marker ID
            image = cv2.drawFrameAxes(
                image, camera_matrix, distortion_coefficients,
                rvecs[i], tvecs[i], marker_length * 0.5
            )
            
            # Put marker ID near the marker
            center = np.mean(corners[i][0], axis=0).astype(int)
            cv2.putText(image, f"ID:{ids[i][0]}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Collect info for top-left display
            distance = np.linalg.norm(tvecs[i])

            print(f'Aruco Marker')
            print(f'Rvec[{i}]: {rvecs[i].flatten()}')
            print(f'Tvec[{i}]: {tvecs[i].flatten()}')
            print(f'Distance[{i}]: {distance:.2f} m')

    
    return image

if __name__ == "__main__":
    image_path = "/home/roman/spacecraft-pose-estimation-trajectories/data/images_last_300_marker/image_01920.jpg"
    image_path = "/home/roman/spacecraft-pose-estimation-trajectories/data_real/lux_sat_data_real_v1/frame_0517.jpg"
    #image_path = "/home/roman/spacecraft-pose-estimation-trajectories/data/images_last_300_marker/image_01920.jpg"
    camera_matrix_path = "/home/roman/spacecraft-pose-estimation-trajectories/scr/realsense/camera_calibration_1280_720.json"  
    marker_length = 0.08  # Physical size of marker in meters

    camera_matrix, distortion_coefficients = load_camera_parameters(camera_matrix_path, dist=True)
    result_image = estimate_pose(image_path, camera_matrix, distortion_coefficients, marker_length)
    
    if result_image is not None:
        cv2.imwrite("images/aruco_marker_pose.png", result_image)
        cv2.imshow("ArUco Marker Pose Estimation", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()