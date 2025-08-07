import cv2
import numpy as np
from cv2 import aruco
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def load_camera_parameters(camera_matrix_path, dist=True):
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

def draw_satellite_model(image, rvec, tvec, camera_matrix, dist_coeffs):
    """Draw the satellite model based on the given pose"""
    # Define the satellite model points (in meters)
    sat_model = np.float32([
        [-0.15, -0.15, -0.30],
        [-0.15,  0.15, -0.30],
        [ 0.15,  0.15, -0.30],
        [ 0.15, -0.15, -0.30],
        [-0.15, -0.15, 0.0],
        [-0.15,  0.15, 0.0],
        [ 0.15,  0.15, 0.0],
        [ 0.15, -0.15, 0.0],
        [-0.649869, -0.150108,  -0.154868],
        [-0.649869,  0.153118,  -0.154868],
        [ 0.649874,  0.153089,  -0.154842],
        [ 0.649874, -0.150088,  -0.154841],
        [ 0.04   ,  0.04    , 0],
        [ 0.04   , -0.04    , 0],
        [-0.04   , -0.04    , 0],
        [-0.04   ,  0.04    , 0]
    ])

    # Project the 3D points to 2D
    imgpts, _ = cv2.projectPoints(sat_model, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.round(imgpts).astype(np.int32)
    
    # Define colors
    red = (0, 0, 255)    # BGR
    green = (0, 255, 0)  # BGR
    blue = (255, 0, 0)   # BGR
    yellow = (0, 255, 255) # BGR
    line_width = 2

    # Draw main body (cube)
    # Bottom face
    for i in range(4):
        cv2.line(image, tuple(imgpts[i][0]), tuple(imgpts[(i+1)%4][0]), red, line_width)
    # Top face
    for i in range(4):
        cv2.line(image, tuple(imgpts[i+4][0]), tuple(imgpts[((i+1)%4)+4][0]), red, line_width)
    # Pillars
    for i in range(4):
        cv2.line(image, tuple(imgpts[i][0]), tuple(imgpts[i+4][0]), red, line_width)

    # Draw solar panels
    cv2.line(image, tuple(imgpts[8][0]), tuple(imgpts[9][0]), blue, line_width)
    cv2.line(image, tuple(imgpts[9][0]), tuple(imgpts[10][0]), blue, line_width)
    cv2.line(image, tuple(imgpts[10][0]), tuple(imgpts[11][0]), blue, line_width)
    cv2.line(image, tuple(imgpts[11][0]), tuple(imgpts[8][0]), blue, line_width)

    # # Draw small cube on top
    # for i in range(4):
    #     cv2.line(image, tuple(imgpts[i+12][0]), tuple(imgpts[((i+1)%4)+12][0]), green, line_width)
    # for i in range(4):
    #     cv2.line(image, tuple(imgpts[i][0]), tuple(imgpts[i+12][0]), green, line_width)

    return image

def estimate_pose(image_path, camera_matrix, distortion_coefficients, marker_length):
    """Estimate pose of ArUco markers with satellite model projection"""
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

        for i in range(len(ids)):
            # Draw axes
            image = cv2.drawFrameAxes(
                image, camera_matrix, distortion_coefficients,
                rvecs[i], tvecs[i], marker_length * 0.5
            )
            
            # Draw satellite model
            image = draw_satellite_model(image, rvecs[i], tvecs[i], 
                                       camera_matrix, distortion_coefficients)
            
            # Put marker ID near the marker
            center = np.mean(corners[i][0], axis=0).astype(int)
            cv2.putText(image, f"ID:{ids[i][0]}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Print pose information
            distance = np.linalg.norm(tvecs[i])
            print(f'Aruco Marker ID {ids[i][0]}')
            print(f'Rvec: {rvecs[i].flatten()}')
            print(f'Tvec: {tvecs[i].flatten()}')
            print(f'Distance: {distance:.2f} m\n')

    return image

if __name__ == "__main__":
    image_path = "/home/roman/spacecraft-pose-estimation-trajectories/data_real/lux_sat_data_real_v4/frame_bright_0203.jpg"
    camera_matrix_path = "/home/roman/spacecraft-pose-estimation-trajectories/scr/realsense/camera_calibration_1280_720.json"  
    marker_length = 0.08  # Physical size of marker in meters

    camera_matrix, distortion_coefficients = load_camera_parameters(camera_matrix_path, dist=True)
    result_image = estimate_pose(image_path, camera_matrix, distortion_coefficients, marker_length)
    
    if result_image is not None:
        cv2.imwrite("images/aruco_marker_pose_with_model.png", result_image)
        cv2.imshow("ArUco Marker Pose Estimation with Satellite Model", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()