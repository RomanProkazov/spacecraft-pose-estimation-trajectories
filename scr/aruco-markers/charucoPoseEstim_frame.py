import cv2
import numpy as np
from cv2 import aruco
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# ------------------------------
# CONFIGURATION (MODIFY THESE)
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7  # Number of squares vertically
SQUARES_HORIZONTALLY = 5  # Number of squares horizontally
SQUARE_LENGTH = 0.037  # 37mm in meters
MARKER_LENGTH = 0.018  # 18mm in meters
CALIBRATION_JSON = "/home/roman/spacecraft-pose-estimation-trajectories/scr/realsense/camera_calibration_1280_720.json"
IMAGE_PATH = "/home/roman/spacecraft-pose-estimation-trajectories/scr/realsense/calib_extrinsic_comare_all/color_charucoboard_0000.png"
# ------------------------------

def load_camera_parameters(calibration_path):
    """Load camera calibration data from JSON file"""
    with open(calibration_path) as f:
        data = json.load(f)
    camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(data['dist'], dtype=np.float32)
    return camera_matrix, dist_coeffs

def detect_pose(image, camera_matrix, dist_coeffs):
    """Detect Charuco board and estimate pose with enhanced visualization"""
    # Initialize board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    # Detect markers
    corners, ids, _ = detector.detectMarkers(image)
    
    if ids is None:
        print("No markers detected!")
        output_image = image.copy()
        cv2.putText(output_image, "No markers detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output_image

    # Interpolate Charuco corners
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, image, board
    )

    if not retval or charuco_ids is None:
        print("Charuco corners not found!")
        output_image = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
        cv2.putText(output_image, "Charuco corners not found", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output_image

    # Estimate pose
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board,
        camera_matrix, dist_coeffs, None, None
    )

    # Prepare output image
    output_image = image.copy()
    output_image = cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
    
    if not success:
        print("Pose estimation failed!")
        cv2.putText(output_image, "Pose estimation failed", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output_image

    # Draw coordinate axes
    cv2.drawFrameAxes(output_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # Convert rotation to degrees
    rvec_deg = np.degrees(rvec).flatten()

    # Print to console
    print(f'Charuco Board')
    print(f'Rvec: {rvec.flatten()}')
    print(f'Tvec: {tvec.flatten()}')
    print(f"\nDistance: {np.linalg.norm(tvec):.2f} m")

    return output_image

def main():
    # Load calibration
    camera_matrix, dist_coeffs = load_camera_parameters(CALIBRATION_JSON)

    # Load image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image {IMAGE_PATH}")
        return

    # Detect pose
    result_image = detect_pose(image, camera_matrix, dist_coeffs)

    # Display results
   
    cv2.imwrite("images/charuco_board_pose.png", result_image)
    cv2.imshow("Charuco Pose Estimation", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()