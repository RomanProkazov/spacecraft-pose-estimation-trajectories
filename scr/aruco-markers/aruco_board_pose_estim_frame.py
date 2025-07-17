import numpy as np
import pyrealsense2 as rs
import json
from pathlib import Path
import sys
import cv2

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


def get_extrinsic_matrix(rvec, tvec):
    """Convert rotation vector and translation to 4x4 extrinsic matrix"""
    R, _ = cv2.Rodrigues(rvec)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = tvec.flatten()
    return extrinsic


if __name__ == "__main__":

    camera_matrix_path = "/home/roman/spacecraft-pose-estimation-trajectories/scr/realsense/camera_calibration_1280_720_ros.json"
    image_path = "/home/roman/spacecraft-pose-estimation-trajectories/scr/realsense/calib_extrinsic_comare_all/color_arucoboard_0000.png"
    MARKER_SIZE = 0.0315      # 31.5mm in meters
    MARKER_SEPARATION = 0.004  # 4mm in meters (edge-to-edge gap)
    BOARD_SIZE = (5, 7)       # Number of markers (width, height)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.GridBoard(
        size=BOARD_SIZE,
        markerLength=MARKER_SIZE,
        markerSeparation=MARKER_SEPARATION,
        dictionary=aruco_dict
    )

    camera_matrix, dist_coeffs = load_camera_parameters(camera_matrix_path, dist=True)
    # Load image
    frame = cv2.imread(image_path)  # Replace with your image path
    if frame is None:
        raise FileNotFoundError("Image not found!")

    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        # Estimate board pose
        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, ids, board, camera_matrix, dist_coeffs, None, None
        )
        distance = np.linalg.norm(tvec)
        if retval:
            extrinsic = get_extrinsic_matrix(rvec, tvec)
            # print("Extrinsic Matrix (World-to-Camera):\n", extrinsic)

            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw marker IDs
            for i in range(len(ids)):
                center = np.mean(corners[i][0], axis=0).astype(int)
                cv2.putText(frame, str(ids[i][0]), tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            # Verify reprojection
            projected_origin, _ = cv2.projectPoints(
                np.array([0, 0, 0], dtype=np.float32), rvec, tvec, camera_matrix, dist_coeffs)
            cv2.circle(frame, tuple(projected_origin[0][0].astype(int)), 5, (255, 0, 0), -1)
        print(f'Aruco Board')
        print(f'Rvec: {rvec.flatten()}')
        print(f'Tvec: {tvec.flatten()}')
        print(f'Distance: {distance:.2f} m')

    # Show result
    cv2.imwrite("images/aruco_board_pose.png", frame)
    cv2.imshow("ArUco Board Pose Estimation", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()