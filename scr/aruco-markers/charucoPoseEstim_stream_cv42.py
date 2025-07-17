import cv2
import numpy as np
import pyrealsense2 as rs
import json
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import pyrealsense2 as rs
import json

# ====== CONFIGURATION ======
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7        # Number of squares vertically
SQUARES_HORIZONTALLY = 5      # Number of squares horizontally
SQUARE_LENGTH = 0.0362        # 11.5mm in meters (must match printed board)
MARKER_LENGTH = 0.0184         # 8mm in meters (must match printed board)
CALIBRATION_JSON = "/home/roman/spacecraft-pose-estimation-trajectories/scr/realsense/camera_calibration_1280_720.json"

# ====== REALSENSE SETUP ======
def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

# ====== CHARUCO DETECTION ======
def initialize_detector():
    dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard_create(SQUARES_HORIZONTALLY, SQUARES_VERTICALLY, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    parameters = cv2.aruco.DetectorParameters_create()
    return board, dictionary, parameters  # Return dictionary along with board and parameters

def process_frame(frame, board, dictionary, parameters, camera_matrix, dist_coeffs):
    # Detect markers (OpenCV 4.2 style)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    
    if ids is None:
        return frame, None, None
    
    # Interpolate Charuco corners
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, frame, board)
    
    if not retval:
        output = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    
        return output, None, None
    
    # Estimate pose
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board,
        camera_matrix, dist_coeffs, None, None)
    
    if not success:
        output = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        return output, None, None
    
    # Draw results
    output = frame.copy()
    output = cv2.aruco.drawDetectedMarkers(output, corners, ids)
    cv2.aruco.drawAxis(output, camera_matrix, dist_coeffs, rvec, tvec, 0.1)  # Use drawAxis in 4.2
    
    return output, rvec, tvec


# ====== REALSENSE SETUP ======
def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

# ====== CHARUCO DETECTION ======
def initialize_detector():
    dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard_create(SQUARES_HORIZONTALLY, SQUARES_VERTICALLY, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    parameters = cv2.aruco.DetectorParameters_create()
    return board, dictionary, parameters  # Return dictionary along with board and parameters

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees, ZYX convention."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy > 1e-6:  # Normal case
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:  # Gimbal lock case (pitch near ±90°)
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0
    return np.degrees([roll, pitch, yaw])

# ====== MAIN LOOP ======
def main():
    # Load calibration
    with open(CALIBRATION_JSON) as f:
        calib = json.load(f)
    camera_matrix = np.array(calib['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(calib['dist'], dtype=np.float32)
    
    # Initialize detector and RealSense
    board, dictionary, parameters = initialize_detector()
    pipeline = setup_realsense()
    
    try:
        while True:
            # Get frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            
            # Process frame
            result, rvec, tvec = process_frame(frame, board, dictionary, parameters, camera_matrix, dist_coeffs)

            if rvec is not None:
                rot_mat, _ = cv2.Rodrigues(rvec.flatten())
                xyz_angles = rotation_matrix_to_euler_angles(rot_mat)
                
                rot = R.from_rotvec(rvec.flatten())
                quat = rot.as_quat()
                rot = R.from_quat(quat)
                euler = rot.as_euler('xyz', degrees=True)
                euler[0] -= 180

                # print(xyz_angles)
                # print(f'XYZ angles: {xyz_angles.flatten()}')
                print(f'Euler: {euler.flatten()}')
            
            # Display info if pose found
            # if rvec is not None:
            #     rvec_deg = np.degrees(rvec).flatten()
            #     cv2.putText(result, f"Pos: {tvec.flatten()[:3].round(3)}m", (20, 40), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #     cv2.putText(result, f"Rot: {rvec_deg.round(1)}deg", (20, 80), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show result
            cv2.imshow("RealSense Charuco Tracking", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()