import cv2
import numpy as np
import pyrealsense2 as rs
import json
from pathlib import Path
import sys

# Project setup (modify as needed)
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

# Initialize ArUco board
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
BOARD = cv2.aruco.GridBoard(
    size=(5, 7),            # markersX, markersY
    markerLength=0.0315,      # in meters
    markerSeparation=0.006, # in meters
    dictionary=ARUCO_DICT
)

def setup_realsense():
    """Configure RealSense pipeline"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def get_extrinsic_matrix(rvec, tvec):
    """Convert rvec and tvec to a 4x4 extrinsic matrix"""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = tvec.flatten()
    return extrinsic_matrix

def process_frame(frame, camera_matrix, dist_coeffs):
    """Process frame for ArUco board detection and pose estimation"""
    # Detect markers
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame)
    
    if ids is not None:
        # Estimate board pose
        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, ids, BOARD, camera_matrix, dist_coeffs, None, None)
        extrinsic_matrix = get_extrinsic_matrix(rvec, tvec)
        print("Extrinsic Matrix:\n", extrinsic_matrix)
        
        if retval > 0:
            # Draw axis and markers
            frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Convert rotation to degrees
            rvec_deg = np.degrees(rvec).flatten()
            
            # Display pose info
            cv2.putText(frame, f"Position (m): X:{tvec[0][0]:.2f} Y:{tvec[1][0]:.2f} Z:{tvec[2][0]:.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Rotation (°): X:{rvec_deg[0]:.1f} Y:{rvec_deg[1]:.1f} Z:{rvec_deg[2]:.1f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw marker IDs
            for i in range(len(ids)):
                marker_id = ids[i][0]
                marker_center = np.mean(corners[i][0], axis=0).astype(int)
                cv2.putText(frame, str(marker_id), tuple(marker_center),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return frame, rvec, tvec
    
    # Fallback: show detected markers even if pose fails
    if ids is not None:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            marker_id = ids[i][0]
            marker_center = np.mean(corners[i][0], axis=0).astype(int)
            cv2.putText(frame, str(marker_id), tuple(marker_center),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame, None, None

def main():
    # Load calibration
    camera_matrix, dist_coeffs = load_camera_parameters(
        "/home/roman/spacecraft-pose-estimation-trajectories/scr/realsense/camera_calibration_1280_720.json",
        dist=True
    )
    
    # Setup RealSense
    pipeline = setup_realsense()
    
    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())
            
            # Process frame
            processed_frame, rvec, tvec = process_frame(frame, camera_matrix, dist_coeffs)
            
            # Display
            cv2.imshow("RealSense ArUco Board Pose", processed_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()