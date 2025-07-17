import cv2
import numpy as np
import pyrealsense2 as rs
import json
from cv2 import aruco

# ====== CONFIGURATION ======
ARUCO_DICT = aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7        # Number of squares vertically
SQUARES_HORIZONTALLY = 5      # Number of squares horizontally
SQUARE_LENGTH = 0.037        # 11.5mm in meters (must match printed board)
MARKER_LENGTH = 0.018         # 8mm in meters (must match printed board)
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
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    board = aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    return board, detector




def process_frame(frame, board, detector, camera_matrix, dist_coeffs):
    # Detect markers
    corners, ids, _ = detector.detectMarkers(frame)
    
    if ids is None:
        return frame, None, None
    
    # Interpolate Charuco corners
    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        corners, ids, frame, board)
    
    if not retval:
        output = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        return output, None, None
    
    # Estimate pose
    success, rvec, tvec = aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board,
        camera_matrix, dist_coeffs, None, None)
    
    if not success:
        output = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        return output, None, None
    
    # Draw results
    output = frame.copy()
    output = cv2.aruco.drawDetectedMarkers(output, corners, ids)
    cv2.drawFrameAxes(output, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
    
    return output, rvec, tvec

# ====== MAIN LOOP ======
def main():
    # Load calibration
    with open(CALIBRATION_JSON) as f:
        calib = json.load(f)
    camera_matrix = np.array(calib['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(calib['dist'], dtype=np.float32)
    
    # Initialize detector and RealSense
    board, detector = initialize_detector()
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
            result, rvec, tvec = process_frame(frame, board, detector, camera_matrix, dist_coeffs)
            
            # Display info if pose found
            if rvec is not None:
                rvec_deg = np.degrees(rvec).flatten()
                cv2.putText(result, f"Pos: {tvec.flatten()[:3].round(3)}m", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(result, f"Rot: {rvec_deg.round(1)}deg", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # Show result
            cv2.imshow("RealSense Charuco Tracking", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()