import PySpin
import cv2
import numpy as np
from cv2 import aruco
import screeninfo
import json
import pyrealsense2 as rs


def setup_aruco():
    """Initialize ArUco detection and pose estimation parameters"""
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detector_params)   
    return detector

def load_camera_parameters(camera_matrix_path):
    with open(camera_matrix_path, 'r') as f:
        cam_dist = json.load(f)
    camera_matrix = np.array(cam_dist['camera_matrix'])
    dist_coeffs = np.array(cam_dist['dist'])
    return camera_matrix, dist_coeffs


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


def detect_and_estimate_pose(frame, detector, camera_matrix, dist_coeffs, marker_length):
    """Detect markers and estimate their 6DOF pose"""
    corners, ids, _ = detector.detectMarkers(frame)
    
    if ids is not None:
        # Draw detected markers
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Estimate pose for each marker
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
            corners, 
            marker_length, 
            camera_matrix, 
            dist_coeffs
        )
        
        # Draw axis and display pose info
        for i in range(len(ids)):
            frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                                    rvecs[i], tvecs[i], marker_length/2)
            
            # Display position and orientation
            pos_text = f"Pos: {tvecs[i][0].round(2)}m"
            rot_text = f"Rot: {np.degrees(rvecs[i][0]).round(1)}deg"
            cv2.putText(frame, pos_text, (10, 30+i*60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, rot_text, (10, 60+i*60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    return frame

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    # Setup ArUco detection and pose estimation
    marker_length = 0.08  
    detector = setup_aruco()
    camera_matrix, dist_coeffs = load_camera_parameters('../../data_640px/labels/cam_sat.json')
    # dist_coeffs = np.zeros(5)
    

    
    try:
        while True:
            # Wait for a coherent color frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            # Convert image to numpy array
            img_bgr = np.asanyarray(color_frame.get_data())
            
            # Detect markers and estimate pose
            img_bgr = detect_and_estimate_pose(img_bgr, detector, 
                                            camera_matrix, dist_coeffs, 
                                            marker_length)

            
            # Display stream
            cv2.imshow("ArUco Pose Estimation", img_bgr)
          
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
            # Convert image to numpy array