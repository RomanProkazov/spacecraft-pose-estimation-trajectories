import PySpin
import cv2
import numpy as np
from cv2 import aruco
import screeninfo
import json


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
    dist_coeffs = np.array(cam_dist['distortion_coefficients'])
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
    # Initialize system and camera
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    
    if cam_list.GetSize() == 0:
        print("No cameras detected.")
        system.ReleaseInstance()
        return
    
    cam = cam_list.GetByIndex(0)
    cam.Init()
    
    # Setup ArUco detection and pose estimation
    marker_length = 0.064  
    detector = setup_aruco()
    camera_matrix, dist_coeffs = load_camera_parameters('../../data_3072px/labels/cam_sat.json')
    # dist_coeffs = np.zeros(5)
    
    # Start acquisition
    cam.BeginAcquisition()
    print("Streaming with ArUco pose estimation. Press 'q' to exit.")
    
    # Get screen dimensions
    screen = screeninfo.get_monitors()[0]
    screen_width, screen_height = screen.width, screen.height
    
    while True:
        image = cam.GetNextImage()
        
        if image.IsIncomplete():
            print(f"Image incomplete with status: {image.GetImageStatus()}")
            image.Release()
            continue
        
        # Convert to OpenCV format
        img_data = image.GetNDArray()
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_BAYER_BG2BGR)
        
        # Detect markers and estimate pose
        img_bgr = detect_and_estimate_pose(img_bgr, detector, 
                                         camera_matrix, dist_coeffs, 
                                         marker_length)
        
        # Resize for display
        h, w = img_bgr.shape[:2]
        scale = min(screen_width/w, screen_height/h)
        img_resized = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
        
        # Display stream
        cv2.imshow("ArUco Pose Estimation", img_resized)
        image.Release()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cam.EndAcquisition()
    cam.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()
    cv2.destroyAllWindows()
    print("Streaming stopped.")

if __name__ == "__main__":
    main()