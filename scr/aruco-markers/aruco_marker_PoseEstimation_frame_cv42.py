import cv2
import numpy as np
from cv2 import aruco

def detect_and_draw_markers(image_path):
    """
    Simple ArUco marker detection and visualization
    Args:
        image_path: Path to the input image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize ArUco dictionary and parameters (OpenCV 4.2 style)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    
    # Detect markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Draw detected markers
    if ids is not None:
        # Draw all markers
        image = aruco.drawDetectedMarkers(image.copy(), corners, ids)
        
        # Print detection info
        print(f"Detected {len(ids)} markers:")
        for i, marker_id in enumerate(ids):
            print(f"Marker ID: {marker_id[0]} at position {np.mean(corners[i][0], axis=0)}")
    
    # Display result
    cv2.imshow("Detected Markers", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    # Example usage
    image_path = "/home/roman/spacecraft-pose-estimation-trajectories/data_real/lux_sat_data_real_v1/frame_dark_0224.jpg"  # Change this to your image path
    detect_and_draw_markers(image_path)