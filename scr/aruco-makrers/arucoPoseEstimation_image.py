import cv2
import numpy as np
from cv2 import aruco


def load_camera_parameters(camera_matrix_path, distortion_coefficients_path):
    camera_matrix = np.load(camera_matrix_path)
    distortion_coefficients = np.load(distortion_coefficients_path)
    return camera_matrix, distortion_coefficients


def estimate_pose(image_path, camera_matrix, distortion_coefficients, marker_length):
    """
    Estimate the pose of an ArUco marker in an image.

    Args:
        image_path (str): Path to the input image.
        camera_matrix (numpy.ndarray): Camera matrix.
        distortion_coefficients (numpy.ndarray): Distortion coefficients.
        marker_length (float): Length of the marker's side in meters.

    Returns:
        None
    """
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the predefined dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw detected markers
        aruco.drawDetectedMarkers(image, corners, ids)

        # Estimate pose for each marker
        for i, corner in enumerate(corners):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, distortion_coefficients)

            # Draw axis for each marker
            aruco.drawAxis(image, camera_matrix, distortion_coefficients, rvec, tvec, marker_length * 0.5)

            # Print pose information
            print(f"Marker ID: {ids[i][0]}")
            print(f"Rotation Vector (rvec):\n{rvec}")
            print(f"Translation Vector (tvec):\n{tvec}")

    else:
        print("No markers detected.")

    # Display the image
    cv2.imshow("ArUco Pose Estimation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Paths to input files
    image_path = "../../data/images/aruco_image.jpg"  # Replace with your image path
    camera_matrix_path = "../../data/calibration/camera_matrix.npy"  # Replace with your camera matrix file path
    distortion_coefficients_path = "../../data/calibration/dist_coeffs.npy"  # Replace with your distortion coefficients file path

    # Marker length in meters
    marker_length = 0.05  # Replace with the actual marker size

    # Load camera parameters
    camera_matrix, distortion_coefficients = load_camera_parameters(camera_matrix_path, distortion_coefficients_path)

    # Estimate pose
    estimate_pose(image_path, camera_matrix, distortion_coefficients, marker_length)
