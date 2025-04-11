import cv2
import numpy as np
import os
from tqdm import tqdm
import json

def create_distortion_maps(camera_matrix, dist_coeffs, image_shape):
    """
    Creates distortion maps for efficient image distortion.
    
    Args:
        camera_matrix: 3x3 numpy array
        dist_coeffs: [k1, k2, p1, p2, k3] format
        image_shape: (height, width) tuple
    
    Returns:
        map_x, map_y: distortion maps for cv2.remap()
    """
    h, w = image_shape[:2]
    
    # Create grid of pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    px_undist = np.vstack((x.flatten(), y.flatten())).T.astype(np.float32)
    
    # Convert to normalized coordinates
    px_undist_norm = cv2.undistortPoints(
        px_undist.reshape(-1, 1, 2), 
        camera_matrix, 
        None
    )
    
    # Apply distortion model
    px_dist_norm = cv2.projectPoints(
        px_undist_norm, 
        np.zeros(3), np.zeros(3),
        camera_matrix, 
        dist_coeffs
    )[0].reshape(-1, 2)
    
    # Create remap LUTs
    map_x = px_dist_norm[:, 0].reshape(h, w).astype(np.float32)
    map_y = px_dist_norm[:, 1].reshape(h, w).astype(np.float32)
    
    return map_x, map_y

def distort_images(input_dir, output_dir, camera_matrix, dist_coeffs):
    """
    Distorts all images in a directory using camera parameters.
    
    Args:
        input_dir: Path to undistorted images
        output_dir: Where to save distorted images
        camera_matrix: 3x3 numpy array
        dist_coeffs: distortion coefficients
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process first image to create maps
    test_img = cv2.imread(os.path.join(input_dir, image_files[0]))
    map_x, map_y = create_distortion_maps(camera_matrix, dist_coeffs, test_img.shape)
    
    for img_file in tqdm(image_files, desc="Distorting images"):
        img = cv2.imread(os.path.join(input_dir, img_file))
        distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, f"distorted_{img_file}"), distorted)

if __name__ == "__main__":

    # Path configuration
    input_directory = "../../data_3072px/images"  # Replace with your path
    output_directory = "../../data_3072px/distorted_images"    # Replace with your path
    json_data_path = "../../data_3072px/labels/meta_keypoints.json"  # Replace with your path   
    camera_sat_json = "../../data_3072px/labels/cam_sat.json"  # Replace with your path

    with open(json_data_path, 'r') as f:
        annotations = json.load(f)

    with open(camera_sat_json, 'r') as json_file:
        data = json.load(json_file)

    camera_matrix = np.array(data['camera_matrix_real'])
    dist_coeffs = np.array(data['distortion_coefficients']).reshape(5, 1)
   
    

    
    # Process all images
    map_x, map_y = distort_images(input_directory, output_directory, camera_matrix, dist_coeffs)
    
    print(f"Distorted images saved to {output_directory}")

    # Visual verification
    test_img = cv2.imread(os.path.join(input_directory, os.listdir(input_directory)[0]))
    distorted_img = cv2.remap(test_img, map_x, map_y, cv2.INTER_LINEAR)
    
    cv2.imshow("Original vs Distorted", np.hstack((test_img, distorted_img)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()