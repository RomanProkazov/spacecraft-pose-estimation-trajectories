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
    px_undist = np.vstack((x.flatten(), y.flatten())).T.reshape(-1, 1, 2).astype(np.float32)
    
    # Convert to normalized coordinates
    px_undist_norm = cv2.undistortPoints(
        px_undist,
        camera_matrix, 
        None
    )
    
    # Reshape for projectPoints (needs 3D points)
    object_points = np.zeros((px_undist_norm.shape[0], 1, 3), dtype=np.float32)
    object_points[:, 0, :2] = px_undist_norm.reshape(-1, 2)
    
    # Apply distortion model
    px_dist_norm, _ = cv2.projectPoints(
        object_points, 
        np.zeros(3), np.zeros(3),
        camera_matrix, 
        dist_coeffs
    )
    
    # Create remap LUTs
    map_x = px_dist_norm[:, 0, 0].reshape(h, w).astype(np.float32)
    map_y = px_dist_norm[:, 0, 1].reshape(h, w).astype(np.float32)
    
    return map_x, map_y

def distort_images(input_dir, output_dir, camera_matrix, dist_coeffs):
    """
    Distorts all images in a directory using camera parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return None, None
    
    # Get image dimensions from first image
    test_img = cv2.imread(os.path.join(input_dir, image_files[0]))
    if test_img is None:
        print(f"Failed to load test image: {image_files[0]}")
        return None, None
    
    map_x, map_y = create_distortion_maps(camera_matrix, dist_coeffs, test_img.shape)
    
    for img_file in tqdm(image_files, desc="Distorting images"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping corrupt image: {img_file}")
            continue
            
        distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(os.path.join(output_dir, f"distorted_{img_file}"), distorted)
    
    return map_x, map_y

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
    
    # Process images
    map_x, map_y = distort_images(input_directory, output_directory, 
                                camera_matrix, dist_coeffs)
    
    if map_x is not None:
        print(f"\nSuccess! Distorted images saved to {output_directory}")
        
        # Verification
        test_img = cv2.imread(os.path.join(input_directory, os.listdir(input_directory)[0]))
        if test_img is not None:
            distorted = cv2.remap(test_img, map_x, map_y, cv2.INTER_LINEAR)
            cv2.imshow("Original (Left) vs Distorted (Right)", np.hstack((test_img, distorted)))
            cv2.waitKey(3000)
            cv2.destroyAllWindows()