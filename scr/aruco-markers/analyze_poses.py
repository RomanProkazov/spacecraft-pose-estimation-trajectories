import json
import numpy as np
import cv2

def rvec_to_euler(rvec):
    """Convert rotation vector to Euler angles (XYZ convention) in degrees"""
    R, _ = cv2.Rodrigues(np.array(rvec))
    
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    
    return np.degrees([x, y, z])

def compare_poses(rvec1, tvec1, rvec2, tvec2):
    """Compare two poses and return difference metrics"""
    R1, _ = cv2.Rodrigues(np.array(rvec1))
    R2, _ = cv2.Rodrigues(np.array(rvec2))
    
    # Rotation difference
    R_diff = R1 @ R2.T
    rvec_diff, _ = cv2.Rodrigues(R_diff)
    angle_diff_deg = np.degrees(np.linalg.norm(rvec_diff))
    
    # Translation difference
    t_diff = np.linalg.norm(np.array(tvec1) - np.array(tvec2))
    
    return {
        "rotation_angle_diff_deg": float(angle_diff_deg),
        "translation_diff_m": float(t_diff),
        "relative_rotation": rvec_diff.flatten().tolist()
    }

def analyze_poses(input_file='inference_data/pose_data.json', output_file='inference_data/pose_analysis.json'):
    # Load the input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each pose
    for pose_type in data['poses']:
        pose = data['poses'][pose_type]
        
        # Convert rvec to rotation matrix
        R, _ = cv2.Rodrigues(np.array(pose['rvec']))
        pose['rotation_matrix'] = R.tolist()
        
        # Convert to Euler angles
        pose['euler_angles'] = rvec_to_euler(pose['rvec']).tolist()
    
    # Perform comparisons
    poses = data['poses']
    data['comparisons']['charuco_vs_board'] = compare_poses(
        poses['charuco']['rvec'], poses['charuco']['tvec'],
        poses['aruco_board']['rvec'], poses['aruco_board']['tvec']
    )
    
    data['comparisons']['charuco_vs_marker'] = compare_poses(
        poses['charuco']['rvec'], poses['charuco']['tvec'],
        poses['aruco_marker']['rvec'], poses['aruco_marker']['tvec']
    )
    
    data['comparisons']['board_vs_marker'] = compare_poses(
        poses['aruco_board']['rvec'], poses['aruco_board']['tvec'],
        poses['aruco_marker']['rvec'], poses['aruco_marker']['tvec']
    )
    
    # Save the analyzed data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    analyze_poses()