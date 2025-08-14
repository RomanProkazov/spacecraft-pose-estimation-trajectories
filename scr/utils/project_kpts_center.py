# from mathutils import Quaternion
import cv2
from cv2 import aruco
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import tf.transformations 



def quaternion_to_rvec(quat):
    quat = np.array(quat, dtype=np.float64)
    rotation = R.from_quat(np.roll(quat, -1))  # Convert [w, x, y, z] to [x, y, z, w]
    rot_matrix = rotation.as_matrix()
    rvec, _ = cv2.Rodrigues(rot_matrix)
    return rvec

def quat_tf_to_rvec(quat):
    quat_xyzw = np.roll(quat, -1)
    rot_matrix = tf.transformations.quaternion_matrix(quat_xyzw)
    rvec, _ = cv2.Rodrigues(rot_matrix[:3, :3])
    return rvec


def load_camera_matrix_sat_model(camera_sat_json, object='sat_model_center'):
    with open(camera_sat_json, 'r') as json_file:
        data = json.load(json_file)
    sat_model, camera_matrix, dist = np.array(data[object]), np.array(data['camera_matrix_rs_1280']), np.array(data['dist']) 
    return camera_matrix, dist, sat_model


def load_labels(json_data_path):
    with open(json_data_path, 'r') as f:
        labels = json.load(f)
    return labels


def rotate_keypoints_projection(image_points, angle_deg, axis='z', origin=None):
    """
    Rotate projected keypoints by a specified angle around a given axis
    
    Args:
        image_points: numpy array of shape (N, 2) containing 2D keypoints
        angle_deg: rotation angle in degrees
        axis: rotation axis ('x', 'y', or 'z')
        origin: rotation center point (if None, uses centroid of points)
    
    Returns:
        Rotated keypoints as numpy array of shape (N, 2)
    """
    # Convert to homogeneous coordinates (add z=0 and w=1)
    points = np.column_stack([image_points, np.zeros(len(image_points)), np.ones(len(image_points))])
    
    # Set rotation center
    if origin is None:
        origin = np.mean(image_points, axis=0)
    
    # Create transformation matrix
    angle_rad = np.radians(angle_deg)
    
    # Translation matrix to origin
    T1 = np.eye(4)
    T1[:2, 3] = -origin
    
    # Rotation matrix
    if axis == 'x':
        R_mat = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
            [0, np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        R_mat = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
    else:  # default z-axis
        R_mat = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    # Translation matrix back from origin
    T2 = np.eye(4)
    T2[:2, 3] = origin
    
    # Combined transformation matrix
    transform = T2 @ R_mat @ T1
    
    # Apply transformation to each point
    rotated_points = np.array([transform @ point for point in points])
    
    # Return only x,y coordinates
    return rotated_points[:, :2]


def project_kpts_from_image(image_path,
                            quaternion, translation,
                            camera_sat_json,
                            use_distortion=False,
                            look_marker=False):                           
    
    cmt, dist, sat_model = load_camera_matrix_sat_model(camera_sat_json)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Ground truth data
    q_gt =np.array(quaternion)
    
    t_gt = np.array(translation)
   
    # t_gt = np.array([t_gt[0][0], t_gt[0][1], t_gt[0][2]])
    # q_gt =np.array([1, 0, 0, 0])
    # t_gt = np.array([[-0.01280525,  0.08241902,  0.79136958]])
    # t_gt = np.array([[0,  0,  2.79]])
    # print(q_gt, t_gt)
    # t_gt_init = np.array([[-0.01280525,  0.08241902,  0.79136958]])

    if use_distortion:
        distCoeffs = dist
    else:
        distCoeffs = np.zeros((5, 1), dtype=np.float32)

    r_vec = quaternion_to_rvec(q_gt)
    # r_vec = quat_tf_to_rvec(q_gt)
    image_points, _ = cv2.projectPoints(sat_model, r_vec, t_gt, cmt, distCoeffs)
    image_points = image_points.reshape(-1, 2)
    # print(f'GT image_points: {image_points}')
    # image_points = rotate_keypoints_projection(image_points, angle_deg=90, axis='z', origin=None)

    for point in image_points:
        x, y = round(point[0]), round(point[1])
        cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=5)  # red dots

    if  look_marker:
        image_rgb = cv2.imread(image_path)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
    
        # Detect markers
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        image_points = corners[0][0]
        image_points = image_points[[0, 3, 2, 1]]
        # print(f'Aruco image points: {image_points}')

        for point in image_points:
            x, y = round(point[0]), round(point[1])
            cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=5)  # red dots

    cv2.imshow("Projected Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    camera_sat_model = "/home/roman/spacecraft-pose-estimation-trajectories/cam_sat.json"
    json_labels_path = "/home/roman/Desktop/ros_bags/extracted_ros_data/poses/mockup_in_cam_pose.json"
    image_path = "/home/roman/Desktop/ros_bags/extracted_ros_data/images/_camera_color_image_raw_1754574770204414875.jpg"
    annotations = load_labels(json_labels_path)
    
    idx=0
    labels = annotations[idx]
  
    # Ground truth data
    q_gt =labels['pose']
    t_gt = np.array([labels['translation']])
    # print(q_gt, t_gt)
  
    project_kpts_from_image(image_path, q_gt, t_gt,
                            camera_sat_json=camera_sat_model,
                            use_distortion=True,
                            look_marker=False)

