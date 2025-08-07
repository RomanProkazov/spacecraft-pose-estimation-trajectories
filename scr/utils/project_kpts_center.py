# from mathutils import Quaternion
import cv2
from cv2 import aruco
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import tf.transformations 



# def quaternion_to_rvec(quat):
#     quat = Quaternion(quat)  
#     rot_matrix = np.array(quat.to_matrix())  
#     rvec, _ = cv2.Rodrigues(rot_matrix) 
#     return rvec

def quaternion_to_rvec(quat):
    quat = np.array(quat, dtype=np.float64)
    rotation = R.from_quat(np.roll(quat, -1))  # Convert [w, x, y, z] to [x, y, z, w]
    rot_matrix = rotation.as_matrix()
    rvec, _ = cv2.Rodrigues(rot_matrix)
    return rvec

def quat_tf_to_rvec(quat):
    quat_xyzw = np.roll(quat, -1)
    rot_matrix = tf.transformations.quaternion_matrix(quat_xyzw)
    print(rot_matrix)
    rvec, _ = cv2.Rodrigues(rot_matrix[:3, :3])
    return rvec


def load_camera_matrix_sat_model(camera_sat_json, object='sat_model_center'):
    with open(camera_sat_json, 'r') as json_file:
        data = json.load(json_file)
    sat_model, camera_matrix, dist = np.array(data[object]), np.array(data['camera_matrix']), np.array(data['dist']) 
    return camera_matrix, dist, sat_model


def load_labels(json_data_path):
    with open(json_data_path, 'r') as f:
        labels = json.load(f)
    return labels


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

    if use_distortion:
        distCoeffs = dist
    else:
        distCoeffs = np.zeros((5, 1), dtype=np.float32)

    # r_vec = quaternion_to_rvec(q_gt)
    r_vec = quat_tf_to_rvec(q_gt)
    image_points, _ = cv2.projectPoints(sat_model, r_vec, t_gt, cmt, distCoeffs)
    image_points = image_points.reshape(-1, 2)
    # print(f'GT image_points: {image_points}')

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
    json_labels_path = "/home/roman/spacecraft-pose-estimation-trajectories/data/labels/meta_keypoints.json"
    image_path = "/home/roman/spacecraft-pose-estimation-trajectories/image_00001.jpg"
    annotations = load_labels(json_labels_path)
    
    idx=0
    labels = annotations[idx]
  
    # Ground truth data
    q_gt =labels['pose']
    t_gt = labels['translation']
    print(q_gt, t_gt)
  
    project_kpts_from_image(image_path, q_gt, t_gt,
                            camera_sat_json=camera_sat_model,
                            use_distortion=False,
                            look_marker=False)

