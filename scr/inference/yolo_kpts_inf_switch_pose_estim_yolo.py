import cv2
import numpy as np
from ultralytics import YOLO
import json
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
input_video_path = '/home/roman/spacecraft-pose-estimation-trajectories/videos/1280_leo_v1/traj_1280px_15fps_v1_val1.mp4'
output_video_path = '../../videos/inf_videos/1280_leo_v1_infval_traj_15fps_pose_estim_2.mp4'
spacecraft_model_path = '../../trained_models/yolo_krn/sc_krn_leo_v1.pt'
marker_model_path = '/home/roman/spacecraft-pose-estimation-trajectories/runs/pose/train11/weights/best.pt'
cam_sat_json = "../../data/labels/cam_sat.json"
COVERAGE_RATIO_THRESHOLD = 0.01

with open(cam_sat_json, 'r') as json_file:
    data = json.load(json_file)
sat_model_points = np.array(data['sat_model_no_marker'])
sat_model_points[:, 0] *= -1
marker_model_points = np.array(data['marker_model'])
marker_model_points[:, 0] *= -1
camera_matrix = np.array(data['camera_matrix'])
dist_coeffs = np.zeros((4,1))  # Adjust if you have distortion

# --- Load YOLO Models ---
spacecraft_model = YOLO(spacecraft_model_path)
marker_model = YOLO(marker_model_path)
current_model = spacecraft_model
current_model_points = sat_model_points
model_switched = False

# --- Video Setup ---
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def calculate_coverage_ratio(keypoints, frame_shape):
    if keypoints.shape[0] == 0:
        return 0.0
    kps = keypoints.cpu().numpy()
    x_coords = kps[:, 0]
    y_coords = kps[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    bbox_area = (max_x - min_x) * (max_y - min_y)
    frame_area = frame_shape[1] * frame_shape[0]
    return bbox_area / frame_area

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference with current model
    results = current_model.predict(frame, imgsz=640, conf=0.5, verbose=False)

    # Model switching logic
    if not model_switched:
        try:
            kps = results[0].keypoints.xy[0]
            ratio = calculate_coverage_ratio(kps, frame.shape)
            if ratio > COVERAGE_RATIO_THRESHOLD:
                print(f"Switching to marker model at frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
                current_model = marker_model
                current_model_points = marker_model_points
                model_switched = True
        except (IndexError, AttributeError):
            pass

    # --- Keypoint Extraction and PnP Estimation ---
    try:
        kps = results[0].keypoints.xy[0].cpu().numpy()  # Shape: (N, 2)
        if kps.shape[0] == current_model_points.shape[0]:
            # Solve PnP with RANSAC
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                current_model_points, kps, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                R_mat, _ = cv2.Rodrigues(rvec)
                euler_deg = R.from_matrix(R_mat).as_euler('zyx', degrees=True)
                distance = np.linalg.norm(tvec)
                pose_str = f"Orientation (deg): Z={euler_deg[0]:.1f} Y={euler_deg[1]:.1f} X={euler_deg[2]:.1f}"
                dist_str = f"Distance: {distance:.2f} m"
            else:
                pose_str = "Pose estimation failed"
                dist_str = ""
        else:
            pose_str = "Keypoint mismatch"
            dist_str = ""
    except Exception as e:
        pose_str = "Detection failed"
        dist_str = ""

    # Render and annotate frame
    rendered_frame = results[0].plot()
    cv2.putText(rendered_frame, f"Model: {'Marker' if model_switched else 'Spacecraft'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(rendered_frame, dist_str, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(rendered_frame, pose_str, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)

    out.write(rendered_frame)
    frame_idx += 1

cap.release()
out.release()
print(f"Output saved to {output_video_path}")
