import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
input_video_path = '../../videos/1280_leo_v1/traj_1280px_15fps_v1_1.mp4'
output_video_path = '../../videos/inf_videos/1280_leo_v1_infval_traj_15fps_dist_1.mp4'
spacecraft_model_path = '../../trained_models/yolo_krn/sc_krn_leo_v1.pt'
marker_model_path = '/home/roman/spacecraft-pose-estimation-trajectories/runs/pose/train11/weights/best.pt'
json_labels_path = "../../data/labels/labels_sat_1280px_20kimgs_leo_2.json"
COVERAGE_RATIO_THRESHOLD = 0.01

# --- Load Annotations ---
with open(json_labels_path, 'r') as f:
    annotations = json.load(f)

# --- Helper: Convert pose to Euler angles (degrees) ---
def pose_to_euler_deg(pose):
    pose = np.array(pose)
    if len(pose) == 4:  # Quaternion [w, x, y, z]
        r = R.from_quat([pose[1], pose[2], pose[3], pose[0]])  # scipy uses [x, y, z, w]
        euler_rad = r.as_euler('zyx', degrees=False)
        return np.degrees(euler_rad)
    elif len(pose) == 3:  # Euler (radians)
        return np.degrees(pose)
    else:
        raise ValueError("Unknown pose format")

# --- Load Models ---
spacecraft_model = YOLO(spacecraft_model_path)
marker_model = YOLO(marker_model_path)
current_model = spacecraft_model
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
    if not ret or frame_idx >= len(annotations):
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
                model_switched = True
        except (IndexError, AttributeError):
            pass

    # --- Extract pose and translation for this frame ---
    annotation = annotations[frame_idx]
    pose = annotation['pose']
    translation = annotation['translation']

    # Convert pose to Euler angles (degrees)
    euler_deg = pose_to_euler_deg(pose)
    # Compute Euclidean distance
    distance = np.linalg.norm(translation)

    # Render and annotate frame
    rendered_frame = results[0].plot()
    cv2.putText(rendered_frame, f"Model: {'Marker' if model_switched else 'Spacecraft'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(rendered_frame, f"Distance: {distance:.2f} m", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(rendered_frame, f"Orientation (deg): Z={euler_deg[0]:.1f} Y={euler_deg[1]:.1f} X={euler_deg[2]:.1f}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)

    out.write(rendered_frame)
    frame_idx += 1

cap.release()
out.release()
print(f"Output saved to {output_video_path}")
