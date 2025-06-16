import cv2
import numpy as np
from ultralytics import YOLO
import os
import json

# --- CONFIGURATION ---
input_video_path = '../../videos/1280_leo_v1/traj_1280px_15fps_v1_1.mp4'
output_video_path = '../../videos/inf_videos/1280_leo_v1_infval_traj_15fps_1.mp4'
spacecraft_model_path = '../../trained_models/yolo_krn/sc_krn_leo_v1.pt'
marker_model_path = '/home/roman/spacecraft-pose-estimation-trajectories/runs/pose/train11/weights/best.pt' 
json_labels_path = "../../data/labels/labels_sat_1280px_20kimgs_leo_2.json" # Update with your marker model path
COVERAGE_RATIO_THRESHOLD = 0.01  # Switch when keypoints cover 70% of frame

with open(json_labels_path, 'r') as f:
    annotations = json.load(f)
pose = annotations[0]['pose']
translation = annotations[0]['translation']

# Load both models
spacecraft_model = YOLO(spacecraft_model_path)
marker_model = YOLO(marker_model_path)
current_model = spacecraft_model
model_switched = False

# Video setup
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
                model_switched = True
        except (IndexError, AttributeError):
            pass

    # Render and save
    rendered_frame = results[0].plot()
    cv2.putText(rendered_frame, f"Model: {'Marker' if model_switched else 'Spacecraft'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(rendered_frame)

cap.release()
out.release()
print(f"Output saved to {output_video_path}")
