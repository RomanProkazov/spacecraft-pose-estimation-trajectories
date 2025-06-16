import cv2
from ultralytics import YOLO
import os
 
# --- CONFIGURATION ---
input_video_path = '../../videos/1280_leo_v1/traj_1280px_v1_1.mp4'  # Path to your input video
output_video_path = '../../videos/inf_videos/1280_leo_v1_infval_traj_1_inf_marker.mp4'  # Output video file
# model_path = '/home/roman/spacecraft-pose-estimation-trajectories/runs/pose/train3/weights/best.pt'  # Or your custom trained YOLOv8 pose model
model_path = '../../trained_models/yolo_krn/marker_krn_leo_v1.pt'  # Or your custom trained YOLOv8 pose model
 
# Load YOLOv8 Pose model
model = YOLO(model_path)
 
# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video {input_video_path}")
 
# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
 
# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # Run YOLOv8 Pose detection
    results = model.predict(frame, save=False, imgsz=640, conf=0.5)
 
    # Get rendered frame with keypoints drawn
    rendered_frame = results[0].plot()
 
    # Write to output video
    out.write(rendered_frame)
 
    # Optional: show live preview (comment out if running headless)
    # cv2.imshow("Pose Detection", rendered_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
 
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Saved output video to: {output_video_path}")