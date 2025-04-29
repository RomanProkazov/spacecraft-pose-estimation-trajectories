import cv2
import torch
from ultralytics import YOLO
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scr.krn import config as config


def detect_object_video(model_path, video_path, output_path):

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    model = YOLO(model_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if video ends
  
        results = model(frame)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0].item()  # Confidence score
                
                # Draw rectangle & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Spacecraft {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write processed frame to output video
        out.write(frame)

        # # Display (optional)
        # cv2.imshow("YOLOv11n Spacecraft Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        #     break

    # Release resources
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    print(f"Processed video saved as {output_path}")



if __name__ == "__main__":
    seg_model = config.ODN_MODEL_PATH
    input_video_path = config.INPUT_VIDEO_PATH
    output_video_path = config.OUTPUT_VIDEO_PATH_SEG

    detect_object_video(seg_model, input_video_path, output_video_path)