import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scr.krn import config as config


def segment_object_video(model_path, video_path, output_path):
    """
    Segment objects in a video using YOLO segmentation and save the output.

    Args:
        model_path (str): Path to the YOLO segmentation model.
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video.
    """
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

        # Draw segmentation masks and bounding boxes
        for result in results:
            for mask, box in zip(result.masks.data.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
                # Resize mask to match the frame dimensions
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_resized = mask_resized.astype(bool)

                # Apply a random color to the mask
                color = np.random.randint(0, 255, 3, dtype=np.uint8)
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask_resized] = color

                # Blend the mask with the frame
                frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)

        # Write processed frame to output video
        out.write(frame)

        # # Display (optional)
        # cv2.imshow("YOLO Segmentation", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        #     break

    # Release resources
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    print(f"Processed video saved as {output_path}")


if __name__ == "__main__":
    seg_model = config.SEG_MODEL_PATH  # Path to the YOLO segmentation model
    input_video_path = config.INPUT_VIDEO_PATH  # Path to the input video
    output_video_path = config.OUTPUT_VIDEO_PATH_SEG  # Path to save the output video

    segment_object_video(seg_model, input_video_path, output_video_path)
