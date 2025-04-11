import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_groundtruth(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def visualize_groundtruth(video_path, json_file, output_video_path=None):
    """
    Visualize ground truth bounding boxes and keypoints on a video.

    Args:
        video_path (str): Path to the input video.
        json_file (str): Path to the JSON file containing ground truth data.
        output_video_path (str, optional): Path to save the output video. If None, the video is not saved.
    """
    # Load ground truth data
    groundtruth = load_groundtruth(json_file)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # Initialize video writer if output path is provided
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get ground truth for the current frame
        if str(frame_idx) in groundtruth:
            gt_data = groundtruth[str(frame_idx)]
            bbox = gt_data["bbox"]  # [x_min, y_min, x_max, y_max]
            keypoints = np.array(gt_data["keypoints"]).reshape(-1, 2)

            # Draw bounding box
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw keypoints
            for x, y in keypoints:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Display the frame
        cv2.imshow("Ground Truth Visualization", frame)

        # Write the frame to the output video if specified
        if out:
            out.write(frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "../../data/videos/floating_platform.mp4"
    json_file = "../../data/labels/groundtruth.json"
    output_video_path = "../../videos/groundtruth_visualization.mp4"

    visualize_groundtruth(video_path, json_file, output_video_path)
