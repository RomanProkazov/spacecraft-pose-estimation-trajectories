import cv2
import json
import numpy as np
import os


def load_groundtruth(json_file, frame_idx):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data[499]


def visualize_image(image_path, json_file, frame_idx):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Load ground truth data for the specified frame
    groundtruth = load_groundtruth(json_file, frame_idx)
    if groundtruth is None:
        print(f"No ground truth data found for frame {frame_idx}")
        return

    # Draw bounding box
    bbox = groundtruth["bbox_xyxy"]  # [x_min, y_min, x_max, y_max]
    if bbox:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    else:
        print(f"No bounding box for frame {frame_idx}")

    # Draw keypoints
    keypoints = groundtruth.get("keypoints", None)
    if keypoints:
        keypoints = np.array(keypoints).reshape(-1, 2)
        for x, y in keypoints:
            cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
    else:
        print(f"No keypoints for frame {frame_idx}")

    # Display the image
    cv2.imshow("Ground Truth Visualization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "../../data_platform/images/img_0.png"  # Path to the input image
    json_file = "../../data_platform/labels/labels_plat_5kimgs.json"  # Path to the JSON file
    frame_idx = 0  # Index of the frame to visualize

    visualize_image(image_path, json_file, frame_idx)
