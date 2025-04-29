import cv2
import torch
from ultralytics import YOLO
from efficientnet_pytorch import EfficientNet
from torch import nn
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
sys.path.append("../../scr/krn")
import config
import os
import json


def detect_object(image, detection_model):
    """Detect objects in the image using YOLO."""
    results = detection_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    return boxes


def preprocess_crop(image, bbox, target_size=(224, 224)):
    """Crop, pad, and resize the image to make it square."""
    x_min, y_min, x_max, y_max = map(int, bbox)
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Pad the cropped image to make it square
    orig_h, orig_w = cropped_image.shape[:2]
    if orig_h > orig_w:
        pad = (orig_h - orig_w) // 2
        padded_image = cv2.copyMakeBorder(cropped_image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        pad = (orig_w - orig_h) // 2
        padded_image = cv2.copyMakeBorder(cropped_image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Resize to target size
    resized_image = cv2.resize(padded_image, target_size)
    return resized_image, pad, orig_h, orig_w


def predict_keypoints(image, krn_model, device, num_kpts):
    """Predict keypoints using the keypoint regression network."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)

    krn_model.eval()
    with torch.no_grad():
        preds = krn_model(image_tensor).squeeze(0).cpu().numpy()
    keypoints = np.array([preds[0::2], preds[1::2]]).T
    return keypoints


def visualize_results(frame, bbox, keypoints, pad, orig_h, orig_w):
    """Visualize the bounding box and keypoints on the frame."""
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Adjust keypoints back to the original frame coordinates
    if orig_h > orig_w:
        # Padding was applied to the width
        scale = orig_h/224.0
        keypoints[:, 0] = (keypoints[:, 0] * scale - pad) + x_min
        keypoints[:, 1] = (keypoints[:, 1] * scale) + y_min
    else:
        # Padding was applied to the height
        scale = orig_w/224.0
        keypoints[:, 0] = (keypoints[:, 0] * scale) + x_min
        keypoints[:, 1] = (keypoints[:, 1] * scale - pad) + y_min

    # Draw bounding box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)

    # Draw keypoints
    for x, y in keypoints:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    return frame


def process_video(input_video_path, output_video_path, detection_model, krn_model, device, num_kpts, kpts_gt=None):
    """Process a video frame by frame, detect objects, predict keypoints, and save the output."""
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        bboxes = detect_object(frame, detection_model)
        if len(bboxes) > 0:
            # Process the first detected object
            bbox = bboxes[0]
            cropped_image, pad, orig_h, orig_w = preprocess_crop(frame, bbox)

            # Keypoint prediction
            keypoints = predict_keypoints(cropped_image, krn_model, device, num_kpts)

            # Visualize results
            frame = visualize_results(frame, bbox, keypoints, pad, orig_h, orig_w)

        # Initialize video writer if not already initialized
        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    if out:
        out.release()


if __name__ == "__main__":
    input_video_path = "../../videos/3072_2048/val/trajectories_videos/val_traj_1.mp4"
    output_video_path = "../../videos/3072_2048/val/detect_regress_videos/infer_val_traj_1.mp4"

    # with open("../../data_3072px/labels/labels_sat_27kimgs.json", "r") as f:
    #     gt_labels = json.load(f)

    # test_labels = gt_labels[24300:]

    # print(test_labels)
    # exit()
    # Load models
    detection_model = YOLO(config.ODN_MODEL_PATH)  # Object detection model
    krn_model = EfficientNet.from_pretrained("efficientnet-b0")
    krn_model._fc = nn.Linear(1280, config.NUM_KPTS_INF * 2)  # Assuming 16 keypoints
    krn_model.load_state_dict(torch.load(config.KRN_MODEL_PATH, map_location=config.DEVICE)["state_dict"], strict=False)
    krn_model = krn_model.to(config.DEVICE)

    # Process video
    process_video(input_video_path, output_video_path, detection_model, krn_model, config.DEVICE, config.NUM_KPTS_INF)
