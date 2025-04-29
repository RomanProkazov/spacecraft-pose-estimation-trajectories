from efficientnet_pytorch import EfficientNet
from torch import nn
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
import sys
sys.path.append("../../scr/krn")
import config 


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
    return resized_image


def predict_keypoints(image, krn_model, device, num_kpts):
    """Predict keypoints using the keypoint regression network."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[-0.6018, -0.6015, -0.6016], std=[0.5889, 0.5888, 0.5888]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)

    krn_model.eval()
    with torch.no_grad():
        preds = krn_model(image_tensor).squeeze(0).cpu().numpy()
    keypoints = np.array([preds[0::2], preds[1::2]]).T
    return keypoints


def visualize_keypoints_on_crop(cropped_image, keypoints):
    """Visualize keypoints on the padded 224x224 crop."""
    for x, y in keypoints:
        cv2.circle(cropped_image, (int(x), int(y)), 1, (0, 255, 0), -1)

    # Display the image
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title("Keypoints on Cropped Image")
    plt.show()


if __name__ == "__main__":
    # Load models
    detection_model = YOLO(config.ODN_MODEL_PATH)  # Object detection model
    krn_model = EfficientNet.from_pretrained("efficientnet-b0")
    krn_model._fc = nn.Linear(1280, config.NUM_KPTS_INF * 2)
    krn_model.load_state_dict(torch.load(config.KRN_MODEL_PATH, map_location=config.DEVICE)["state_dict"], strict=False)
    krn_model = krn_model.to(config.DEVICE)

    # Input image
    idx = 492
    image_names = sorted(os.listdir(config.TEST_IMG_DIR), key=lambda x: int(x[4:-4]))
    images_list = [os.path.join(config.TEST_IMG_DIR, filename) for filename in image_names]
    original_image = cv2.imread(images_list[idx])

    # Object detection
    bboxes = detect_object(original_image, detection_model)
    if len(bboxes) == 0:
        print("No objects detected.")
    else:
        # Process the first detected object
        bbox = bboxes[0]
        cropped_image = preprocess_crop(original_image, bbox)
        keypoints = predict_keypoints(cropped_image, krn_model, config.DEVICE, num_kpts=config.NUM_KPTS)
        visualize_keypoints_on_crop(cropped_image, keypoints)