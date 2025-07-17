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
from scipy.stats import norm


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


def enable_mc_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def predict_keypoints_with_uncertainty(image, model, device, num_kpts, T=30):
    """Predict keypoints and estimate uncertainty using MC Dropout."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)

    model.eval()
    enable_mc_dropout(model)  # Enable dropout during inference

    predictions = []
    with torch.no_grad():
        for _ in range(T):
            preds = model(image_tensor)  # shape: [1, num_kpts*2]
            predictions.append(preds)

    predictions = torch.stack(predictions, dim=0)  # [T, 1, num_kpts*2]
    mean_preds = predictions.mean(dim=0).squeeze(0)  # [num_kpts*2]
    std_preds = predictions.std(dim=0).squeeze(0)    # [num_kpts*2]

    mean_kpts = np.stack([mean_preds[0::2].cpu().numpy(), mean_preds[1::2].cpu().numpy()], axis=1)
    std_kpts = np.stack([std_preds[0::2].cpu().numpy(), std_preds[1::2].cpu().numpy()], axis=1)

    return mean_kpts, std_kpts

import torch
import numpy as np
from scipy.stats import norm

def predict_keypoints_with_uncertainty_confidence(image, krn_model, device, num_kpts, n_passes=50):
    krn_model.train()  # Enable dropout at inference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)

    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = krn_model(image_tensor).squeeze(0).cpu().numpy()
            preds.append(out)

    preds = np.array(preds)  # [n_passes, 2 * num_kpts]

    # Separate X and Y
    x_preds = preds[:, 0::2]
    y_preds = preds[:, 1::2]

    # Compute mean and std per keypoint
    x_mean, x_std = x_preds.mean(axis=0), x_preds.std(axis=0)
    y_mean, y_std = y_preds.mean(axis=0), y_preds.std(axis=0)

    keypoints_mean = np.stack([x_mean, y_mean], axis=1)  # shape: [num_kpts, 2]
    keypoints_std = np.stack([x_std, y_std], axis=1)     # shape: [num_kpts, 2]

    # Compute spatial uncertainty (in pixels)
    spatial_uncertainty = np.linalg.norm(keypoints_std, axis=1)  # Euclidean uncertainty per keypoint

    # 95% confidence interval per coordinate (assuming normal dist)
    z = norm.ppf(0.975)
    ci_95 = z * keypoints_std  # shape: [num_kpts, 2]

    # Print summary
    for i in range(num_kpts):
        print(f"Keypoint {i+1}:")
        print(f"  Mean (x, y): ({keypoints_mean[i, 0]:.2f}, {keypoints_mean[i, 1]:.2f})")
        print(f"  Std dev (x, y): ({keypoints_std[i, 0]:.2f}, {keypoints_std[i, 1]:.2f})")
        print(f"  95% CI (x ± zσ, y ± zσ): (±{ci_95[i, 0]:.2f}, ±{ci_95[i, 1]:.2f})")
        print(f"  Spatial uncertainty: {spatial_uncertainty[i]:.2f} px\n")

    return keypoints_mean, spatial_uncertainty




def visualize_keypoints_on_crop(cropped_image, keypoints, uncertainties=None):
    """Visualize keypoints on the padded 224x224 crop."""
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(cropped_image, (int(x), int(y)), 2, (0, 255, 0), -1)
        if uncertainties is not None:
            std = uncertainties[i]
            cv2.ellipse(
                cropped_image,
                (int(x), int(y)),
                (int(std[0] * 1), int(std[1] * 1)),  # scaled for visibility
                0, 0, 360,
                (0, 255, 255), 1
            )

    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title("Keypoints with Uncertainty")
    plt.show()



if __name__ == "__main__":
    # Load models
    detection_model = YOLO(config.ODN_MODEL_PATH)  # Object detection model
    krn_model = EfficientNet.from_pretrained("efficientnet-b0")
    krn_model._fc = nn.Linear(1280, config.NUM_KPTS_INF * 2)
    krn_model.load_state_dict(torch.load(config.KRN_MODEL_PATH, map_location=config.DEVICE)["state_dict"], strict=False)
    krn_model = krn_model.to(config.DEVICE)

    # Input image
    idx = 0
    image_names = sorted(os.listdir(config.IMG_DIR), key=lambda x: int(x[4:-4]))
    images_list = [os.path.join(config.IMG_DIR, filename) for filename in image_names]
    original_image = cv2.imread(images_list[idx])

    # Object detection
    bboxes = detect_object(original_image, detection_model)
    if len(bboxes) == 0:
        print("No objects detected.")
    else:
        bbox = bboxes[0]
        cropped_image = preprocess_crop(original_image, bbox)
        # keypoints, uncertainties = predict_keypoints_with_uncertainty(
        #     cropped_image, krn_model, config.DEVICE, num_kpts=config.NUM_KPTS_INF, T=30
        # )

        keypoints, uncertainties = predict_keypoints_with_uncertainty_confidence(
        cropped_image, krn_model, config.DEVICE, num_kpts=config.NUM_KPTS_INF, n_passes=30
        )
        print(uncertainties)
        # print("Keypoint uncertainties (std dev per (x,y)):")
        # for i, (std_x, std_y) in enumerate(uncertainties):
        #     print(f"Keypoint {i+1}: std_x = {std_x:.2f}, std_y = {std_y:.2f}, magnitude = {np.sqrt(std_x**2 + std_y**2):.2f}")
        # mean_uncertainty = np.mean(np.linalg.norm(uncertainties, axis=1))
        # print(f"Mean spatial uncertainty per keypoint: {mean_uncertainty:.2f} pixels")

        # visualize_keypoints_on_crop(cropped_image, keypoints, uncertainties)
