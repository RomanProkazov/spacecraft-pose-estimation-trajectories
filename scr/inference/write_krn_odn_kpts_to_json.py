import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("../../scr/krn")
import config
from pathlib import Path
import torch
from torchvision import transforms
from detect_and_regress_frame import (detect_object,
                                      preprocess_crop,
                                      predict_keypoints,
                                      visualize_results)
from efficientnet_pytorch import EfficientNet
from torch import nn
from ultralytics import YOLO


def load_krn_model(model_path, num_kpts_inf):
    krn_model = EfficientNet.from_pretrained("efficientnet-b0")
    krn_model._fc = nn.Linear(1280, num_kpts_inf * 2)  # Assuming 16 keypoints
    krn_model.load_state_dict(torch.load(model_path, weights_only=True)["state_dict"], strict=False)
    krn_model = krn_model.to(config.DEVICE)
    krn_model.eval()
    return krn_model


def unscale_keypoints(bbox, keypoints, pad, orig_h, orig_w):
    x_min, y_min, x_max, y_max = map(int, bbox)
    if orig_h > orig_w:
        scale = orig_h / 224.0
        keypoints[:, 0] = (keypoints[:, 0] * scale - pad) + x_min  
        keypoints[:, 1] = (keypoints[:, 1] * scale) + y_min        
    else:
        scale = orig_w / 224.0
        keypoints[:, 0] = (keypoints[:, 0] * scale) + x_min        
        keypoints[:, 1] = (keypoints[:, 1] * scale - pad) + y_min

    return keypoints


def main():
    detection_model = YOLO(config.ODN_MODEL_PATH)
    krn_model = load_krn_model(config.KRN_MODEL_PATH, config.NUM_KPTS_INF)

    image_folder_path = config.TEST_IMG_DIR
    json_path = config.LABELS_JSON

    with open(json_path, 'r') as f:
        annotations = json.load(f)
    annotations = annotations[7000:]

    image_path_list = sorted([image for image in Path(image_folder_path).rglob('*.jpg')], key=lambda x: int(x.stem.split('_')[-1]))
    image_path_list = image_path_list[7000:] 

    for idx in tqdm(range(len(image_path_list))):
        image_path = image_path_list[idx]
        image = cv2.imread(str(image_path))
        bbox = detect_object(image, detection_model)
        if len(bbox) == 0:
            print(f"No objects detected in {image_path}")
            continue

        cropped_image, pad, orig_h, orig_w = preprocess_crop(image, bbox[0], target_size=(224, 224))
        keypoints = predict_keypoints(cropped_image, krn_model, config.DEVICE)

        # visualize_results(image, bbox[0], keypoints, pad, orig_h, orig_w)
        # exit()
        keypoints = unscale_keypoints(bbox[0], keypoints, pad, orig_h, orig_w)

        annotations[idx]['kpts_preds'] = keypoints.tolist()
    
    with open(config.LABELS_JSON_PREDS, 'w') as f:
        json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    main()
