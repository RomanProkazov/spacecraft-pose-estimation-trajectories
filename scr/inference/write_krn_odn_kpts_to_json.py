import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("../../scr/krn")
import config
from PIL import Image
import torch
from torchvision import transforms
from detect_and_regress_frame import (detect_object,
                                      preprocess_crop,
                                      predict_keypoints)
from efficientnet_pytorch import EfficientNet
from torch import nn
from ultralytics import YOLO


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

def load_krn_model(model_path, num_kpts_inf):
    krn_model = EfficientNet.from_pretrained("efficientnet-b0")
    krn_model._fc = nn.Linear(1280, num_kpts_inf * 2)  # Assuming 16 keypoints
    krn_model.load_state_dict(torch.load(model_path, weights_only=True)["state_dict"], strict=False)
    krn_model = krn_model.to(config.DEVICE)




if __name__ == "__main__":
    detection_model = YOLO(config.ODN_MODEL_PATH)
    krn_model = load_krn_model(config.KRN_MODEL_PATH, config.NUM_KPTS_INF)