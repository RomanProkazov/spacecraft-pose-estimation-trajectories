import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
sys.path.append("../../scr/krn")
import config
from efficientnet_pytorch import EfficientNet 
from ultralytics import YOLO
import os


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



def get_conv_layer(model, layer_name):
    """Retrieve convolutional layer by name"""
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer {layer_name} not found")

def compute_gradcam_keypoint(model, img_tensor, keypoint_index, conv_layer_name):
    """
    Generate Grad-CAM heatmap for a specific keypoint
    Args:
        model: EfficientNet model
        img_tensor: Preprocessed input tensor (1, 3, H, W)
        keypoint_index: Index of keypoint (0 to num_keypoints-1)
        conv_layer_name: Name of target convolutional layer
    Returns:
        heatmap: Normalized heatmap (H, W)
    """
    conv_layer = get_conv_layer(model, conv_layer_name)
    activations = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    hook = conv_layer.register_forward_hook(forward_hook)
    
    # Forward pass
    preds = model(img_tensor)
    target = preds[0, keypoint_index*2] + preds[0, keypoint_index*2+1]
    
    # Backward pass
    model.zero_grad()
    grad = torch.autograd.grad(target, activations[0], retain_graph=True)[0]
    
    # Global average pooling of gradients
    weights = torch.mean(grad, dim=(2, 3))
    
    # Weight activations
    weighted_activations = weights[:, :, None, None] * activations[0]
    heatmap = torch.sum(weighted_activations, dim=1).squeeze(0)
    heatmap = torch.relu(heatmap)
    
    # Normalize
    heatmap_np = heatmap.detach().cpu().numpy()
    heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np) + 1e-8)
    
    hook.remove()
    return heatmap_np

def overlay_heatmap_on_crop(cropped_img, heatmap, alpha=0.5):
    """
    Overlay heatmap on cropped image
    Args:
        cropped_img: Original cropped image (H, W, 3)
        heatmap: Resized heatmap (H, W)
        alpha: Transparency for overlay
    Returns:
        overlayed_img: Combined image with heatmap
    """
    heatmap_resized = cv2.resize(heatmap, (cropped_img.shape[1], cropped_img.shape[0]))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(cropped_img, alpha, heatmap_colored, 1 - alpha, 0)
    return overlayed_img




def predict_keypoints_with_confidence(cropped_img, krn_model, device, num_kpts, keypoints, conv_layer_name="_conv_head"):
    """
    Predict keypoints with confidence scores using Grad-CAM
    Args:
        cropped_img: Cropped image (numpy array)
        krn_model: EfficientNet model
        device: Torch device
        num_kpts: Number of keypoints
        conv_layer_name: Target layer name
    Returns:
        keypoints: Predicted keypoints (num_kpts, 2)
        confidences: Confidence scores (num_kpts,)
        overlayed_img: Image with combined heatmap
        keypoint_heatmaps: List of individual heatmaps
    """
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = transform(Image.fromarray(cropped_img)).unsqueeze(0).to(device)
    
    # Predict keypoints
    # krn_model.eval()
    # with torch.no_grad():
    #     preds = krn_model(img_tensor).squeeze(0).cpu().numpy()
    # keypoints = np.array([preds[0::2], preds[1::2]]).T
    
    
    # Compute heatmaps and confidences
    keypoint_heatmaps = []
    confidences = []
    
    for k in range(num_kpts):
        heatmap = compute_gradcam_keypoint(
            krn_model, 
            img_tensor, 
            k,
            conv_layer_name
        )
        heatmap_224 = cv2.resize(heatmap, (224, 224))
        
        # Get confidence (heatmap value at predicted keypoint)
        x, y = keypoints[k]
        x_pix = min(max(0, int(x * 224)), 223)
        y_pix = min(max(0, int(y * 224)), 223)
        confidence = heatmap_224[y_pix, x_pix]
        
        keypoint_heatmaps.append(heatmap_224)
        confidences.append(confidence)
    
    # Create combined heatmap
    combined_heatmap = np.mean(keypoint_heatmaps, axis=0)
    combined_heatmap = (combined_heatmap - combined_heatmap.min()) / \
                      (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
    
    # Overlay heatmap
    overlayed_img = overlay_heatmap_on_crop(cropped_img, combined_heatmap)
    
    return keypoints, np.array(confidences), overlayed_img, keypoint_heatmaps



if __name__ == "__main__":
        # Load models
    detection_model = YOLO(config.ODN_MODEL_PATH)  # Object detection model
    krn_model = EfficientNet.from_pretrained("efficientnet-b0")
    krn_model._fc = nn.Linear(1280, config.NUM_KPTS_INF * 2)
    krn_model.load_state_dict(torch.load(config.KRN_MODEL_PATH, map_location=config.DEVICE)["state_dict"], strict=False)
    krn_model = krn_model.to(config.DEVICE)
    
    # Process image
    idx = 478
    image_names = sorted(os.listdir(config.IMG_DIR), key=lambda x: int(x[4:-4]))
    images_list = [os.path.join(config.IMG_DIR, filename) for filename in image_names]
    original_image = cv2.imread(images_list[idx])
    
    # Object detection
    bboxes = detect_object(original_image, detection_model)
    if len(bboxes) == 0:
        print("No objects detected.")
    else:
        bbox = bboxes[0]
        cropped_img = preprocess_crop(original_image, bbox)

        keypoints = predict_keypoints(cropped_img, krn_model, config.DEVICE, config.NUM_KPTS)
        
        # Predict keypoints with confidence and heatmaps
        keypoints, confidences, overlayed_img, heatmaps = predict_keypoints_with_confidence(
            cropped_img, 
            krn_model, 
            config.DEVICE, 
            config.NUM_KPTS,
            keypoints,
            conv_layer_name="_conv_head"  # For EfficientNet-B0
        )
        
        # Visualize results
        print("Keypoint Confidences:", confidences)
        
        # Show combined heatmap overlay
        cv2.imshow("Combined Heatmap", overlayed_img)
        cv2.waitKey(0)
        
        # Visualize individual keypoints
        for i, (x, y) in enumerate(keypoints):
            x_pix = int(x * 224)
            y_pix = int(y * 224)
            cv2.circle(cropped_img, (x_pix, y_pix), 1, (0, 255, 0), -1)
            cv2.putText(cropped_img, f"{i}: {confidences[i]:.2f}", 
                        (x_pix+5, y_pix), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        
        cv2.imshow("Keypoints with Confidence", cropped_img)
        cv2.waitKey(0)
        
        # Save outputs
        cv2.imwrite("heatmap_overlay.jpg", overlayed_img)
        cv2.imwrite("keypoints_with_confidence.jpg", cropped_img)