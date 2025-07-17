import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt

# Load an image and preprocess it for EfficientNet
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # EfficientNet expects RGB
    img = cv2.resize(img, (224, 224))  # EfficientNet-B0 input size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Function to get the target convolutional layer
def get_conv_layer(model):
    # Use the last convolutional layer before pooling
    return model._conv_head

# Function to generate Grad-CAM heatmap for keypoints
def compute_gradcam(model, img_tensor, keypoint_idx, conv_layer):
    # Forward hook to store activations
    activations = []
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    hook = conv_layer.register_forward_hook(forward_hook)
    
    # Forward pass
    preds = model(img_tensor)
    
    # Get the target keypoint (x and y coordinates)
    target_x = preds[0, keypoint_idx * 2]
    target_y = preds[0, keypoint_idx * 2 + 1]
    
    # Zero gradients and backward pass for x coordinate
    model.zero_grad()
    target_x.backward(retain_graph=True)
    grads_x = img_tensor.grad.clone()
    
    # Backward pass for y coordinate
    model.zero_grad()
    target_y.backward()
    grads_y = img_tensor.grad.clone()
    
    # Combine gradients (use magnitude)
    grads = torch.sqrt(grads_x**2 + grads_y**2)
    
    # Remove the hook
    hook.remove()
    
    # Process activations and gradients
    activations = activations[0][0]
    pooled_grads = torch.mean(grads, dim=(0, 2, 3))[0]
    
    # Weight activations by gradients
    for i in range(activations.shape[0]):
        activations[i] *= pooled_grads[i]
    
    # Create heatmap
    heatmap = torch.mean(activations, dim=0)
    heatmap = torch.relu(heatmap)
    heatmap = heatmap.detach().cpu().numpy()
    
    # Normalize
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    return heatmap

# Overlay heatmap on image
def overlay_heatmap(img, heatmap, alpha=0.5):
    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image to RGB for proper blending
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img_rgb, alpha, heatmap, 1 - alpha, 0)
    
    return superimposed_img

if __name__ == "__main__":
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_KEYPOINTS = 5  # Number of keypoints your model predicts
    KEYPOINT_IDX = 0   # Index of keypoint to visualize (0 to NUM_KEYPOINTS-1)
    
    # Load EfficientNet-B0 model for keypoint detection
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(1280, NUM_KEYPOINTS * 2)  # 2 coordinates per keypoint
    
    # Load your trained weights
    model_path = "path/to/your/model_weights.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    # Get target convolutional layer
    conv_layer = get_conv_layer(model)
    
    # Process image
    img_path = "path/to/your/image.jpg"
    img = cv2.imread(img_path)
    img_tensor = preprocess_image(img_path).to(DEVICE)
    
    # Compute Grad-CAM heatmap
    heatmap = compute_gradcam(model, img_tensor, KEYPOINT_IDX, conv_layer)
    
    # Overlay heatmap on original image
    result_img = overlay_heatmap(img, heatmap)
    
    # Save and display results
    cv2.imwrite("keypoint_heatmap.jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    
    plt.subplot(122)
    plt.imshow(result_img)
    plt.title(f"Keypoint {KEYPOINT_IDX} Heatmap")
    plt.show()