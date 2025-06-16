import os
import cv2
from rembg import remove
from glob import glob
import numpy as np

# Paths 
input_dir = "../../data_real/lux_sat_data_real_v1"  # your folder of images
output_dir = "../../data_real/lux_sat_data_real_v1_nobck"
os.makedirs(output_dir, exist_ok=True)

# Process each image
for file_path in glob(os.path.join(input_dir, "*.jpg")):  # or .png
    with open(file_path, "rb") as f:
        input_image = f.read()
    
    # Remove background (RGBA output)
    output_image = remove(input_image)
    
    # Convert to numpy array for OpenCV
    img_rgba = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)

    # Create black background
    if img_rgba.shape[2] == 4:
        alpha = img_rgba[:, :, 3] / 255.0
        black_bg = np.zeros_like(img_rgba[:, :, :3])
        composite = (img_rgba[:, :, :3] * alpha[..., None] + black_bg * (1 - alpha[..., None])).astype(np.uint8)
    else:
        composite = img_rgba  # fallback
    
    # Save final image
    filename = os.path.basename(file_path)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, composite)
    
    print(f"Processed: {filename}")
