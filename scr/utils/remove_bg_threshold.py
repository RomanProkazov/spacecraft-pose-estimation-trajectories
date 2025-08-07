from rembg import remove, new_session
from PIL import Image
import numpy as np
import os
import cv2

# Configuration
input_folder = "../../data_real/lux_sat_data_real_v4"
output_folder = "../../data_real/lux_sat_data_real_v4_nobck"
model_name = "u2net"  # or try "isnet-general-use"
alpha_matting = True
foreground_threshold = 10
background_threshold = 10
erode_size = 10  # You can try 0–10, lower = preserve thin features

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Create rembg session
session = new_session(model_name)

# Process all images
for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    print(f"Processing {filename}...")

    # Open image
    input_path = os.path.join(input_folder, filename)
    input_image = Image.open(input_path).convert("RGBA")

    # Remove background with alpha matting
    output_image = remove(
        input_image,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=foreground_threshold,
        alpha_matting_background_threshold=background_threshold,
        alpha_matting_erode_size=erode_size,
        session=session
    )

    # Convert to numpy
    output_np = np.array(output_image)

    # Replace background with black
    rgb = output_np[:, :, :3]
    alpha = output_np[:, :, 3] / 255.0
    black_background = np.zeros_like(rgb)

    composited = (rgb * alpha[..., None] + black_background * (1 - alpha[..., None])).astype(np.uint8)

    # Save final image
    final_image = Image.fromarray(composited)
    final_image.save(os.path.join(output_folder, filename))

print("✅ Done. All processed images saved to:", output_folder)
