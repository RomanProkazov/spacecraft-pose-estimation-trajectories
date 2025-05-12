import numpy as np
import math

# Intrinsic parameters from camera_matrix
fx = 392.88  # focal length in x (pixels)
fy = 392.56  # focal length in y (pixels)
pixel_size_mm = 0.003  # e.g., for OV9782 sensor

# Resolution of your camera
width = 640
height = 480

# Calculate FOV in degrees
shift_x = ideal_xc / 
focal_length_mm = fx * pixel_size_mm
sensor_width_mm = (focal_length_mm * width) / fx
sensor_height_mm = (focal_length_mm * height) / fy


print(f"Sensor width for Blender: {sensor_width_mm:.3f} mm")
print(f"Sensor hight for Blender: {sensor_height_mm:.3f} mm")
print(f"Focal length in mm: {focal_length_mm:.2f} mm")
print(f"Horizontal FOV: {fov_x:.2f}°")
print(f"Vertical FOV: {fov_y:.2f}°")