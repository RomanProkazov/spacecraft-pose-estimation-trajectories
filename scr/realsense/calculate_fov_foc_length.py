import numpy as np
import math

# Intrinsic parameters from camera_matrix
fx = 392.88  # focal length in x (pixels)
fy = 392.56  # focal length in y (pixels)
pixel_size_mm = 0.003  # pixel size in mm (example: OV9782 = 3 microns)

# Resolution of the camera
width = 640   # pixels
height = 480  # pixels

# Convert focal length to mm using pixel size
focal_length_mm = fx * pixel_size_mm

# Compute sensor size in mm
sensor_width_mm = width * pixel_size_mm
sensor_height_mm = height * pixel_size_mm

# Compute field of view in degrees
fov_x = 2 * math.degrees(math.atan(sensor_width_mm / (2 * focal_length_mm)))
fov_y = 2 * math.degrees(math.atan(sensor_height_mm / (2 * focal_length_mm)))

# Print results
print(f"Sensor width for Blender: {sensor_width_mm:.3f} mm")
print(f"Sensor height for Blender: {sensor_height_mm:.3f} mm")
print(f"Focal length in mm: {focal_length_mm:.2f} mm")
print(f"Horizontal FOV: {fov_x:.2f}°")
print(f"Vertical FOV: {fov_y:.2f}°")
