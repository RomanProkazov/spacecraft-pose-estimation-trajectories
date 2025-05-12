import cv2
import numpy as np
import os
import json
import glob

# ================= Configuration =================
CHECKERBOARD_SIZE = (10, 7)        # Inner corners (width, height)
SQUARE_SIZE_MM = 25               # Checkerboard square size in mm
IMAGE_FOLDER = "checkerboard_images"  # Update with your image path
SAVE_DIR = "calibration_data"     # Output directory
# Supported image formats
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
# ==================================================

os.makedirs(SAVE_DIR, exist_ok=True)

# Prepare object points (3D)
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

# Storage for calibration points
objpoints = []  # 3D world points
imgpoints = []  # 2D image points
image_size = None  # Will be determined from first image

# Get list of calibration images
images = []
for ext in IMAGE_EXTENSIONS:
    images.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))

if not images:
    raise FileNotFoundError(f"No images found in {IMAGE_FOLDER} with extensions {IMAGE_EXTENSIONS}")

print(f"Found {len(images)} images. Processing...")

# Process each image
for idx, fname in enumerate(images):
    print(f"Processing image {idx+1}/{len(images)}: {os.path.basename(fname)}")
    img = cv2.imread(fname)
    
    if img is None:
        print(f"Warning: Could not read image {fname}")
        continue
    
    # Set image size from first valid image
    if image_size is None:
        image_size = img.shape[:2][::-1]  # (width, height)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # Refine corner locations
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        
        # Optional: Visualize detected corners
        cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners_refined, ret)
        cv2.imshow('Chessboard Detection', img)
        cv2.waitKey(500)  # Display for 500ms
    else:
        print(f"Checkerboard not found in {os.path.basename(fname)}")

cv2.destroyAllWindows()

# Check if we have enough calibration points
if len(objpoints) < 10:
    raise ValueError(f"Only {len(objpoints)} valid images found. Need at least 10 for calibration.")

print(f"\nStarting calibration with {len(objpoints)} valid images...")

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

# Save calibration data
np.savez(os.path.join(SAVE_DIR, "calibration.npz"),
         camera_matrix=mtx, dist_coeffs=dist)

# Save human-readable report
calibration_report = {
    "camera_matrix": mtx.tolist(),
    "distortion_coefficients": dist.tolist(),
    "reprojection_error": ret,
    "image_resolution": f"{image_size[0]}x{image_size[1]}",
    "checkerboard_size": CHECKERBOARD_SIZE,
    "square_size_mm": SQUARE_SIZE_MM,
    "num_images_used": len(objpoints),
    "image_paths": [os.path.abspath(f) for f in images]
}

with open(os.path.join(SAVE_DIR, "calibration_report.json"), 'w') as f:
    json.dump(calibration_report, f, indent=4)

print(f"\nCalibration complete!")
print(f"Reprojection error: {ret:.2f} (lower is better)")
print(f"Results saved to {SAVE_DIR}")

# Optional: Show undistorted example
if len(images) > 0:
    test_img = cv2.imread(images[0])
    if test_img is not None:
        undistorted = cv2.undistort(test_img, mtx, dist)
        cv2.imshow('Original vs Undistorted', np.hstack((test_img, undistorted)))
        cv2.waitKey(3000)
        cv2.destroyAllWindows()