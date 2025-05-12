import pyrealsense2 as rs
import cv2
import numpy as np
import os
import json

# ================= Configuration =================
CHECKERBOARD_SIZE = (10, 7)        # Number of inner corners (width, height)
SQUARE_SIZE_MM = 25               # Size of one checkerboard square in millimeters
MIN_FRAMES = 15                   # Minimum frames required for calibration
SAVE_DIR = "calibration_data"     # Directory to save calibration results
# ==================================================

os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD_SIZE[0]*CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE_MM

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

cv2.namedWindow("Calibration", cv2.WINDOW_AUTOSIZE)
print("Press 's' to capture frame, 'c' to calibrate, 'q' to quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # If found, refine and store points
        if ret:
            # Refine corner locations
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            
            # Draw detected corners
            cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners_refined, ret)
            status = f"Detected! Captured: {len(objpoints)}/{MIN_FRAMES}"
        else:
            status = "No checkerboard detected"

        # Display status
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Calibration", frame)

        # Handle key input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and ret:
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            print(f"Frame {len(objpoints)} captured")
        elif key == ord('c'):
            if len(objpoints) < MIN_FRAMES:
                print(f"Need at least {MIN_FRAMES} frames. Currently have {len(objpoints)}")
                continue
                
            print("Calibrating...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            # Save calibration data
            np.savez(os.path.join(SAVE_DIR, "calibration.npz"),
                     camera_matrix=mtx, dist_coeffs=dist)
            
            # Save human-readable report
            calibration_report = {
                "camera_matrix": mtx.tolist(),
                "distortion_coefficients": dist.tolist(),
                "reprojection_error": ret,
                "resolution": "640x480",
                "checkerboard_size": CHECKERBOARD_SIZE,
                "square_size_mm": SQUARE_SIZE_MM,
                "num_frames": len(objpoints)
            }
            
            with open(os.path.join(SAVE_DIR, "calibration_report.json"), 'w') as f:
                json.dump(calibration_report, f, indent=4)
            
            print(f"Calibration saved to {SAVE_DIR}")
            print(f"Reprojection error: {ret:.2f} (lower is better)")
            
            # Show undistorted image
            undistorted = cv2.undistort(frame, mtx, dist)
            cv2.imshow("Undistorted", undistorted)
            cv2.waitKey(2000)
            
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()