import pyrealsense2 as rs
import cv2
import os
import numpy as np

# Configure directories and parameters
CHECKERBOARD_SIZE = (10, 7)  # Inner corners (width, height)
SAVE_DIR = "checkerboard_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Create alignment object
align_to = rs.stream.color
align = rs.align(align_to)

# Create window
cv2.namedWindow("Checkerboard Capture", cv2.WINDOW_AUTOSIZE)

frame_count = 0

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert to grayscale for checkerboard detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # # Find chessboard corners
        # ret, corners = cv2.findChessboardCorners(
        #     gray, CHECKERBOARD_SIZE, 
        #     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        # )

        # # Draw detected corners
        # if ret:
        #     cv2.drawChessboardCorners(color_image, CHECKERBOARD_SIZE, corners, ret)
        #     status_text = "Checkerboard detected - Press 's' to save"
        # else:
        #     status_text = "No checkerboard detected"

        # # Display status
        # cv2.putText(color_image, status_text, (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show images
        cv2.imshow("Checkerboard Capture", color_image)

        # Handle key press
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Save images
            color_filename = os.path.join(SAVE_DIR, f"color5_{frame_count:04d}.png")
            depth_filename = os.path.join(SAVE_DIR, f"depth_{frame_count:04d}.png")
            
            cv2.imwrite(color_filename, color_image)
            # cv2.imwrite(depth_filename, depth_image)
            print(f"Saved frame pair {frame_count:04d}")
            frame_count += 1

        elif key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Capture completed. Saved {frame_count} image pairs to {SAVE_DIR}")