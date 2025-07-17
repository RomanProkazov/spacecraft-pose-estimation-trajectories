import pyrealsense2 as rs
import cv2
import os
import numpy as np


SAVE_DIR = "charuco_1280_720_cv42"
os.makedirs(SAVE_DIR, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

cv2.namedWindow("Checkerboard Capture", cv2.WINDOW_AUTOSIZE)

frame_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
  
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Checkerboard Capture", color_image)

        key = cv2.waitKey(1)
        if key == ord('s'):
            # Save images
            color_filename = os.path.join(SAVE_DIR, f"color_arucoboard_{frame_count:04d}.png")
            cv2.imwrite(color_filename, color_image)
            print(f"Saved frame pair {frame_count:04d}")
            frame_count += 1

        elif key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Capture completed. Saved {frame_count} image pairs to {SAVE_DIR}")