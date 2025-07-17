import cv2
import numpy as np
import cv2
import numpy as np
from cv2 import aruco
import json

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scr.utils.general_utils import(load_camera_matrix_sat_model,
                                    load_images,
                                    load_labels)



def crop_with_margin(image_path, bbox, margin=1.3, save_path=None):
    """
    Crops an image with a margin around the bounding box.

    Args:
        image_path (str): Path to the image file.
        bbox (tuple): (xmin, ymin, xmax, ymax) bounding box coordinates.
        margin (float): Margin multiplier (e.g., 1.3 means 30% larger).
        save_path (str, optional): If provided, saves the cropped image.

    Returns:
        cropped_img (np.ndarray): Cropped image array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = map(int, bbox)

    # Calculate bbox width and height
    box_w = xmax - xmin
    box_h = ymax - ymin

    # Calculate margin in pixels
    margin_x = int((box_w * (margin - 1)) / 2)
    margin_y = int((box_h * (margin - 1)) / 2)

    # Compute new coordinates, ensuring they stay within image bounds
    x1 = max(0, xmin - margin_x)
    y1 = max(0, ymin - margin_y)
    x2 = min(w, xmax + margin_x)
    y2 = min(h, ymax + margin_y)

    cropped_img = img[y1:y2, x1:x2]

    if save_path:
        cv2.imwrite(save_path, cropped_img)
    return cropped_img



def order_points(pts):
    """Order points as: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def detect_marker(image_path):
    # # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=31, C=5
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])

    possible_markers = []
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 500:  # Threshold may need tuning
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = np.linalg.norm(np.array([cx, cy]) - img_center)
                    possible_markers.append((dist, approx))

    if not possible_markers:
        print("No marker found.")
        return

    # Select the closest-to-center contour
    possible_markers.sort(key=lambda x: x[0])
    best_marker = possible_markers[0][1]

    # Extract and order points
    corners = best_marker.reshape(4, 2)
    ordered_corners = order_points(corners)

    # Draw the detected corners
    vis_img = img.copy()
    for i, pt in enumerate(ordered_corners):
        pt = tuple(pt.astype(int))
        cv2.circle(vis_img, pt, 8, (0, 255, 0), -1)
        cv2.putText(vis_img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Show results
    cv2.imwrite("output_pose_estimation.png", vis_img)

    print("Ordered corners:")
    print(ordered_corners)

    return ordered_corners

if __name__ == "__main__":
    idx = 1000
    image_list = load_images(Path("../../data/images_render"))
    labels = load_labels("../../data/labels/labels_sat_1280px_20kimgs_leo_no_earth_v6.json")
    marker_bbox = labels[idx]['bbox_marker']
    bbox_margin = crop_with_margin(image_list[idx], marker_bbox, margin=1.3, save_path="cropped_marker_image.png")

    camera_matrix, object_points = load_camera_matrix_sat_model("../../data/labels/cam_sat.json", object='marker_model')
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # Assuming no distortion for simplicity
    # corners = detect_marker(image_list[idx])
    corners = detect_marker("cropped_marker_image.png")

    # Example use for PnP (if needed)
    if corners is not None:
        # Define 3D model points (real-world coordinates in your object frame)
        # Assume square marker of size 1 unit centered at origin
        object_points = object_points

        image_points = corners.astype(np.float32)


        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

        if success:
            print("Pose Estimation Successful!")
            print("Rotation Vector:\n", rvec)
            print("Translation Vector:\n", tvec)
        else:
            print("solvePnP failed.")
