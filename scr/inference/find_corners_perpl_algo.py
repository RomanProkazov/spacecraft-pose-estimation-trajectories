import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# Load image
image_path = "cropped_marker_image.png"
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Image not found or cannot be loaded.")

# Convert to grayscale and preprocess
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

# Approximate contour to polygon
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)
points = approx.reshape(-1, 2)

# If more than 4 points, select 4 furthest apart
if len(points) > 4:
    dist_matrix = squareform(pdist(points))
    dist_sums = dist_matrix.sum(axis=1)
    idx = np.argsort(dist_sums)[-4:]
    corners = points[idx]
else:
    corners = points

# Order corners clockwise from top-left
def order_corners(corners):
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sums = corners.sum(axis=1)
    top_left_idx = np.argmin(sums)
    start_idx = np.where(sorted_indices == top_left_idx)[0][0]
    ordered_indices = np.roll(sorted_indices, -start_idx)
    return corners[ordered_indices]

ordered_corners = order_corners(corners)

# Plot and save
for corner in ordered_corners:
    cv2.circle(img, tuple(corner), 6, (0, 0, 255), -1)  # Red dot

output_path = "cropped_marker_with_corners.jpg"
cv2.imwrite(output_path, img)

# Also show with matplotlib (for inline visualization)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Detected Marker Corners')
plt.axis('off')
plt.show()
