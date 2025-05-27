import cv2
import numpy as np
from cv2 import aruco
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm



# === CONFIGURATION ===
marker_id = 23
marker_size_cm = 8
dpi = 300
cm_per_inch = 2.54
pixels = int((marker_size_cm / cm_per_inch) * dpi)  # Convert cm to pixels

# === GENERATE MARKER ===
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_img = aruco.generateImageMarker(aruco_dict, marker_id, pixels)

# === CONVERT TO PIL IMAGE AND SAVE TO PDF ===
# Convert to RGB for PDF export
marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
image_pil = Image.fromarray(marker_rgb)

# Set correct DPI and save as PDF
image_pil.save("aruco_marker_8cm.pdf", "PDF", dpi=(dpi, dpi))

print("✅ Marker saved as aruco_marker_8cm.pdf with 8×8 cm size at 300 DPI.")
