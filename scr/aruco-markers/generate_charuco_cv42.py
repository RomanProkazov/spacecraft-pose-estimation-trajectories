import os
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ------------------------------
# ENTER YOUR PARAMETERS HERE:
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
LENGTH_PX = 640   # total height of the page in pixels (vertical orientation)
MARGIN_PX = 20    # size of the margin in pixels
SAVE_NAME = 'images/ChArUco_Markercv42.png'
# ------------------------------

def create_and_save_new_board():
    dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard_create(SQUARES_HORIZONTALLY, SQUARES_VERTICALLY, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = board.draw((int(LENGTH_PX * size_ratio), int(LENGTH_PX)), marginSize=MARGIN_PX)
    cv2.imshow("img", img)
    cv2.waitKey(2000)
    cv2.imwrite(SAVE_NAME, img)

def png_to_pdf_centered(input_png, output_pdf):
    """Convert PNG to PDF with centered, vertical, and aspect-ratio-preserved image"""
    # Load image (no rotation to keep vertical orientation)
    img = cv2.imread(input_png)
    height, width = img.shape[:2]
    
    # Temporarily save if needed (no rotation here)
    rotated_path = "temp.png"
    cv2.imwrite(rotated_path, img)

    # Calculate scaling to fit A4 height while preserving aspect ratio
    a4_width, a4_height = A4  # A4 in points: 595 x 842 (portrait)
    img_ratio = width / height
    a4_ratio = a4_width / a4_height

    # Scale to fit the full height of A4 (842 points), adjust width accordingly
    scale = a4_height / height
    scaled_width = width * scale
    scaled_height = height * scale

    # Ensure width doesn't exceed A4 width (595 points)
    if scaled_width > a4_width:
        scale = a4_width / width
        scaled_width = a4_width
        scaled_height = height * scale

    # Calculate centering offsets
    x_offset = (a4_width - scaled_width) / 2
    y_offset = (a4_height - scaled_height) / 2

    # Create PDF with centered image
    c = canvas.Canvas(output_pdf, pagesize=A4)
    c.drawImage(
        rotated_path,
        x_offset,
        y_offset,
        width=scaled_width,
        height=scaled_height,
        preserveAspectRatio=True
    )
    c.save()
    
    # Clean up temporary file
    os.remove(rotated_path)

if __name__ == "__main__":
    create_and_save_new_board()
    png_to_pdf_centered(SAVE_NAME, "images/ChArUco_Marker_a4cv42.pdf")