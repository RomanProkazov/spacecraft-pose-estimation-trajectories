import os
import numpy as np
import cv2
import img2pdf
from reportlab.lib.pagesizes import A4  # or LETTER
from reportlab.pdfgen import canvas


# ------------------------------
# ENTER YOUR PARAMETERS HERE:
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
LENGTH_PX = 640   # total length of the page in pixels
MARGIN_PX = 20    # size of the margin in pixels
SAVE_NAME = 'images/ChArUco_Markercv24.png'
# ------------------------------

def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    cv2.imshow("img", img)
    cv2.waitKey(2000)
    cv2.imwrite(SAVE_NAME, img)


def png_to_pdf(input_png, output_pdf):
    """Convert PNG to PDF (preserves quality)"""
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert(input_png))

import os
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ... (keep your existing constants and create_and_save_new_board() function)

def png_to_pdf_centered(input_png, output_pdf):
    """Convert PNG to PDF with centered, rotated, and aspect-ratio-preserved image"""
    # Load and rotate image 90 degrees clockwise
    img = cv2.imread(input_png)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated_height, rotated_width = img.shape[:2]
    
    # Temporarily save rotated image
    rotated_path = "rotated_temp.png"
    cv2.imwrite(rotated_path, img)

    # Calculate scaling to fit A4 while preserving aspect ratio
    a4_width, a4_height = A4
    img_ratio = rotated_width / rotated_height
    a4_ratio = a4_width / a4_height

    if img_ratio > a4_ratio:
        scale = a4_width / rotated_width  # Scale to fit width
    else:
        scale = a4_height / rotated_height  # Scale to fit height

    scaled_width = rotated_width * scale
    scaled_height = rotated_height * scale

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
    png_to_pdf_centered("images/ChArUco_Markercv2.png", "images/ChArUco_Marker_a4cv42.pdf")