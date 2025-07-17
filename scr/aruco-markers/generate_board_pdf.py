import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from cv2 import aruco

# --- Generate ArUco Board ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.GridBoard(
    size=(5, 7),          # markersX, markersY
    markerLength=30,      # in pixels (arbitrary, scaling adjusted later)
    markerSeparation=5,  # spacing between markers
    dictionary=aruco_dict
)

# Generate the image (OpenCV format)
img_width_px = 5 * 30 + 4 * 5  # Total width in pixels
img_height_px = 7 * 30 + 6 * 5  # Total height in pixels
board_img = board.generateImage((img_width_px, img_height_px), marginSize=0)

# Save temporarily as PNG
temp_img_path = "temp_aruco_board_margin0.png"
cv2.imwrite(temp_img_path, board_img)

# --- PDF Generation (Centered & Scaled to Fit) ---
pdf_path = "aruco_board.pdf"
c = canvas.Canvas(pdf_path, pagesize=A4)

# Get PDF dimensions
pdf_width, pdf_height = A4  # (595.27, 841.89 in points, 1 point = 1/72 inch)

# Calculate scaling to fit the page (preserve aspect ratio)
img_ratio = img_width_px / img_height_px
pdf_ratio = pdf_width / pdf_height

if img_ratio > pdf_ratio:
    # Image is wider than PDF → scale to fit width
    scale = pdf_width / img_width_px
else:
    # Image is taller than PDF → scale to fit height
    scale = pdf_height / img_height_px

scaled_width = img_width_px * scale
scaled_height = img_height_px * scale

# Center the image on the page
x_pos = (pdf_width - scaled_width) / 2
y_pos = (pdf_height - scaled_height) / 2

# Draw the image (centered and scaled)
c.drawImage(
    temp_img_path,
    x_pos,
    y_pos,
    width=scaled_width,
    height=scaled_height,
    preserveAspectRatio=True,
    mask='auto'
)

c.save()  # Save PDF
print(f"ArUco board saved to {pdf_path}")