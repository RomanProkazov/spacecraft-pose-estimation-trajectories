from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# Set up page
width, height = A4
c = canvas.Canvas("aruco_marker_centered.pdf", pagesize=A4)

# Marker size
marker_size_cm = 8  # 8 cm
marker_path = "aruco_marker.png"

# Calculate position to center the marker
x = (width - marker_size_cm * cm) / 2
y = (height - marker_size_cm * cm) / 2

# Draw image
c.drawImage(marker_path, x, y, width=marker_size_cm * cm, height=marker_size_cm * cm)

# Save PDF
c.save()
