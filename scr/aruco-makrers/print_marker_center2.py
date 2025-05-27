from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# Configuration
MARKER_SIZE_CM = 8.4  # Exact 8 cm
output_path = "aruco_marker_centered3.pdf"
marker_image = "aruco_marker.png"

# Create PDF with explicit dimensions
c = canvas.Canvas(output_path, pagesize=A4)
width, height = A4

# Calculate center position (compensate for printer margins)
x = (width - MARKER_SIZE_CM * cm) / 2
y = (height - MARKER_SIZE_CM * cm) / 2

# Draw with precise dimensions
c.drawImage(
    marker_image,
    x,
    y,
    width=MARKER_SIZE_CM * cm,
    height=MARKER_SIZE_CM * cm,
    preserveAspectRatio=True  # Prevents distortion
)

# Force PDF to use physical units
c.setPageCompression(0)  # Disable compression that might affect dimensions
c.save()