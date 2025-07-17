from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import io

# Input and output PDF paths
input_pdf = "charuco_board.pdf"
output_pdf = "scaled_charuco.pdf"
scaling_factor = 1.0714  # 15mm / 14mm

# Read the original PDF
reader = PdfReader(input_pdf)
writer = PdfWriter()

# Scale each page
for page in reader.pages:
    page.scale_by(scaling_factor)  # Apply scaling
    writer.add_page(page)

# Save the scaled PDF
with open(output_pdf, "wb") as f:
    writer.write(f)
print(f"Scaled PDF saved: {output_pdf}")