import requests
import io
from pypdf import PdfWriter

# Create an in-memory PDF
writer = PdfWriter()
page = writer.add_blank_page(width=612, height=792)
pdf_bytes = io.BytesIO()
writer.write(pdf_bytes)
pdf_bytes.seek(0)

# Test invalid file type upload
res_bad = requests.post("http://127.0.0.1:8000/upload-pdf", files={"file": ("test.txt", b"Hello", "text/plain")})
print("Bad Upload Status:", res_bad.status_code, res_bad.json())

# Test valid PDF upload
res_good = requests.post("http://127.0.0.1:8000/upload-pdf", files={"file": ("test_doc.pdf", pdf_bytes, "application/pdf")})
print("Good Upload Status:", res_good.status_code, res_good.json())
