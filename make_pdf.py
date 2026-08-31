from pypdf import PdfWriter

writer = PdfWriter()
page = writer.add_blank_page(width=612, height=792)
with open("test_sample.pdf", "wb") as f:
    writer.write(f)
print("Created test_sample.pdf")
