from docx import Document
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def convert_to_input_format(text, file_type):
    temp_file = os.path.join(tempfile.gettempdir(), f"anon_output.{file_type}")
    if file_type == 'pdf':
        c = canvas.Canvas(temp_file, pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        for line in text.split('\n'):
            c.drawString(100, y, line[:100])  # Truncate for simplicity
            y -= 20
        c.save()
    elif file_type == 'docx':
        doc = Document()
        doc.add_paragraph(text)
        doc.save(temp_file)
    elif file_type == 'txt':
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(text)
    return temp_file