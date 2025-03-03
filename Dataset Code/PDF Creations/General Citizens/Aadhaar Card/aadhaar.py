import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_aadhaar_pdf():
    for i in range(20):
        c = canvas.Canvas(f"aadhaar_card_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Government of India")
        y -= 20
        c.drawString(100, y, "Unique Identification Authority of India")
        y -= 40
        
        # Details
        c.drawString(100, y, f"Name: {fake.first_name()} {fake.last_name()}")
        y -= 20
        c.drawString(100, y, f"Aadhaar Number: {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}")
        y -= 20
        c.drawString(100, y, f"Date of Birth: {fake.date_of_birth(minimum_age=18, maximum_age=80)}")
        y -= 20
        c.drawString(100, y, f"Address: {fake.address().replace('\n', ', ')}")
        
        c.save()

if __name__ == "__main__":
    generate_aadhaar_pdf()
    print("20 Aadhaar Card PDFs generated!")