import random
import string
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_pan_pdf():
    for i in range(20):
        c = canvas.Canvas(f"pan_card_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Income Tax Department, Govt. of India")
        y -= 40
        
        # Details
        c.drawString(100, y, f"Name: {fake.first_name()} {fake.last_name()}")
        y -= 20
        pan = ''.join(random.choices(string.ascii_uppercase, k=5)) + ''.join(random.choices(string.digits, k=4)) + random.choice(string.ascii_uppercase)
        c.drawString(100, y, f"PAN Number: {pan}")
        y -= 20
        c.drawString(100, y, f"Father's Name: {fake.last_name()} {fake.first_name()}")
        y -= 20
        c.drawString(100, y, f"Date of Birth: {fake.date_of_birth(minimum_age=18, maximum_age=80)}")
        
        c.save()

if __name__ == "__main__":
    generate_pan_pdf()
    print("20 PAN Card PDFs generated!")