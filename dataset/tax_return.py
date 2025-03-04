import random
import string
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_tax_return_pdf():
    for i in range(20):
        c = canvas.Canvas(f"tax_return_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Income Tax Return")
        y -= 20
        c.drawString(100, y, f"Name: {fake.name()}")
        y -= 20
        pan = ''.join(random.choices(string.ascii_uppercase, k=5)) + ''.join(random.choices(string.digits, k=4)) + random.choice(string.ascii_uppercase)
        c.drawString(100, y, f"PAN: {pan}")
        y -= 20
        c.drawString(100, y, f"Assessment Year: {random.randint(2023, 2025)}-{random.randint(2024, 2026)}")
        y -= 40
        
        # Income
        salary = random.randint(300000, 5000000)
        tax = salary * 0.2
        c.drawString(100, y, f"Salary Income: INR {salary}")
        y -= 20
        c.drawString(100, y, f"Tax Payable: INR {tax:.2f}")
        y -= 20
        c.drawString(100, y, f"Filed Date: {fake.date_between(start_date='-1y', end_date='today')}")
        
        c.save()

if __name__ == "__main__":
    generate_tax_return_pdf()
    print("20 Tax Return PDFs generated!")