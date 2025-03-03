import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_pay_slip_pdf():
    for i in range(20):
        c = canvas.Canvas(f"pay_slip_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Pay Slip")
        y -= 20
        c.drawString(100, y, f"Employee: {fake.name()}")
        y -= 20
        c.drawString(100, y, f"Employee ID: EMP{random.randint(1000, 9999)}")
        y -= 20
        c.drawString(100, y, f"Month: {fake.month_name()} {random.randint(2023, 2025)}")
        y -= 40
        
        # Earnings
        basic = random.randint(20000, 100000)
        hra = basic * 0.4
        total = basic + hra
        c.drawString(100, y, f"Basic Salary: INR {basic}")
        y -= 20
        c.drawString(100, y, f"HRA: INR {hra:.2f}")
        y -= 20
        c.drawString(100, y, f"Total Earnings: INR {total:.2f}")
        y -= 20
        c.drawString(100, y, f"Company: {fake.company()}")
        
        c.save()

if __name__ == "__main__":
    generate_pay_slip_pdf()
    print("20 Pay Slip PDFs generated!")