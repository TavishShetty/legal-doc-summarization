import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_utility_bill_pdf():
    for i in range(20):
        c = canvas.Canvas(f"utility_bill_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Electricity Bill")
        y -= 20
        c.drawString(100, y, f"Consumer Name: {fake.name()}")
        y -= 20
        c.drawString(100, y, f"Consumer ID: {random.randint(100000, 999999)}")
        y -= 20
        c.drawString(100, y, f"Address: {fake.address().replace('\n', ', ')}")
        y -= 20
        c.drawString(100, y, f"Bill Date: {fake.date_this_month()}")
        y -= 40
        
        # Details
        units = random.randint(50, 500)
        rate = random.uniform(5, 10)
        amount = units * rate
        c.drawString(100, y, f"Units Consumed: {units}")
        y -= 20
        c.drawString(100, y, f"Rate per Unit: INR {rate:.2f}")
        y -= 20
        c.drawString(100, y, f"Total Amount: INR {amount:.2f}")
        y -= 20
        c.drawString(100, y, f"Due Date: {fake.date_between(start_date='today', end_date='+1m')}")
        
        c.save()

if __name__ == "__main__":
    generate_utility_bill_pdf()
    print("20 Utility Bill PDFs generated!")