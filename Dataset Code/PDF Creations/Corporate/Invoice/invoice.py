import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_invoice_pdf():
    for i in range(20):
        c = canvas.Canvas(f"invoice_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "GST Invoice")
        y -= 30
        c.drawString(100, y, f"Invoice No: INV{random.randint(10000, 99999)}")
        y -= 20
        c.drawString(100, y, f"Date: {fake.date_between(start_date='-1y', end_date='today')}")
        y -= 40
        
        # Seller Details
        c.drawString(100, y, "Seller:")
        y -= 20
        c.drawString(100, y, f"{fake.company()}")
        y -= 20
        c.drawString(100, y, f"{fake.address().replace('\n', ', ')}")
        y -= 20
        c.drawString(100, y, f"GSTIN: {random.randint(10, 99)}{fake.lexify('??')}{random.randint(100000, 999999)}")
        y -= 40
        
        # Buyer Details
        c.drawString(100, y, "Buyer:")
        y -= 20
        c.drawString(100, y, f"{fake.name()}")
        y -= 20
        c.drawString(100, y, f"{fake.address().replace('\n', ', ')}")
        y -= 40
        
        # Items
        c.drawString(100, y, "Description    Quantity    Rate    Amount")
        y -= 20
        total = 0
        for _ in range(3):
            qty = random.randint(1, 10)
            rate = random.randint(500, 5000)
            amount = qty * rate
            total += amount
            c.drawString(100, y, f"Item {fake.word()}    {qty}    INR {rate}    INR {amount}")
            y -= 20
        y -= 20
        c.drawString(100, y, f"Total: INR {total}")
        
        c.save()

if __name__ == "__main__":
    generate_invoice_pdf()
    print("20 Invoice PDFs generated!")