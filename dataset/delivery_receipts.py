import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_delivery_receipt_pdf():
    for i in range(2000):
        c = canvas.Canvas(f"delivery_receipt_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Delivery Receipt")
        y -= 20
        c.drawString(100, y, f"Order ID: ORD{random.randint(10000, 99999)}")
        y -= 20
        c.drawString(100, y, f"Date: {fake.date_between(start_date='-1m', end_date='today')}")
        y -= 40
        
        # Details
        c.drawString(100, y, f"Customer: {fake.name()}")
        y -= 20
        c.drawString(100, y, f"Address: {fake.address().replace('\n', ', ')}")
        y -= 20
        c.drawString(100, y, f"Item: {fake.word()} (Qty: {random.randint(1, 5)})")
        y -= 20
        c.drawString(100, y, f"Amount: INR {random.randint(100, 10000)}")
        y -= 20
        c.drawString(100, y, f"Delivered By: {fake.name()}")
        
        c.save()

if __name__ == "__main__":
    generate_delivery_receipt_pdf()
    print("2000 Delivery Receipt PDFs generated!")