import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_purchase_order_pdf():
    for i in range(20):
        c = canvas.Canvas(f"purchase_order_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Purchase Order")
        y -= 20
        c.drawString(100, y, f"PO Number: PO{random.randint(10000, 99999)}")
        y -= 20
        c.drawString(100, y, f"Date: {fake.date_between(start_date='-1y', end_date='today')}")
        y -= 40
        
        # Buyer and Supplier
        c.drawString(100, y, f"Buyer: {fake.company()}")
        y -= 20
        c.drawString(100, y, f"Supplier: {fake.company()}")
        y -= 40
        
        # Items
        c.drawString(100, y, "Item    Quantity    Rate    Total")
        y -= 20
        total = 0
        for _ in range(3):
            qty = random.randint(1, 100)
            rate = random.randint(100, 10000)
            amount = qty * rate
            total += amount
            c.drawString(100, y, f"{fake.word()}    {qty}    INR {rate}    INR {amount}")
            y -= 20
        c.drawString(100, y, f"Grand Total: INR {total}")
        
        c.save()

if __name__ == "__main__":
    generate_purchase_order_pdf()
    print("20 Purchase Order PDFs generated!")