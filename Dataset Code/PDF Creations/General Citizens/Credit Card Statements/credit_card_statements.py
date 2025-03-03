import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_credit_card_statement_pdf():
    for i in range(20):
        c = canvas.Canvas(f"credit_card_statement_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Credit Card Statement")
        y -= 20
        c.drawString(100, y, f"Cardholder: {fake.name()}")
        y -= 20
        c.drawString(100, y, f"Card Number: 4{random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}")
        y -= 20
        c.drawString(100, y, f"Statement Date: {fake.date_between(start_date='-1m', end_date='today')}")
        y -= 40
        
        # Transactions
        c.drawString(100, y, "Date    Description    Amount")
        y -= 20
        total = 0
        for _ in range(5):
            amount = random.randint(100, 10000)
            total += amount
            c.drawString(100, y, f"{fake.date_this_month()}    {fake.company()}    INR {amount}")
            y -= 20
        y -= 20
        c.drawString(100, y, f"Total Due: INR {total}")
        y -= 20
        c.drawString(100, y, f"Due Date: {fake.date_between(start_date='today', end_date='+1m')}")
        
        c.save()

if __name__ == "__main__":
    generate_credit_card_statement_pdf()
    print("20 Credit Card Statement PDFs generated!")