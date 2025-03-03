import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_bank_statement_pdf():
    for i in range(20):
        c = canvas.Canvas(f"bank_statement_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Bank Statement")
        y -= 20
        c.drawString(100, y, f"Account Holder: {fake.name()}")
        y -= 20
        c.drawString(100, y, f"Account Number: {random.randint(1000000000, 9999999999)}")
        y -= 20
        c.drawString(100, y, f"Period: {fake.date_this_month()} to {fake.date_this_month()}")
        y -= 40
        
        # Transactions
        c.drawString(100, y, "Date    Description    Debit    Credit    Balance")
        y -= 20
        balance = random.randint(10000, 100000)
        for _ in range(5):
            amount = random.randint(500, 20000)
            is_debit = random.choice([True, False])
            if is_debit:
                balance -= amount
                c.drawString(100, y, f"{fake.date_this_month()}    {fake.company()}    INR {amount}    -    INR {balance}")
            else:
                balance += amount
                c.drawString(100, y, f"{fake.date_this_month()}    {fake.company()}    -    INR {amount}    INR {balance}")
            y -= 20
        
        c.save()

if __name__ == "__main__":
    generate_bank_statement_pdf()
    print("20 Bank Statement PDFs generated!")