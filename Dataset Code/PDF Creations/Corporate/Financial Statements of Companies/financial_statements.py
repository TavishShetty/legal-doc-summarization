import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_financial_statement_pdf():
    for i in range(20):
        c = canvas.Canvas(f"financial_statement_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Profit & Loss Statement")
        y -= 20
        c.drawString(100, y, f"Company: {fake.company()}")
        y -= 20
        c.drawString(100, y, f"Period: FY {random.randint(2020, 2024)}-{random.randint(2021, 2025)}")
        y -= 40
        
        # Revenue
        revenue = random.randint(1000000, 100000000)
        c.drawString(100, y, f"Revenue: INR {revenue}")
        y -= 20
        
        # Expenses
        expenses = random.randint(500000, revenue - 100000)
        c.drawString(100, y, f"Expenses: INR {expenses}")
        y -= 20
        
        # Net Profit
        profit = revenue - expenses
        c.drawString(100, y, f"Net Profit: INR {profit}")
        y -= 20
        
        c.drawString(100, y, f"Location: {fake.city()}")
        
        c.save()

if __name__ == "__main__":
    generate_financial_statement_pdf()
    print("20 Financial Statement PDFs generated!")