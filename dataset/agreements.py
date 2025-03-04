import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_agreement_pdf():
    for i in range(20):
        c = canvas.Canvas(f"agreement_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Service Agreement")
        y -= 20
        c.drawString(100, y, f"Agreement No: AGR{random.randint(1000, 9999)}")
        y -= 20
        c.drawString(100, y, f"Date: {fake.date_between(start_date='-1y', end_date='today')}")
        y -= 40
        
        # Parties
        c.drawString(100, y, f"Between: {fake.company()} (Service Provider)")
        y -= 20
        c.drawString(100, y, f"And: {fake.company()} (Client)")
        y -= 40
        
        # Terms
        c.drawString(100, y, "Terms:")
        y -= 20
        c.drawString(100, y, f"Service: {fake.catch_phrase()}")
        y -= 20
        c.drawString(100, y, f"Duration: {random.randint(1, 36)} months")
        y -= 20
        c.drawString(100, y, f"Payment: INR {random.randint(10000, 1000000)}")
        y -= 20
        c.drawString(100, y, f"Location: {fake.city()}")
        
        c.save()

if __name__ == "__main__":
    generate_agreement_pdf()
    print("20 Agreement PDFs generated!")