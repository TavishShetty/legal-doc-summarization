import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_rental_agreement_pdf():
    for i in range(20):
        c = canvas.Canvas(f"rental_agreement_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Rental Agreement")
        y -= 20
        c.drawString(100, y, f"Agreement No: RA{random.randint(1000, 9999)}")
        y -= 20
        c.drawString(100, y, f"Date: {fake.date_between(start_date='-1y', end_date='today')}")
        y -= 40
        
        # Parties
        c.drawString(100, y, f"Landlord: {fake.name()}")
        y -= 20
        c.drawString(100, y, f"Tenant: {fake.name()}")
        y -= 40
        
        # Terms
        c.drawString(100, y, f"Property: {fake.address().replace('\n', ', ')}")
        y -= 20
        rent = random.randint(5000, 50000)
        c.drawString(100, y, f"Monthly Rent: INR {rent}")
        y -= 20
        c.drawString(100, y, f"Duration: {random.randint(6, 36)} months")
        y -= 20
        c.drawString(100, y, f"Deposit: INR {rent * random.randint(1, 3)}")
        
        c.save()

if __name__ == "__main__":
    generate_rental_agreement_pdf()
    print("20 Rental Agreement PDFs generated!")