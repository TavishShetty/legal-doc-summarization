import random
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker('en_IN')

def generate_medical_certificate_pdf():
    for i in range(2000):
        c = canvas.Canvas(f"medical_certificate_{i+1}.pdf", pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        # Header
        c.drawString(100, y, "Medical Certificate")
        y -= 20
        c.drawString(100, y, f"Cert No: MC{random.randint(1000, 9999)}")
        y -= 20
        c.drawString(100, y, f"Date: {fake.date_between(start_date='-1y', end_date='today')}")
        y -= 40
        
        # Details
        c.drawString(100, y, f"Patient: {fake.name()}")
        y -= 20
        c.drawString(100, y, f"Age: {random.randint(18, 80)}")
        y -= 20
        c.drawString(100, y, f"Condition: {random.choice(['Fever', 'Cold', 'Injury', 'Fatigue'])}")
        y -= 20
        c.drawString(100, y, f"Rest Advised: {random.randint(1, 15)} days")
        y -= 20
        c.drawString(100, y, f"Doctor: Dr. {fake.name()}")
        
        c.save()

if __name__ == "__main__":
    generate_medical_certificate_pdf()
    print("2000 Medical Certificate PDFs generated!")