import csv
import random
import string
from faker import Faker

fake = Faker('en_IN')

def generate_vendor_id():
    return 'VEN' + ''.join([str(random.randint(0, 9)) for _ in range(6)])

def generate_gst():
    return ''.join(random.choices(string.digits, k=2)) + ''.join(random.choices(string.ascii_uppercase, k=2)) + ''.join(random.choices(string.digits, k=6))

def generate_vendor_data():
    data = []
    vendor_types = ['Supplier', 'Service Provider', 'Contractor', 'Consultant']
    payment_terms = ['Net 30', 'Net 45', 'Net 60', 'Advance']
    
    for _ in range(2000):
        company_name = fake.company()
        
        row = {
            'Vendor ID': generate_vendor_id(),
            'Company Name': company_name,
            'Vendor Type': random.choice(vendor_types),
            'GST Number': generate_gst(),
            'Contact Person': fake.name(),
            'Email': f"contact@{company_name.lower().replace(' ', '')}.com",
            'Phone Number': '9' + ''.join([str(random.randint(0, 9)) for _ in range(9)]),
            'Address': fake.address().replace('\n', ', '),
            'Bank Account': ''.join([str(random.randint(0, 9)) for _ in range(12)]),
            'Payment Terms': random.choice(payment_terms),
            'Contract Start Date': fake.date_between(start_date='-2y', end_date='today'),
            'Annual Contract Value': f"INR {random.randint(100000, 50000000)}",
            'Service Rating': random.randint(1, 5),
            'Last Transaction Date': fake.date_between(start_date='-1y', end_date='today'),
            'Primary Location': fake.city()
        }
        data.append(row)
    return data

def write_to_csv(data, filename):
    fieldnames = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    vendor_data = generate_vendor_data()
    write_to_csv(vendor_data, 'vendor_master_data.csv')
    print("Vendor dataset generated successfully!")