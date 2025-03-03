import csv
import random
import string
from faker import Faker

# Initialize Faker with Indian locale
fake = Faker('en_IN')

def generate_aadhaar():
    return ''.join([str(random.randint(0, 9)) for _ in range(12)])

def generate_pan():
    return ''.join(random.choices(string.ascii_uppercase, k=5)) + ''.join(random.choices(string.digits, k=4)) + random.choice(string.ascii_uppercase)

def generate_phone():
    return '9' + ''.join([str(random.randint(0, 9)) for _ in range(9)])

def generate_bank_account():
    return ''.join([str(random.randint(0, 9)) for _ in range(12)])

def generate_card_number():
    return '4' + ''.join([str(random.randint(0, 9)) for _ in range(15)])  # Starting with 4 for Visa-like numbers

def generate_citizen_data():
    data = []
    for _ in range(2000):
        first_name = fake.first_name()
        middle_name = fake.first_name() if random.random() > 0.3 else ''
        last_name = fake.last_name()
        
        row = {
            'First Name': first_name,
            'Middle Name': middle_name,
            'Last Name': last_name,
            'Aadhaar Number': generate_aadhaar(),
            'PAN Number': generate_pan(),
            'Email': f"{first_name.lower()}.{last_name.lower()}@example.com",
            'Phone Number': generate_phone(),
            'Physical Address': fake.address().replace('\n', ', '),
            'Bank Account Number': generate_bank_account(),
            'Credit Card Number': generate_card_number(),
            'Salary': f"INR {random.randint(20000, 1500000)}",
            'Office Location': fake.city()
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
    citizen_data = generate_citizen_data()
    write_to_csv(citizen_data, 'indian_citizen_data.csv')
    print("Citizen dataset generated successfully!")