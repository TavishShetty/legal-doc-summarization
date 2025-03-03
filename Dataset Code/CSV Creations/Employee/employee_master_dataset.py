import csv
import random
import string
from faker import Faker

fake = Faker('en_IN')

def generate_employee_id():
    return 'EMP' + ''.join([str(random.randint(0, 9)) for _ in range(6)])

def generate_employee_data():
    data = []
    departments = ['HR', 'IT', 'Finance', 'Marketing', 'Operations']
    designations = ['Manager', 'Senior Executive', 'Junior Executive', 'Team Lead', 'Associate']
    
    for _ in range(2000):
        first_name = fake.first_name()
        last_name = fake.last_name()
        
        row = {
            'Employee ID': generate_employee_id(),
            'First Name': first_name,
            'Last Name': last_name,
            'Department': random.choice(departments),
            'Designation': random.choice(designations),
            'Joining Date': fake.date_between(start_date='-5y', end_date='today'),
            'Email': f"{first_name.lower()}.{last_name.lower()}@company.com",
            'Phone Number': '9' + ''.join([str(random.randint(0, 9)) for _ in range(9)]),
            'Office Location': fake.city(),
            'Salary': f"INR {random.randint(25000, 2000000)}",
            'Reporting Manager': fake.name(),
            'Performance Rating': random.randint(1, 5),
            'Years of Experience': random.randint(0, 20),
            'Education': random.choice(['B.Tech', 'MBA', 'B.Com', 'M.Tech', 'B.Sc']),
            'Emergency Contact': '9' + ''.join([str(random.randint(0, 9)) for _ in range(9)])
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
    employee_data = generate_employee_data()
    write_to_csv(employee_data, 'employee_master_data.csv')
    print("Employee dataset generated successfully!")