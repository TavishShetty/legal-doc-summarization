import os
from anonymizer import anonymize_text, train_anonymizer
from summarizer import summarize_text, train_summarizer

def generate_training_data(data_folder):
    # Use your earlier synthetic PDF generation code
    # Example: Generate 2000 invoices, agreements, etc.
    pass

def train_models():
    data_folder = "data/"
    generate_training_data(data_folder)
    train_anonymizer(data_folder)
    train_summarizer(data_folder)

if __name__ == "__main__":
    train_models()