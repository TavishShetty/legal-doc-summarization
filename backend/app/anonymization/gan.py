import argparse
import pdfplumber
import spacy
import re
import torch
import torch.nn as nn
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
import json

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 100),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models (for demo, these would normally need to be trained)
generator = Generator()
discriminator = Discriminator()

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"pdfplumber failed: {e}, falling back to OCR")
        images = convert_from_path(pdf_path)
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"
    return text

# NER with spaCy and custom rules
nlp = spacy.load("en_core_web_sm")

def detect_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def add_custom_rules(text):
    patterns = {
        "AADHAAR": r"\b\d{4}\s*\d{4}\s*\d{4}\b|\b\d{12}\b",
        "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "ACCOUNT_NUMBER": r"\b\d{9,18}\b",
        "PHONE": r"\b(\+91[-\s]?)?\d{10}\b",
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "MONEY": r"\b(?:Rs\.?|INR)\s?\d+(?:,\d{3})*(?:\.\d{2})?\b",
        "ADDRESS": r"\b\d{1,5}\s+[A-Za-z\s]+(?:Road|Street|Lane|Avenue|Colony|Nagar|Sector|Phase)[,\s]+[A-Za-z\s]+[,\s]+\d{6}\b"
    }
    custom_entities = []
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            custom_entities.append((match.group(), label))
    return custom_entities

def get_all_entities(text):
    return detect_entities(text) + add_custom_rules(text)

# Generate fake data
def generate_fake_data(model, input_dim=100):
    noise = torch.randn((1, input_dim))
    with torch.no_grad():
        fake_data = model(noise)
    return str(fake_data.numpy())  # Simplification: convert tensor to string

# Anonymize text
def anonymize_text(text, entities):
    anonymized_text = text
    for entity, label in entities:
        if label in ["PERSON", "ADDRESS", "EMAIL", "PAN", "AADHAAR"]:
            fake_data = generate_fake_data(generator)
            anonymized_text = anonymized_text.replace(entity, fake_data)
    return anonymized_text

# Main function
def main(input_file, output_file):
    text = extract_text_from_pdf(input_file)
    entities = get_all_entities(text)
    anonymized_text = anonymize_text(text, entities)
    with open(output_file, "w") as f:
        f.write(anonymized_text)
    print(f"Anonymized document saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Anonymizer using GAN")
    parser.add_argument("input_file", help="Path to the input PDF file")
    parser.add_argument("output_file", help="Path to the output text file")
    args = parser.parse_args()
    main(args.input_file, args.output_file)
