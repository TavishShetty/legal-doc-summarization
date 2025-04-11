import argparse
import pdfplumber
import spacy
import re
import subprocess
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
import json

# Extract text (with OCR fallback for scanned PDFs)
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

    if not text.strip():
        print("No selectable text found, using Tesseract OCR...")
        images = convert_from_path(pdf_path)
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"
    
    print("Extracted Text:", text)
    return text

# NER with spaCy and custom rules
nlp = spacy.load("en_core_web_sm")

def detect_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print("spaCy Entities:", entities)
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
    print("Custom Entities:", custom_entities)
    return custom_entities

def get_all_entities(text):
    return detect_entities(text) + add_custom_rules(text)

# LLaMA 2 via Ollama for pseudonymization
def generate_pseudonym(prompt="Generate a fake name: "):
    result = subprocess.run(
        ["ollama", "run", "llama2:7b", prompt],
        capture_output=True,
        text=True
    )
    generated_text = result.stdout.strip()
    return generated_text if generated_text else "Fake Name"

# Anonymization function with all methods
def anonymize_text(text, entities, method="pseudonymize"):
    anonymized_text = text
    entity_map = {}
    for entity, label in entities:
        if method == "pseudonymize":
            if label == "PERSON" and entity not in entity_map:
                entity_map[entity] = generate_pseudonym()
            elif label == "ADDRESS" and entity not in entity_map:
                entity_map[entity] = "123 Fake Street, Anonymized City, 000000"
        elif method == "mask":
            if label == "AADHAAR" and entity not in entity_map:
                clean_entity = ''.join(entity.split())
                if len(clean_entity) == 12 and clean_entity.isdigit():
                    entity_map[entity] = "XXXX-XXXX-" + clean_entity[-4:]
            elif label == "PAN" and entity not in entity_map:
                entity_map[entity] = "XXXXX" + entity[-5:]
            elif label == "ACCOUNT_NUMBER" and entity not in entity_map:
                entity_map[entity] = "XXXX-XXXX-" + entity[-4:]
            elif label == "PHONE" and entity not in entity_map:
                entity_map[entity] = "XXXXXX" + entity[-4:]
            elif label == "EMAIL" and entity not in entity_map:
                entity_map[entity] = entity.split('@')[0][:3] + "XXX@" + entity.split('@')[1]
            elif label == "MONEY" and entity not in entity_map:
                entity_map[entity] = "[MASKED_AMOUNT]"
        elif method == "redact":
            if label in ["PERSON", "AADHAAR", "PAN", "ACCOUNT_NUMBER", "PHONE", "EMAIL", "DATE", "MONEY", "ADDRESS"] and entity not in entity_map:
                entity_map[entity] = "[REDACTED]"
        if entity in entity_map:
            anonymized_text = anonymized_text.replace(entity, entity_map[entity])
    print("Anonymized Text:", anonymized_text)
    return anonymized_text, entity_map

# Batch anonymization for multiple PDFs
def batch_anonymize(input_dir, output_dir, method="pseudonymize"):
    os.makedirs(output_dir, exist_ok=True)
    for pdf_file in os.listdir(input_dir):
        if pdf_file.endswith(".pdf"):
            input_path = os.path.join(input_dir, pdf_file)
            output_path = os.path.join(output_dir, pdf_file.replace(".pdf", ".txt"))
            text = extract_text_from_pdf(input_path)
            entities = get_all_entities(text)
            anonymized_text, _ = anonymize_text(text, entities, method=method)
            with open(output_path, "w") as f:
                f.write(anonymized_text)
            print(f"Processed {pdf_file} -> {output_path}")

# Create training dataset (modified to use existing outputs)
def create_dataset(input_dir, output_dir, dataset_file="train.jsonl"):
    dataset = []
    for pdf_file in os.listdir(input_dir):
        if pdf_file.endswith(".pdf"):
            input_path = os.path.join(input_dir, pdf_file)
            output_path = os.path.join(output_dir, pdf_file.replace(".pdf", ".txt"))
            orig_text = extract_text_from_pdf(input_path)
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    anon_text = f.read()
                dataset.append({"input": orig_text, "output": anon_text})
            else:
                print(f"Warning: {output_path} not found, skipping.")
    with open(dataset_file, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"Dataset saved to {dataset_file}")

# Main function for single PDF processing
def main(input_file, output_file, method):
    text = extract_text_from_pdf(input_file)
    entities = get_all_entities(text)
    anonymized_text, entity_map = anonymize_text(text, entities, method=method)
    with open(output_file, "w") as f:
        f.write(anonymized_text)
    print(f"Anonymized document saved to {output_file}")
    return text, anonymized_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize documents with LLaMA 2")
    parser.add_argument("--batch", action="store_true", help="Run batch anonymization")
    parser.add_argument("--input_dir", default="/Users/puneetkohli/Downloads/Capstone 2025/legal-doc-summarization/backend/app/anonymization/pdf_data")
    parser.add_argument("--output_dir", default="/Users/puneetkohli/Downloads/Capstone 2025/legal-doc-summarization/backend/app/anonymization/anonymized_output")
    parser.add_argument("input_file", nargs="?", help="Path to single input PDF")
    parser.add_argument("output_file", nargs="?", help="Path to single output text file")
    parser.add_argument("--method", choices=["mask", "pseudonymize", "redact"], default="pseudonymize")
    args = parser.parse_args()

    if args.batch:
        batch_anonymize(args.input_dir, args.output_dir, args.method)
        create_dataset(args.input_dir, args.output_dir)
    elif args.input_file and args.output_file:
        main(args.input_file, args.output_file, args.method)
    else:
        print("Please specify --batch or input/output files.")