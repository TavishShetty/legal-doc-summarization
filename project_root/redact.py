import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import requests
import random
import time

# Hugging Face API setup
API_TOKEN = "REMOVED"  # Your API token
NER_API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
LLM_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to query Hugging Face API with retries
def query_hf_api(url, payload, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=HEADERS, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 503:
                print(f"API unavailable (503), retrying {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            else:
                print(f"API error: {e}")
                return None
    print("Max retries reached. Falling back to default behavior.")
    return None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if not text.strip():
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return text

# AI-driven anonymization function using HF API
def ai_anonymize_text(text):
    # Step 1: Identify entities with NER via API
    ner_payload = {"inputs": text}
    entities = query_hf_api(NER_API_URL, ner_payload)
    if not entities or isinstance(entities, dict) and "error" in entities:
        print("NER API failed or no entities detected. Proceeding with original text.")
        return text
    
    # Step 2: Prepare text segments and entity metadata
    segments = []
    last_end = 0
    for entity in sorted(entities, key=lambda x: x["start"]):
        start, end = entity["start"], entity["end"]
        entity_text = entity["word"]
        entity_type = entity["entity_group"]
        
        if last_end < start:
            segments.append({"text": text[last_end:start], "action": "keep"})
        
        segments.append({"text": entity_text, "action": None, "type": entity_type, "start": start, "end": end})
        last_end = end
    
    if last_end < len(text):
        segments.append({"text": text[last_end:], "action": "keep"})
    
    # Step 3: Use LLM API to decide anonymization strategy
    for segment in segments:
        if segment["action"] is None:
            entity_text = segment["text"]
            entity_type = segment["type"]
            context = f"Given this entity '{entity_text}' (type: {entity_type}) in a document, should it be pseudonymized, masked, or redacted? Provide a one-word decision."
            llm_payload = {"inputs": context, "parameters": {"max_length": 50}}
            decision_response = query_hf_api(LLM_API_URL, llm_payload)
            
            # Handle variable API response formats
            decision = "redact"  # Default fallback
            if decision_response:
                if isinstance(decision_response, list) and decision_response:
                    # Standard format: [{"generated_text": "..."}]
                    if isinstance(decision_response[0], dict) and "generated_text" in decision_response[0]:
                        decision = decision_response[0]["generated_text"].split()[-1].lower()
                    # Possible alternative: list of strings
                    elif isinstance(decision_response[0], str):
                        decision = decision_response[0].split()[-1].lower()
                elif isinstance(decision_response, dict) and "generated_text" in decision_response:
                    # Single dict response
                    decision = decision_response["generated_text"].split()[-1].lower()
            
            # Interpret LLM decision
            if decision == "pseudonymized":
                segment["action"] = "pseudonymize"
            elif decision == "masked":
                segment["action"] = "mask"
            elif decision == "redacted":
                segment["action"] = "redact"
            else:
                segment["action"] = "redact"  # Default fallback
    
    # Step 4: Apply anonymization based on AI decisions
    anonymized_text = ""
    for segment in segments:
        text_segment = segment["text"]
        action = segment["action"]
        
        if action == "keep":
            anonymized_text += text_segment
        elif action == "pseudonymize":
            anonymized_text += "Person_" + str(random.randint(1, 1000))
        elif action == "mask":
            anonymized_text += "X" * len(text_segment)
        elif action == "redact":
            anonymized_text += "[REDACTED]"
    
    return anonymized_text

# Main function to process and anonymize a PDF
def anonymize_pdf(pdf_path, output_path):
    original_text = extract_text_from_pdf(pdf_path)
    if not original_text:
        print("No text could be extracted from the PDF.")
        return
    
    anonymized_text = ai_anonymize_text(original_text)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(anonymized_text)
    print(f"Anonymized PDF saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    pdf_path = "Rishikesh Vadodaria.pdf"  # Your PDF file path
    output_path = "anonymized_output.txt"
    anonymize_pdf(pdf_path, output_path)