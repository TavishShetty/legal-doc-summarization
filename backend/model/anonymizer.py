import re

def anonymize_text(text):
    # Replace with your friend's actual ML model
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
    text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
    text = re.sub(r'\d+ [A-Za-z]+ (Street|Avenue|Road)', '[ADDRESS]', text)
    return text