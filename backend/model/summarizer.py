def summarize_text(text):
    # Replace with your friend's actual ML model
    if "INVOICE" in text.upper():
        return "Invoice with billing and payment details."
    elif "AGREEMENT" in text.upper():
        return "Legal agreement between parties."
    elif "BANK STATEMENT" in text.upper():
        return "Bank statement with transaction history."
    elif "MEDICAL" in text.upper():
        return "Medical record with patient details."
    return "Summary of document content."