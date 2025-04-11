from transformers import pipeline

# Load models
summarizer = pipeline("summarization", model="nlpaueb/legal-bert-small-uncased")
anonymizer = pipeline("text-generation", model="tuner007/pegasus_paraphrase")

def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

def anonymize_text(text):
    anonymized = anonymizer(text, max_length=200)
    return anonymized[0]["generated_text"]
