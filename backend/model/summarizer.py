from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def summarize_text(text):
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer(text, truncation=True, return_tensors="pt", max_length=512)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Training placeholder (run separately)
def train_summarizer(data_folder):
    # Use synthetic PDFs from data/ to fine-tune Pegasus
    pass