from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)
import torch
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
import numpy as np

# Initialize constants at the top
MODEL_NAME = "google/pegasus-xsum"
TOKENIZED_DATASET_PATH = "tokenized_dataset_pegasus.pt"

# Load tokenizer and model first
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)

# Load dataset
try:
    tokenized_dataset = torch.load(TOKENIZED_DATASET_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Tokenized dataset not found at {TOKENIZED_DATASET_PATH}. "
        "Please run datasets.ipynb first."
    )

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

# Custom callback
class LegalSummaryCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        pass

# Metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
    
    return {
        'rouge1': np.mean([s['rouge1'].fmeasure for s in scores]),
        'rouge2': np.mean([s['rouge2'].fmeasure for s in scores]),
        'rougeL': np.mean([s['rougeL'].fmeasure for s in scores]),
    }

# Training config
training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints_pegasus",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=3e-5,
    num_train_epochs=3,
    warmup_steps=200,
    logging_dir="./logs",
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=2,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[LegalSummaryCallback()]
)

# Enable gradient checkpointing and train
model.gradient_checkpointing_enable()
trainer.train()

# Add inference function at the bottom
def generate_summary(text, model=model, tokenizer=tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(**inputs)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Test with a sample if run directly
if __name__ == "__main__":
    sample_text = "The Supreme Court held that the right to privacy is a fundamental right under the Constitution of India."
    print(generate_summary(sample_text))
    