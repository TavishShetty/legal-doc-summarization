from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch
import bitsandbytes

# Paths
base_model_name = "openlm-research/open_llama_7b"  # Base model from Hugging Face
adapter_path = "/Users/puneetkohli/Downloads/Capstone 2025/legal-doc-summarization/backend/app/anonymization/lora_llama2_final"  # Path to your fine-tuned adapters

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(adapter_path)  # Use fine-tuned tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,  # Match your training setup
    device_map="mps",   # Use MPS on Mac Mini
    torch_dtype=torch.float16
)

# Load and merge LoRA adapters
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()  # Merge adapters into base model for inference

# Test the model
test_text = "Name: John Doe\nAadhaar: 1234 5678 9012"
inputs = tokenizer(test_text, return_tensors="pt").to("mps")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))