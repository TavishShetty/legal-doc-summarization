import gguf
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os

gguf_path = "/Users/puneetkohli/.ollama/models/blobs/sha256-8934d96d3f08982e95922b2b7a2c626a1fe873d7c3b06e8e56d7bc0a1fef9246"
output_dir = "llama2-7b-hf"

# Load GGUF file
reader = gguf.GGUFReader(gguf_path)
print("GGUF Metadata:", reader.metadata)

# This is a simplified approach; actual weight mapping requires more work
# For now, assume it’s LLaMA 2 7B and use a base HF model
tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_7b")  # Placeholder
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_7b", torch_dtype=torch.float16)

# Save to HF format (weights need proper mapping from GGUF)
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")