from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from safetensors.torch import save_file
import torch
import os

# === CONFIG ===
model_id = "facebook/nllb-200-distilled-600M"  # You can change this to any HF model with .bin weights
save_path = "./opus-mt-ko-en-safetensors"  # Local path to save the converted model

# === STEP 1: Load model and tokenizer ===
print(f"Loading model from: {model_id}")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# === STEP 2: Save model in safetensors format ===
print(f"Saving model in safetensors format to: {save_path}")
os.makedirs(save_path, exist_ok=True)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path, safe_serialization=True)

print("\n✅ Conversion complete.")
print(f"Use it in your code with: AutoModelForSeq2SeqLM.from_pretrained('{save_path}', use_safetensors=True)")
