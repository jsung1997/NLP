import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load CSV and clean
df = pd.read_csv("eng_kor_data.csv")
df = df[df["src"].notnull() & df["tgt"].notnull()]
df["src"] = df["src"].astype(str)
df["tgt"] = df["tgt"].astype(str)
df = df[df["src"].str.strip() != ""]
df = df[df["tgt"].str.strip() != ""]

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
split = dataset.train_test_split(test_size=0.1)
train_ds = split["train"]
val_ds = split["test"]

# Model names
teacher_model_name = "facebook/nllb-200-1.3B"
student_model_name = "facebook/nllb-200-distilled-600M"
checkpoint_dir = "student_nllb_enko_distilled"

# Load teacher model (for future use if needed)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name).to(device)

# Load or resume student model
if os.path.exists(checkpoint_dir):
    print("🔄 Resuming training from previous checkpoint...")
    student_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    student_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir).to(device)
else:
    print("🆕 Starting from pretrained distilled model...")
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_name).to(device)

# Language codes
SRC_LANG = "eng_Latn"
TGT_LANG = "kor_Hang"

# Preprocessing function
def tokenize(example):
    try:
        teacher_tokenizer.src_lang = SRC_LANG
        teacher_tokenizer.tgt_lang = TGT_LANG
        student_tokenizer.src_lang = SRC_LANG
        student_tokenizer.tgt_lang = TGT_LANG

        teacher_inputs = teacher_tokenizer(
            example["src"], return_tensors="pt", padding="max_length", truncation=True, max_length=128
        )
        input_enc = student_tokenizer(
            example["src"], return_tensors="pt", padding="max_length", truncation=True, max_length=128
        )
        target_enc = student_tokenizer(
            example["tgt"], return_tensors="pt", padding="max_length", truncation=True, max_length=128
        )

        # Print decoded label for sanity check
        print("🔎 SRC:", example["src"])
        print("🔎 TGT:", example["tgt"])
        print("🧾 Decoded label:", student_tokenizer.decode(target_enc["input_ids"][0]))

        return {
            "teacher_input_ids": teacher_inputs["input_ids"].squeeze(0),
            "teacher_attention_mask": teacher_inputs["attention_mask"].squeeze(0),
            "student_input_ids": input_enc["input_ids"].squeeze(0),
            "student_attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": target_enc["input_ids"].squeeze(0),
        }
    except Exception as e:
        print(f"⚠️ Tokenization error: {e}")
        return None

# Tokenize dataset
train_tokens = []
for ex in train_ds:
    result = tokenize(ex)
    if result is not None:
        train_tokens.append(result)

print(f"✅ Loaded {len(train_tokens)} tokenized examples.")

if len(train_tokens) == 0:
    raise ValueError("No valid training samples found.")

train_loader = DataLoader(train_tokens, batch_size=2, shuffle=True)

# Sanity check: student output before training
student_model.eval()
student_tokenizer.src_lang = SRC_LANG
student_tokenizer.tgt_lang = TGT_LANG
forced_bos = student_tokenizer("", return_tensors="pt")["input_ids"][0][0].item()
test_input = student_tokenizer("I love you", return_tensors="pt").to(device)
out = student_model.generate(**test_input, forced_bos_token_id=forced_bos)
print("🧪 Before Training Output:", student_tokenizer.batch_decode(out, skip_special_tokens=True)[0])

# Optimizer
optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

# Training config
max_epochs = 3
max_steps_per_epoch = None

# Training loop (CE loss only)
student_model.train()
for epoch in range(max_epochs):
    total_loss = 0
    steps = 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
        for batch in pbar:
            input_ids = batch["student_input_ids"].to(device)
            attention_mask = batch["student_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            ce_loss = outputs.loss
            ce_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += ce_loss.item()
            steps += 1

            # Update dynamic progress bar
            pbar.set_postfix({"CE Loss": f"{ce_loss.item():.4f}"})

            if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
                break

    print(f"✅ Epoch {epoch+1} | Avg Loss: {total_loss / steps:.4f}")

# Save model
student_model.save_pretrained(checkpoint_dir)
student_tokenizer.save_pretrained(checkpoint_dir)
print("✅ Saved distilled student model.")
