import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset
import random

# Load the model you trained
model_dir = "./saved_medqa_rag"
model = AutoModelForMultipleChoice.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)



#model_name = "GBaker/biolinkbert-base-medqa-usmle-nocontext"
#model = AutoModelForMultipleChoice.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load same dataset
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
dataset = dataset.map(lambda x: {"label": letter_to_index[x["answer_idx"]]})

# Select a small subset (8 samples)
small_subset = dataset["train"].shuffle(seed=42).select(range(8))

# Preprocessing function
def preprocess(example):
    question = example["question"]
    choices = list(example["options"].values())
    text_a = [f"Question: {question}"] * 4
    text_b = [f"Answer: {opt}" for opt in choices]

    encoding = tokenizer(
        text_a,
        text_b,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    return {
        "input_ids": encoding["input_ids"].tolist(),
        "attention_mask": encoding["attention_mask"].tolist(),
        "label": example["label"]
    }

# Encode the small subset
encoded = small_subset.map(preprocess)

# Collate function
def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch])
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch])
    labels = torch.tensor([ex["label"] for ex in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

loader = DataLoader(encoded, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Training for overfitting test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("🔍 Debug overfitting on small batch...")
for epoch in range(20):  # more epochs for small data
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
