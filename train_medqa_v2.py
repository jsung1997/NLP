import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset, DatasetDict, Dataset
import evaluate
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import pandas as pd

# Load MedQA Dataset
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

# Label mapping
letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
dataset = dataset.map(lambda x: {"label": letter_to_index[x["answer_idx"]]})

# Stratified Split
df = dataset["train"].to_pandas()
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["answer_idx"], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

# Tokenizer & Model
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

# Sentence Embedding Model for FAISS
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 1. Build FAISS Index
corpus = [item["question"] + " " + " ".join(item["options"].values()) for item in dataset["train"]]
corpus_embeddings = embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=True)

import faiss
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

# 2. Preprocessing with FAISS retrieval
def preprocess(example):
    query_text = example["question"] + " " + " ".join(example["options"].values())
    query_embedding = embedder.encode([query_text], convert_to_numpy=True)
    
    _, retrieved_idx = index.search(query_embedding, k=3)
    retrieved_contexts = [corpus[i] for i in retrieved_idx[0]]
    context = "\n\n".join(retrieved_contexts)  # ✅ Use newline, not escaped string

    choices = list(example["options"].values())
    text_a = [f"Context: {context}\n\nQuestion: {example['question']}"] * 4
    text_b = [f"Answer: {opt}" for opt in choices]

    encoding = tokenizer(
        text_a,
        text_b,
        truncation=True,
        padding="max_length",
        max_length=384,
        return_tensors="pt"
    )
    return {
        "input_ids": encoding["input_ids"].tolist(),
        "attention_mask": encoding["attention_mask"].tolist(),
        "label": example["label"]
    }

# 3. Apply map
encoded_dataset = dataset.map(preprocess)

# Collator
def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch])
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch])
    labels = torch.tensor([ex["label"] for ex in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# DataLoaders
train_loader = DataLoader(encoded_dataset["train"], batch_size=4, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(encoded_dataset["validation"], batch_size=4, collate_fn=collate_fn)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*3)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_accuracy = 0.0  # To track the best val accuracy

for epoch in range(1, 21):  # Train for 20 epochs
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    total_loss = 0

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # 🔍 Evaluation after each epoch
    model.eval()
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        accuracy.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
        f1.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())

    val_accuracy = accuracy.compute()["accuracy"] * 100
    val_f1 = f1.compute(average="macro")["f1"] * 100

    print(f"✅ Epoch {epoch} - Val Accuracy: {val_accuracy:.2f}%, F1: {val_f1:.2f}%")

    # 💾 Save if best
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        print(f"🎉 Saving new best model (Acc: {val_accuracy:.2f}%)")
        model.save_pretrained("./saved_medqa_model_best")
        tokenizer.save_pretrained("./saved_medqa_model_best")

print(f"✅ Validation Accuracy: {accuracy.compute()['accuracy'] * 100:.2f}%")
print(f"✅ Validation F1 Score (macro): {f1.compute(average='macro')['f1'] * 100:.2f}%")