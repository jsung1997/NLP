import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset, DatasetDict, Dataset
import evaluate
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load dataset
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

# 2. Convert answer_idx from letters to label index
letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
dataset = dataset.map(lambda x: {"label": letter_to_index[x["answer_idx"]]})

# 3. Stratified train/validation split using sklearn
df = dataset["train"].to_pandas()
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["answer_idx"], random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# 4. Load tokenizer and model
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

# 5. Preprocessing
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

encoded_dataset = dataset.map(preprocess)

# 6. Collate function
def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch])
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch])
    labels = torch.tensor([ex["label"] for ex in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 7. DataLoaders
train_loader = DataLoader(encoded_dataset["train"], batch_size=4, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(encoded_dataset["validation"], batch_size=4, collate_fn=collate_fn)

# 8. Optimizer & scheduler with improved learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*3)

# 9. Training with debug prints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(3):  # Increase epochs for better learning
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

# Save model
model.save_pretrained("./saved_medqa_model")
tokenizer.save_pretrained("./saved_medqa_model")

# Evaluation
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

model.eval()
for batch in eval_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    preds = torch.argmax(outputs.logits, dim=-1)
    accuracy.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
    f1.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())

# Print results
print(f"✅ Validation Accuracy: {accuracy.compute()['accuracy'] * 100:.2f}%")
print(f"✅ Validation F1 Score (macro): {f1.compute(average='macro')['f1'] * 100:.2f}%")
