import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset, DatasetDict
import evaluate

# 1. Load saved model
model = AutoModelForMultipleChoice.from_pretrained("./saved_medqa_model")
tokenizer = AutoTokenizer.from_pretrained("./saved_medqa_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Load & preprocess dataset
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
dataset = dataset.map(lambda x: {"label": letter_to_index[x["answer_idx"]]})
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
})

def preprocess(example):
    question = example["question"]
    choices = list(example["options"].values())
    encoding = tokenizer(
        [question] * 4,
        choices,
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

def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch])
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch])
    labels = torch.tensor([ex["label"] for ex in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

eval_loader = DataLoader(encoded_dataset["validation"], batch_size=4, collate_fn=collate_fn)

# 3. Evaluation
metric = evaluate.load("accuracy")
model.eval()
for batch in eval_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    preds = torch.argmax(outputs.logits, dim=-1)
    metric.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())

# 4. Safe compute
if metric._num_examples == 0:
    print("⚠️ No predictions added — something went wrong.")
else:
    print("✅ Validation Accuracy:", metric.compute()["accuracy"] * 100)
