# train_eval_kormedmcqa.py

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForMultipleChoice,
    TrainingArguments, Trainer,
    DataCollatorWithPadding
)
import torch
import numpy as np
import evaluate

# 1. Load KorMedMCQA

#dataset = load_dataset("taeminlee/KorMedMCQA")
dataset = load_dataset("GBaker/medqa-us")


# 2. Model & Tokenizer
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

# 3. Preprocessing
def preprocess(example):
    question = example["question"]
    choices = example["choices"]
    inputs = tokenizer([question]*len(choices), choices, truncation=True, padding="max_length", max_length=256)
    inputs = {k: [v] for k, v in inputs.items()}  # Wrap into batch for each example
    inputs["label"] = example["answer_idx"]
    return inputs

encoded_dataset = dataset.map(preprocess, batched=False)
encoded_dataset = encoded_dataset.remove_columns([c for c in dataset["train"].column_names if c not in ["input_ids", "token_type_ids", "attention_mask", "label"]])

# 4. TrainingArguments
training_args = TrainingArguments(
    output_dir="./pubmedbert_kormedmcqa",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available()
)

# 5. Metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# 7. Train & Evaluate
trainer.train()
trainer.evaluate()
# 8. Save the model
trainer.save_model("./pubmedbert_kormedmcqa_final")
