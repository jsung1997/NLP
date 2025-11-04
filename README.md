

Efficient Low-Level Medical Question Answering (NLP Project)


Quantized LoRA Fine-Tuned LLM for MedQA
This project implements a Quantization + Low-Rank Adaptation (QLoRA) pipeline for fine-tuning a lightweight large language model (LLM) on the MedQA dataset. The goal is to train a parameter-efficient, domain-adapted model capable of answering medical examination-style questions under strict hardware constraints (single GPU).


Objective:
Fine-tune a compact, quantized LLM that maintains high accuracy on medical QA reasoning tasks while dramatically reducing GPU memory usage and training cost.



Key Features:

🔹 4-bit Quantization (bitsandbytes) — Compresses model weights to 4-bit without major performance loss.
🔹 LoRA Adapters (PEFT) — Injects small trainable matrices into attention layers for efficient adaptation.
🔹 Task-Specific Fine-Tuning (MedQA) — Adapts the base LLM to medical question-answering, focusing on reasoning and terminology.
🔹 Low GPU Memory Usage — Fine-tunes 7B-class models on a single RTX 3060 / 3090 GPU.
🔹 Evaluation Metrics — Reports accuracy, loss, and reasoning trace for each validation step.



Base Model (4-bit)  --->  LoRA Injected Layers  --->  Fine-Tuning on MedQA
        │                       │
        ▼                       ▼
 Quantization (bitsandbytes)    Adaptation (PEFT)



Dataset:

MedQA (USMLE-format medical exam)
Multiple-choice medical questions
Domain: physiology, pathology, pharmacology, diagnosis
Data split: train / validation / test
Tokenized using SentencePiece or LLaMA tokenizer




| Model            | Params | Precision    | GPU      | Accuracy (MedQA) | Memory    |
| ---------------- | ------ | ------------ | -------- | ---------------- | --------- |
| Base LLaMA-2-7B  | 7B     | FP16         | RTX 3090 | 43%              | ~28 GB    |
| QLoRA Fine-Tuned | 7B     | 4-bit + LoRA | RTX 3060 | **47%**          | **<8 GB** |

