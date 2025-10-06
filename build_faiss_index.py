import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pickle

# Load dataset
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")["train"]

# Load sentence embedding model
encoder = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

# Prepare texts (question + options)
texts = []
for example in dataset:
    q = example["question"]
    opts = " ".join(example["options"].values())
    texts.append(f"{q} {opts}")

# Encode and normalize
embeddings = encoder.encode(texts, normalize_embeddings=True)

# Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))

# Save index and metadata
faiss.write_index(index, "medqa_index.faiss")
with open("medqa_texts.pkl", "wb") as f:
    pickle.dump(texts, f)
