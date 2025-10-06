import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index + texts
index = faiss.read_index("medqa_index.faiss")
with open("medqa_texts.pkl", "rb") as f:
    texts = pickle.load(f)

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_context(query, top_k=3):
    query_vec = encoder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(query_vec, top_k)
    return [texts[i] for i in I[0]]
