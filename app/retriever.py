# app/retriever.py
import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/embedded/knowledge.faiss"
DOC_STORE_PATH = "data/embedded/doc_store.pkl"

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
with open(DOC_STORE_PATH, "rb") as f:
    docstore = pickle.load(f)   # {'chunks': [...], 'meta':[...]}
chunks = docstore['chunks']
meta = docstore['meta']

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_context(query: str, k: int = 3):
    qvec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(qvec), k)
    results = []
    for idx in I[0]:
        results.append({
            "text": chunks[idx],
            "meta": meta[idx]
        })
    return results
