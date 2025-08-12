# app/build_index.py
import os, pickle
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import json

DOC_DIR = "data/raw/knowledge"   # directory of .txt files or a JSONL
OUT_INDEX = "data/embedded/knowledge.faiss"
OUT_DOCSTORE = "data/embedded/doc_store.pkl"

# load docs
docs = []
meta = []
for fname in sorted(os.listdir(DOC_DIR)):
    path = os.path.join(DOC_DIR, fname)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    # optional: create small title from filename
    docs.append(text)
    meta.append({'id': fname, 'source': 'trusted', 'filename': fname})

# chunk documents into ~500-word chunks with 50-word overlap
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks

all_chunks = []
all_meta = []
for i, d in enumerate(docs):
    chunks = chunk_text(d, chunk_size=400, overlap=60)
    for j, c in enumerate(chunks):
        all_chunks.append(c)
        all_meta.append({'doc_id': meta[i]['id'], 'chunk_id': j})

# embed
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

# build index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

os.makedirs("data/embedded", exist_ok=True)
faiss.write_index(index, OUT_INDEX)
with open(OUT_DOCSTORE, "wb") as f:
    pickle.dump({'chunks': all_chunks, 'meta': all_meta}, f)

print(f"Saved index with {len(all_chunks)} chunks")
