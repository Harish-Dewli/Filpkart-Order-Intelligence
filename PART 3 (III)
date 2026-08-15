from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[1]
POLICIES=json.loads((ROOT/"part3/policies.json").read_text())
CHUNKS=[]
for d in POLICIES:
    for i,s in enumerate([x.strip() for x in d["text"].split(". ") if x.strip()]):
        if not s.endswith("."): s += "."
        CHUNKS.append({"chunk_id":f"{d['id']}_{i}","doc_id":d["id"],"text":s})

MODEL_NAME="all-MiniLM-L6-v2"
embedder=SentenceTransformer(MODEL_NAME)
vec=embedder.encode([c["text"] for c in CHUNKS],normalize_embeddings=True).astype("float32")
index=faiss.IndexFlatIP(vec.shape[1]); index.add(vec)

(ROOT/"part3/index").mkdir(exist_ok=True)
faiss.write_index(index,str(ROOT/"part3/index/policies.faiss"))
(ROOT/"part3/index/chunks.json").write_text(json.dumps(CHUNKS,indent=2))

def retrieve(query,k=3):
    q=embedder.encode([query],normalize_embeddings=True).astype("float32")
    scores,ids=index.search(q,k)
    return [{**CHUNKS[int(i)],"score":float(s)} for s,i in zip(scores[0],ids[0]) if i>=0]
