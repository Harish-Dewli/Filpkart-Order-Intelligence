from pathlib import Path
import json, faiss
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parents[1]
model=SentenceTransformer("all-MiniLM-L6-v2")
index=faiss.read_index(str(ROOT/"part3/index/policies.faiss"))
chunks=json.loads((ROOT/"part3/index/chunks.json").read_text())
def retrieve(query,k=3):
    q=model.encode([query],normalize_embeddings=True).astype("float32")
    scores,ids=index.search(q,k)
    return [{**chunks[int(i)],"score":float(sc)} for sc,i in zip(scores[0],ids[0]) if i>=0]
