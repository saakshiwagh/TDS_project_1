import os
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ===== CONFIGURATION =====
input_chunks = "chunks.jsonl"
output_npz = "chunks_embeddings.npz"
model_name = "text-embedding-3-small"
batch_size = 100 

load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")

client = OpenAI(api_key=api_key, base_url=base_url)

# ===== LOAD CHUNKS =====
texts, sources, chunk_ids, types = [], [], [], []
with open(input_chunks, "r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        txt = chunk["content"]
        if isinstance(txt, str) and txt.strip() != "":
            texts.append(txt)
            sources.append(chunk["source"])
            chunk_ids.append(chunk.get("id") or chunk.get("chunk_id", ""))
            types.append(chunk.get("type", ""))
        else:
            print(f"[SKIP] Not a valid chunk: {chunk}")

# ===== BATCH UTILITY =====
def batcher(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ===== EMBEDDING =====
embeddings = []
for batch_texts in tqdm(list(batcher(texts, batch_size)), desc="Embedding chunks (batched)"):
    # Filter again just in case
    clean_batch = [txt for txt in batch_texts if isinstance(txt, str) and txt.strip() != ""]
    if not clean_batch:
        continue
    try:
        response = client.embeddings.create(
            model=model_name,
            input=clean_batch,
            encoding_format="float"
        )
        # Ensure API returns in order
        for emb in response.data:
            embeddings.append(emb.embedding)
    except Exception as e:
        print(f"[ERROR] Embedding batch failed: {e}")
        # Optionally: retry, skip, or save failed batch for later

# ===== SAVE =====
np.savez_compressed(
    output_npz,
    texts=np.array(texts),
    sources=np.array(sources),
    chunk_ids=np.array(chunk_ids),
    types=np.array(types),
    embeddings=np.array(embeddings, dtype=np.float32),
)
print(f"✅ Saved {len(texts)} embeddings to {output_npz}")
