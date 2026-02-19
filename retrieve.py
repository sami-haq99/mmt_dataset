import faiss
import numpy as np
from transformers import AutoModel
from config import *

DEVICE = DEVICE

# Load index and metadata
index = faiss.read_index(INDEX_FILE)
image_paths = np.load(IMAGE_PATHS_FILE, allow_pickle=True)

# IVF-PQ: set nprobe
index.nprobe = 32

# Load model once
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4",
    trust_remote_code=True
).to(DEVICE)
model.eval()
model.task = "retrieval"

def retrieve(query_text, top_k=TOP_K):
    with torch.no_grad():
        q_emb = model.encode_text([query_text], task="retrieval")
        if isinstance(q_emb, list):
            q_emb = np.array(q_emb).astype("float32")
        else:
            q_emb = q_emb.detach().cpu().numpy().astype("float32")
        faiss.normalize_L2(q_emb)

    distances, indices = index.search(q_emb, top_k)
    results = [image_paths[i] for i in indices[0] if i != -1]
    return results
