import numpy as np
import torch
import faiss
from transformers import AutoModel
from config import *

def load_model():
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4",
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    return model

def retrieve(query_text, top_k=TOP_K):

    # Load index
    index = faiss.read_index(INDEX_FILE)

    # Load metadata
    image_paths = np.load(IMAGE_PATHS_FILE, allow_pickle=True)

    # Load model
    model = load_model()

    with torch.no_grad():
        q_emb = model.encode_text([query_text], task="retrieval")
        q_emb = np.array(q_emb).astype("float32")

    faiss.normalize_L2(q_emb)
    # Important for IVF: set nprobe
    index.nprobe = 10

    distances, indices = index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(image_paths[idx])

    return results
