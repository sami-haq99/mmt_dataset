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

def retrieve(source_text, target_text = None,  top_k=TOP_K):
    with torch.no_grad():
        if target_text is not None:
            q_emb_tgt = model.encode_text([target_text], task="retrieval", return_numpy=True)
            q_emb = model.encode_text([source_text], task="retrieval", return_numpy=True)
            
            joint_emb = (q_emb + q_emb_tgt) / 2.0
            faiss.normalize_L2(joint_emb)
            distances, indices = index.search(joint_emb, top_k)
            results = [image_paths[i] for i in indices[0] if i != -1]
            return results
            
        else:
            q_emb = model.encode_text([source_text], task="retrieval", return_numpy=True)
        
            if not isinstance(q_emb, np.ndarray):
                # Manual fallback if return_numpy=True isn't supported in your version
                q_emb = q_emb.detach().cpu().numpy()
            faiss.normalize_L2(q_emb)

            distances, indices = index.search(q_emb, top_k)
            results = [image_paths[i] for i in indices[0] if i != -1]
            return results
