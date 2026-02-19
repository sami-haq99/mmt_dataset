import os
import numpy as np
import torch
import faiss
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
from config import *

def get_image_paths(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                paths.append(os.path.join(root, f))
    return paths

def load_image(path):
    return Image.open(path).convert("RGB")

def main():

    print("Loading model...")
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4",
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    print("Collecting images...")
    image_paths = get_image_paths(IMAGE_FOLDER)
    np.save(IMAGE_PATHS_FILE, image_paths)

    print(f"Found {len(image_paths)} images")

    embeddings = []

    print("Generating embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
            batch_paths = image_paths[i:i+BATCH_SIZE]
            images = [load_image(p) for p in batch_paths]

            emb = model.encode_image(images, task="retrieval", return_numpy=True)

            if not isinstance(emb, np.ndarray):
                # Manual fallback if return_numpy=True isn't supported in your version
                emb = emb.detach().cpu().numpy()
            
            embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")
    
    faiss.normalize_L2(embeddings)

    print("Building IndexIVFPQ...")

    # Step 1: Coarse quantizer
    quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)

    # Step 2: IVF-PQ index
    index = faiss.IndexIVFPQ(
        quantizer,
        EMBEDDING_DIM,
        N_LIST,
        M,
        N_BITS
    )

    # Step 3: Train
    print("Training index...")
    index.train(embeddings)

    # Step 4: Add vectors
    index.add(embeddings)

    print(f"Total indexed vectors: {index.ntotal}")

    # Save
    faiss.write_index(index, INDEX_FILE)

    print("Index saved!")

if __name__ == "__main__":
    main()
