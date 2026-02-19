import faiss
import numpy as np
from config import *

def main():
    # Load embeddings memmap
    embeddings = np.memmap(
        EMBEDDING_FILE,
        dtype="float32",
        mode="r",
        shape=(241, EMBEDDING_DIM)
    )
    print("Embedding shape:", embeddings.shape)

    # IVF-PQ: inner product for cosine similarity
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFPQ(
        quantizer,
        EMBEDDING_DIM,
        N_LIST,
        M,
        N_BITS
    )
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    # Training requires a sample subset if dataset is huge
    print("Training index on embeddings...")
    sample_size = min(100_000, embeddings.shape[0])
    index.train(embeddings[:sample_size])
    print("Training done.")

    print("Adding embeddings to index...")
    index.add(embeddings)
    print("Total indexed vectors:", index.ntotal)

    print("Saving FAISS index...")
    faiss.write_index(index, INDEX_FILE)
    print("Index saved:", INDEX_FILE)

if __name__ == "__main__":
    main()
