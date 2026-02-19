import os
import torch

IMAGE_FOLDER = "./image-dataset"
IMAGE_PATHS_FILE = "image_paths.npy"
EMBEDDING_FILE = "image_embeddings.memmap"

EMBEDDING_DIM = 2048
BATCH_SIZE = 64
NUM_WORKERS = 4        # PyTorch DataLoader workers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IVF-PQ params for large scale
N_LIST = 1000   # 1000 coarse clusters
M = 64          # 32-dim PQ sub-vectors
N_BITS = 8      # 1 byte per subvector

TOP_K = 5
INDEX_FILE = "ivfpq_index_1M.index"
