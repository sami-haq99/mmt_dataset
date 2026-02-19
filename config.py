import os
import torch

IMAGE_FOLDER = "/path/to/1M_images"
IMAGE_PATHS_FILE = "image_paths.npy"
EMBEDDING_FILE = "image_embeddings.memmap"

EMBEDDING_DIM = 2048
BATCH_SIZE = 64
NUM_WORKERS = 8         # PyTorch DataLoader workers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IVF-PQ params for large scale
N_LIST = 8192
M = 64
N_BITS = 8
TOP_K = 10
INDEX_FILE = "ivfpq_index_1M.index"
