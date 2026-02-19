import os

IMAGE_FOLDER = "./images"  # put <=500 test images here
INDEX_FILE = "ivfpq.index"
IMAGE_PATHS_FILE = "image_paths.npy"

EMBEDDING_DIM = 1024
BATCH_SIZE = 16
DEVICE = "cuda"  # change to "cpu" if no GPU

# IVF-PQ parameters (safe for 500 test)
N_LIST = 50        # number of coarse clusters
M = 16             # number of PQ subvectors
N_BITS = 8         # bits per subvector

TOP_K = 5
