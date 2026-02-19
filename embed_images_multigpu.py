import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from config import *

# -------------------
# Dataset for PyTorch
# -------------------
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return img, idx  # return index for writing to memmap

# -------------------
# Worker collate
# -------------------
def collate_fn(batch):
    images, indices = zip(*batch)
    return list(images), list(indices)

# -------------------
# Main embedding
# -------------------
def main():
    print("Collecting image paths...")
    image_paths = []
    for root, _, files in os.walk(IMAGE_FOLDER):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, f))
    n_images = len(image_paths)
    print(f"Found {n_images} images")
    np.save(IMAGE_PATHS_FILE, image_paths)

    # Memory-mapped embeddings
    embeddings = np.memmap(
        EMBEDDING_FILE,
        dtype="float32",
        mode="w+",
        shape=(n_images, EMBEDDING_DIM)
    )

    # Load model on all GPUs using DataParallel
    print("Loading Jina Embeddings v4 model on multiple GPUs...")
    device = torch.device(DEVICE)
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4",
        trust_remote_code=True
    )
    model.task = "retrieval"
    model.eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # DataLoader
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Batch embedding
    with torch.no_grad():
        for images, indices in tqdm(dataloader, total=len(dataloader)):
            emb = model.module.encode_image(images, task="retrieval") \
                if isinstance(model, torch.nn.DataParallel) else model.encode_image(images, task="retrieval")

            # Convert to numpy
            if isinstance(emb, list):
                emb = np.array(emb).astype("float32")
            elif isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy().astype("float32")
            else:
                raise TypeError(f"Unexpected type for embeddings: {type(emb)}")

            # Normalize
            faiss.normalize_L2(emb)

            # Write to memmap
            for idx, e in zip(indices, emb):
                embeddings[idx] = e

    print("Embeddings saved to memmap:", EMBEDDING_FILE)

if __name__ == "__main__":
    main()
