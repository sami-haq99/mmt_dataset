import os
import torch
import numpy as np
from transformers import AutoModel
import faiss
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4",
    trust_remote_code=True
).to(DEVICE)
model.eval()
model.task = "retrieval"


#create function which takes two images and return the similarity score between them
def calculate_similarity(image_path1, image_path2):
    from PIL import Image
    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")

    with torch.no_grad():
        emb1 = model.encode_image([img1], task="retrieval", return_numpy=True)
        emb2 = model.encode_image([img2], task="retrieval", return_numpy=True)

        if not isinstance(emb1, np.ndarray):
            emb1 = emb1.detach().cpu().numpy()
        if not isinstance(emb2, np.ndarray):
            emb2 = emb2.detach().cpu().numpy()

        faiss.normalize_L2(emb1)
        faiss.normalize_L2(emb2)

        similarity = np.dot(emb1, emb2.T).item()
    
    return similarity


#load json file and each item is a path of an image
def load_image_paths(json_file):
    list_of_candidate_images = []
    with open(json_file, "r") as f:
        data = json.load(f)
    
    for item in data:
        dlist = item[0]
        list_of_candidate_images.append(dlist)
    return list_of_candidate_images

#load the image paths from the images.txt file
def load_image_paths_from_txt(txt_file):
    image_dir = "/home/shaq/mmss/mmt_dataset/image-dataset/coco2017-images_testset/train2017-img/"
    image_paths = []
    with open(txt_file, "r") as f:
        for line in f:
            #search if the image exists in the image_dir, if yes, add the path to the list, if not, add the path from the txt file
            if os.path.exists(os.path.join(image_dir, line.strip())):
                image_paths.append(os.path.join(image_dir, line.strip()))
            else:
                image_paths.append('None')
    return image_paths

cand_images = load_image_paths("./eval_data/joint_dataset_fr.json")

ref_images = load_image_paths_from_txt("./eval_data/images.txt")

all_scores = []
for i in range(len(cand_images)):
    cand = cand_images[i]
    ref = ref_images[i]
    print(f"Calculating similarity for candidate: {cand} and reference: {ref}")
    if ref == 'None' or cand == 'None':
        print(f"Skipping similarity calculation for candidate: {cand} and reference: {ref} due to missing image.")
        continue
    sim_score = calculate_similarity(cand, ref)
    all_scores.append(sim_score)
    #print(f"Similarity between {cand} and {ref}: {sim_score}")
        
print("Average Similarity Score:", np.mean(all_scores))

#save similarity scores in a text file
with open("./eval_data/similarity_joint_scores.txt", "w") as f:
    #write the average similarity score at the top of the file
    f.write(f"Average Similarity Score: {np.mean(all_scores)}\n")
    for score in all_scores:
        f.write(f"{score}\n")