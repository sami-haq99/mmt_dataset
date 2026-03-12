import csv
import os
import torch
import numpy as np
from transformers import AutoModel
import faiss
import json

# This script evaluates the similarity between candidate (returned by the index) images and reference image (Coco test dataset)
# This similarity score can be used to evaluate the performance of the retrieval system and comparison with human evaluation scores. The script calculates the similarity score for each candidate-reference pair and saves the scores in a text file for further analysis. It also prints the average similarity score across all pairs.


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

cand_images = load_image_paths("./eval_data/src_only_dataset.json")

ref_images = load_image_paths_from_txt("./eval_data/images.txt")

all_scores = []
for i in range(len(cand_images)):
    cand = cand_images[i]
    ref = ref_images[i]
    print(f"Calculating similarity for candidate: {cand} and reference: {ref}")
    if ref == 'None' or cand == 'None':
        print(f"Skipping similarity calculation for candidate: {cand} and reference: {ref} due to missing image.")
        all_scores.append(None)  # Assign a similarity score of 0 for missing images
    sim_score = calculate_similarity(cand, ref)
    all_scores.append(sim_score)
    #print(f"Similarity between {cand} and {ref}: {sim_score}")
#saveall similarity scores in a text file
with open("./eval_data/src_only_similarity_scores.txt", "w") as f:
    for score in all_scores:
        f.write(f"{score}\n")
#read the human evaluation scores from the csv file and save them in a list
human_scores = []
with open('./eval_data/eng-eng-img-retrieval_human_eval.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) # skip header
    human_scores = [float(row[6]) for row in reader]

human_scores  = human_scores[:len(all_scores)]  # Ensure human_scores has the same length as all_scores]

#Calculate the correlation between the similarity scores and human evaluation scores
from scipy.stats import pearsonr
# Filter out None values from all_scores and corresponding human_scores
filtered_scores = [(s, h) for s, h in zip(all_scores, human_scores) if s is not None]
if filtered_scores:
    filtered_all_scores, filtered_human_scores = zip(*filtered_scores)
    correlation, p_value = pearsonr(filtered_all_scores, filtered_human_scores)
    print(f"Pearson correlation between similarity scores and human evaluation scores: {correlation}, p-value: {p_value}")
else:
    print("No valid similarity scores to calculate correlation.")
#print("Average Similarity Score:", np.mean(all_scores))

#save similarity scores in a text file
#with open("./eval_data/similarity_tgt_only_de_scores.txt", "w") as f:
    #write the average similarity score at the top of the file
#    f.write(f"Average Similarity Score: {np.mean(all_scores)}\n")
#    for score in all_scores:
#        f.write(f"{score}\n")