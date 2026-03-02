import csv
import json
from retrieve import retrieve
import os

def featch_images(text_queries, tgt_queries= None):
    dataset = []
    for i, text in enumerate(text_queries):
        if tgt_queries is not None:
            images = retrieve(text, tgt_queries[i], top_k=5)
            dataset.append([img['image'] for img in images])
        else:
            images = retrieve(text, top_k=5)
            dataset.append([img['image'] for img in images])
            
    return dataset

def calculate_accuracy(retrieved_images, image_names):
    correct = 0
    num_1st_retrieved = 0
    #check if the each image_names match with the crossponding set of retrieved images
    #perform operation row by row 
    for i in range(len(image_names)):
        for item in retrieved_images[i]:
            if image_names[i] in os.path.basename(item):
                correct += 1
                break
        if retrieved_images[i] and os.path.basename(retrieved_images[i][0]) == image_names[i]:
            num_1st_retrieved += 1
            
    accuracy = correct / len(retrieved_images) if retrieved_images else 0
    num_1st_retrieved_accuracy = num_1st_retrieved / len(retrieved_images) if retrieved_images else 0
    return accuracy, num_1st_retrieved_accuracy

if __name__ == "__main__":
    # Example text queries
    #reading csv file ans extracting the 'src' column as text queries and save it in a list (csv headers: lp,src,mt,ref,score,system,annotators,domain,year)
    
    input_dir = "./eval_data/"
    src_lang = "en"
    tgt_langs = ["de", "fr"]
    image_file = f"{input_dir}images.txt"
    
    src_queries = []
    tgt_queries = {}
    image_names = []
    src_file = f"{input_dir}src.{src_lang}"
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            src_queries.append(line.strip())
    
    for tgt in tgt_langs:
       tgt_file = f"{input_dir}ref.{tgt}"
       tgt_queries[tgt] = []
       with open(tgt_file, "r", encoding="utf-8") as f:
           for line in f:
                tgt_queries[tgt].append(line.strip())

    with open(image_file, "r", encoding="utf-8") as f:
        for line in f:
            image_names.append(line.strip())


    src_only_dataset = featch_images(src_queries[:10])
    joint_dataset = {}
    for tgt in tgt_langs:
        joint_dataset[tgt] = featch_images(src_queries[:10], tgt_queries[tgt][:10])
    tgt_only_dataset = {}
    for tgt in tgt_langs:
        tgt_only_dataset[tgt] = featch_images(tgt_queries[tgt][:10])
    
    #dump the datasets in json files
    with open(f"{input_dir}src_only_dataset.json", "w") as f:
        json.dump(src_only_dataset, f, indent=2)
    for tgt in tgt_langs:
        with open(f"{input_dir}joint_dataset_{tgt}.json", "w") as f:
            json.dump(joint_dataset[tgt], f, indent=2)
        with open(f"{input_dir}tgt_only_dataset_{tgt}.json", "w") as f:
            json.dump(tgt_only_dataset[tgt], f, indent=2)
            
    print("Calculating accuracy for each dataset:")
    print("SRC Only Accuracy:", calculate_accuracy(src_only_dataset, image_names[:10]))
    for tgt in tgt_langs:
        print(f"Joint {tgt} Accuracy:", calculate_accuracy(joint_dataset[tgt], image_names[:10]))
        print(f"TGT Only {tgt} Accuracy:", calculate_accuracy(tgt_only_dataset[tgt], image_names[:10]))