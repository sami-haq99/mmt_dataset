import csv
import json
from retrieve import retrieve

def create_dataset(text_queries, tgt_queries= None, output_file="retrieval_dataset_ende_joint.json"):
    dataset = []
    for i, text in enumerate(text_queries):
        if tgt_queries is not None:
            images = retrieve(text, tgt_queries[i], top_k=5)
            dataset.append({"text": text, "images": [img['image'] for img in images], "distances": [img['distance'] for img in images]})
        else:
            images = retrieve(text, top_k=5)
            dataset.append({"text": text, "images": [img['image'] for img in images], "distances": [img['distance'] for img in images]})

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print("Dataset created:", output_file)

if __name__ == "__main__":
    # Example text queries
    #reading csv file ans extracting the 'src' column as text queries and save it in a list (csv headers: lp,src,mt,ref,score,system,annotators,domain,year)
    
    CSV_FILE = "en-de-ecommerce.csv"
    src_queries = []
    tgt_queries = []
    with open(CSV_FILE, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            src_queries.append(row["src"])
            tgt_queries.append(row["ref"])

    create_dataset(src_queries[:10], tgt_queries[:10])  # Create dataset for the first 100 queries
