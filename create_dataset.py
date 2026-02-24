import csv
import json
from retrieve import retrieve

def create_dataset(text_queries, output_file="retrieval_dataset_de.json"):
    dataset = []
    for text in text_queries:
        images = retrieve(text, top_k=5)
        dataset.append({"text": text, "images": images})

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print("Dataset created:", output_file)

if __name__ == "__main__":
    # Example text queries
    #reading csv file ans extracting the 'src' column as text queries and save it in a list (csv headers: lp,src,mt,ref,score,system,annotators,domain,year)
    
    CSV_FILE = "en-de-ecommerce.csv"
    queries = []
    with open(CSV_FILE, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            queries.append(row["ref"])

    create_dataset(queries[:10])  # Create dataset for the first 100 queries
