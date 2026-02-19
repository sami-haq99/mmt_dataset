import json
from retrieve import retrieve

def create_dataset(text_queries, output_file="retrieval_dataset.json"):
    dataset = []
    for text in text_queries:
        images = retrieve(text, top_k=5)
        dataset.append({"text": text, "images": images})

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print("Dataset created:", output_file)

if __name__ == "__main__":
    # Example text queries
    queries = [
        "a dog playing in the park",
        "modern office building",
        "mountain landscape with snow"
    ]
    create_dataset(queries)
