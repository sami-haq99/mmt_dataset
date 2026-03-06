import csv
import json
from retrieve import retrieve

#This script reads the filtered JSON file created from the WMT25 human evaluation CSV, extracts the source text and reference translations,
# retrieves relevant images using a retrieval function,
# stores the retrieved images and their relevance scores along with the original data in a new JSON file for further use in multimodal evaluation tasks.

def create_dataset(src, tgt= None):
    
    images, distance = retrieve(src, tgt,top_k=1)

    if distance[0][0] > 0.5:  # Threshold for relevance, adjust as needed
        return images[0], distance[0][0]
    else:
        return None, None

if __name__ == "__main__":
    # Example text queries
    #reading csv file ans extracting the 'src' column as text queries and save it in a list (csv headers: lp,src,mt,ref,score,system,annotators,domain,year)
    
    JSON_FILE = "wmt_sqm_hf.json"
    src_queries = []
    tgt_queries = []
    mm_outputs = {}
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        for lp in data: 
            if mm_outputs.get(lp) is None:
                mm_outputs[lp] = []
            for entry in data[lp]: #run only for few entries for testing
                src = entry["src"]
                ref = entry["ref"]
                image, distance = create_dataset(src, ref) 
                if image is not None:
                    #aadd the image and distance to the entry and save it in a json file, entry is read from json file and contains: domain, year, src, ref, mt_outputs (list of dict with system, mt, score, annotators)
                    entry["image"] = image
                    entry["distance"] = distance
                    mm_outputs[lp].append(entry)
        with open("output-mm-dict.json", "w", encoding="utf-8") as f:
            json.dump(mm_outputs, f, ensure_ascii=False, indent=2, default=str) 
