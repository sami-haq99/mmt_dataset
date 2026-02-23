import json

# Configuration

# {'en-cs', 'en-et', 'en-uk', 'unknown', 'en-bho', 'en-sr', 'en-it', 'en-ja', 'cs-de', 'en-mas', 'cs-uk', 'en-ru', 'en-is', 'en-zh', 'en-ar'}

INPUT_FILE = "/home/sami/mmt-eval/doc-mte/mmss/wmt25-genmt-humeval.jsonl"
OUTPUT_FILE = "filtered_results.txt"
# Target language pairs to keep
TARGET_LPS = ["en-cs", "en-it", "en-zh"]

def extract_system_data(file_path):
    extracted_records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            # 1. Identify Language Pair (Checking annotator IDs)
            # We assume a document belongs to one pair based on its first available score
            first_sys = next(iter(data["scores"].values()))
            current_lp = first_sys[0]["annotator"].split("_#_")[0]

            # 2. Filter for specific language pairs
            if current_lp not in TARGET_LPS:
                continue

            record = {
                "doc_id": data.get("doc_id"),
                "lp": current_lp,
                "src": data.get("src_text"),
                "systems": []
            }

            # 3. Get systems, their scores, and their specific translations
            # We iterate through tgt_text as it usually contains the full list of MT systems
            for sys_name, translation in data["tgt_text"].items():
                # Extract scores (if annotated)
                sys_scores = data["scores"].get(sys_name, [])
                #calculate average score if available
                score_values = [s["score"] for s in sys_scores]
                avg_score = sum(score_values) / len(score_values) if score_values else 0
                record["systems"].append({
                    "name": sys_name,
                    "avg_score": avg_score,
                    "output": translation
                })
            
            extracted_records.append(record)
    
    return extracted_records

# --- Execution ---
results = extract_system_data(INPUT_FILE)

# Displaying the first record as a sample
if results:
    r = results[0]
    print(f"Record: {r['doc_id']} ({r['lp']})")
    print(f"Source: {r['src']}")
    for sys in r['systems'][:3]: # Showing first 3 systems
        score_str = f"Average Score: {sys['avg_score']}" if sys['avg_score'] else "Not annotated"
        print(f"  [{sys['name']}] {score_str}")
        print(f"  Output: {sys['output']}\n")

#write a csv file with columns: doc_id, lp,system_name, avg_score, src, output
#write the csv with '' as the text contains commas and newlines
import csv
CSV_FILE = "filtered_results.csv"
with open(CSV_FILE, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(["doc_id", "lp", "system_name", "avg_score", "src", "output"])
    for r in results:
        for sys in r['systems']:
            writer.writerow([r['doc_id'], r['lp'], sys['name'], sys['avg_score'], r['src'], sys['output']])