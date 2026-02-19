import json

with open("wmt25-genmt-humeval.jsonl", "r") as f:
    data = [json.loads(x) for x in f.readlines()]
    
# Print the first 5 entries
for entry in data[:5]:
    print(json.dumps(entry, indent=2))    