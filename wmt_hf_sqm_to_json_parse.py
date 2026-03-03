import pandas as pd
import json

# load CSV
input_csv = "/home/sami/mmt-eval/eval-datasets/WMT24 human eval/HF WMT/wmt-sqm-human-evaluation-train.csv"
df = pd.read_csv(input_csv)
#extact only following language pairs: en-de, en-ja, en-zh, en-cs, en-ru
language_pairs = ['en-de', 'en-ja', 'en-zh', 'en-cs', 'en-ru']
domain = 'ecommerce'
df = df[df['lp'].isin(language_pairs) & (df['domain'] == domain)]
# columns defining ONE unique source instance
group_cols = ["lp", "src", "ref", "domain", "year"]

output = {}

# group rows
for keys, group in df.groupby(group_cols):

    lp, src, ref, domain, year = keys

    entry = {
        "domain": domain,
        "year": int(year),
        "src": src,
        "ref": ref,
        "mt_outputs": []
    }

    # collect MT outputs
    for _, row in group.iterrows():
        entry["mt_outputs"].append({
            "system": row["system"],
            "mt": row["mt"],
            "score": float(row["score"]),
            "annotators": int(row["annotators"])
        })
    if output.get(f"{lp}") is None:
        output[f"{lp}"] = []
    output[f"{lp}"].append(entry)

# save JSON
with open("output-dict.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("JSON file created!")