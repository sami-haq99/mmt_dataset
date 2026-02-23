

#Load the dataset from .csv file
#Here is the examples:
'''
lp,src,mt,ref,score,system,annotators,domain,year
en-de,You can come back any time as our chat service window is open 24/7,"Sie können jederzeit wiederkommen, da unser Chat-Service-Fenster geöffnet ist 24/7","Sie können jederzeit wiederkommen, da unser Chat-Service-Fenster täglich rund um die Uhr geöffnet ist",100.0,PROMT,1,conversation,2022 
'''

#Print all the different language pairs in the dataset
#Print all domains in the dataset
#read the .csv file and print the number of rows in the dataset
import csv
CSV_FILE = "/home/sami/mmt-eval/eval-datasets/WMT24 human eval/HF WMT/wmt-sqm-human-evaluation-train.csv"
#For langauge en-de, and domain ecommerce and save entire row in a csv file called en-de-ecommerce.csv
OUTPUT_FILE = "en-de-ecommerce.csv"
with open(CSV_FILE, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)
    print(f"Number of rows in the dataset: {len(rows)}")
    
    language_pairs = set()
    domains = set()
    
    for row in rows:
        language_pairs.add(row["lp"])
        domains.add(row["domain"])
    
    print("Language pairs in the dataset:", language_pairs)
    print("Domains in the dataset:", domains)
    
    # Filter for en-de and ecommerce
    filtered_rows = [row for row in rows if row["lp"] == "en-de" and row["domain"] == "ecommerce"]
    
    # Save filtered rows to a new CSV file
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as output_csvfile:
        writer = csv.DictWriter(output_csvfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"Filtered data saved to {OUTPUT_FILE} with {len(filtered_rows)} rows.")
