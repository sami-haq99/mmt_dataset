import csv
import os
import torch
import numpy as np


#saveall similarity scores in a text file


    #read the similarity scores from the text file and save them in a list
all_scores = []
with open("./eval_data/src_only_similarity_scores.txt", "r") as score_file:
    for line in score_file:
        score = line.strip()
        all_scores.append(score)
        
 
#read the human evaluation scores from the csv file and save them in a list
human_scores = []
with open('./eval_data/eng-eng-img-retrieval_human_eval.csv', 'r') as file:
    reader = csv.reader(file)

    human_scores = [float(row[6]) for row in reader]

#check if the length of all_scores and human_scores are the same, if not, print a warning message
if len(all_scores) != len(human_scores):
    print("Warning: The length of similarity scores and human evaluation scores are not the same. Please check the data.")
else:
    print("The length of similarity scores and human evaluation scores are the same. Proceeding with correlation calculation.")
    

#Calculate the correlation between the similarity scores and human evaluation scores
from scipy.stats import pearsonr
# Filter out None values from all_scores and corresponding human_scores
filtered_scores = [(float(s), h) for s, h in zip(all_scores, human_scores) if s != 'None']
if filtered_scores:
    filtered_all_scores, filtered_human_scores = zip(*filtered_scores)
    correlation, p_value = pearsonr(filtered_all_scores, filtered_human_scores)
    print(f"Pearson correlation between similarity scores and human evaluation scores: {correlation}, p-value: {p_value}")
else:
    print("No valid similarity scores to calculate correlation.")
    
    #Pearson correlation between similarity scores and human evaluation scores: -0.035609143228357176, p-value: 0.6266470978019204
#print("Average Similarity Score:", np.mean(all_scores))
#perfrom any other correlation calculation if needed, such as spearman correlation
from scipy.stats import spearmanr

    
correlation, p_value = spearmanr(filtered_all_scores, filtered_human_scores)
print(f"Spearman correlation between similarity scores and human evaluation scores: {correlation}, p-value: {p_value}")
