

#open this file, read the third column and calcuate the average, mode and mdian of the values in that column
import csv
from statistics import mean, mode, median
with open('./eval_data/eng-eng-img-retrieval_human_eval.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) # skip header
    values = [float(row[6]) for row in reader]
    print(values)
    avg = mean(values)
    mod = mode(values)
    med = median(values)
    print(f'Average: {avg}, Mode: {mod}, Median: {med}')
    
#plot the distribution of the values in the 6 column using matplotlib
import matplotlib.pyplot as plt
plt.hist(values, bins=10)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Values in Column 6')
plt.savefig('./eval_data/eng-eng-img-retrieval_human_eval_distribution.png')
