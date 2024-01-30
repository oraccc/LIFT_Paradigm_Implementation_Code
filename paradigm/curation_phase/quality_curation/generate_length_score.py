import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
    with open("path/to/variety/curated/dataset") as file:
        data = json.load(file)

    length_list = []
    for item in tqdm(data):
        text = (
            item["instruction"] + item["input"] + item["output"]
            if item["input"] != "" 
            else item["instruction"] + item["output"]
            )
        length_list.append(len(text.replace("\t", "").replace("\n", "")))

    # plt.hist(length_list, bins=100, edgecolor='black')
    # plt.title('Length Distribution')
    # plt.xlabel('Length')
    # plt.ylabel('Frequency')
    # plt.show()
        
    # # determine `length_threshold` through length distribution graph
    # length_threshold = 7500
    # total_samples = len(length_list)

    # count_below_threshold = sum(1 for length in length_list if length <= length_threshold)
    # percentage_below_threshold = (count_below_threshold / total_samples) * 100
    # print(f"Percentage of samples shorter or equal to {length_threshold}: {percentage_below_threshold:.2f}%")
        
    threshold = np.percentile(length_list, 98)
    scores = []

    for length in length_list:
        if length >= threshold:
            score = 100
        else:
            score = (length / threshold) * 100
        scores.append(score)

    # plt.hist(scores, bins=50, edgecolor='black')
    # plt.title('Length Distribution')
    # plt.xlabel('Score')
    # plt.ylabel('Frequency')
    # plt.show()
    
    with open("length_scores.txt", "w") as file:
        for idx, item in enumerate(scores):
            file.write(f"{idx}\t{item}\n")