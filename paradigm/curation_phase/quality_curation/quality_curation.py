import json
import numpy as np
import random



if __name__ == "__main__":
    gpt_score_path = "gpt_score.txt"
    length_score_path = "length_score.txt"

    gpt_scores = []
    length_scores = []

    with open(gpt_score_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            score = float(parts[1])
            gpt_scores.append(score)


    with open(length_score_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            score = float(parts[1])
            length_scores.append(score)

    assert(len(gpt_scores) == len(length_scores))

    total_scores = []

    for i in range(len(gpt_scores)):
        total_scores.append(0.9 * gpt_scores[i] + 0.1 * length_scores[i])
    
    sorted_indices = np.argsort(total_scores)[::-1]
    # select top 10k instruction data
    top_indices = sorted_indices[:10000]

    with open("path/to/variety/curated/dataset.json") as file:
        original_dataset = json.load(file)

    processed_dataset = []

    for i in top_indices:
        processed_dataset.append(original_dataset[i])


    print(len(processed_dataset))
    random.shuffle(processed_dataset)

    with open("path/to/save/quality/curated/dataset.json", "w") as f:
        json.dump(processed_dataset, f, indent=4)