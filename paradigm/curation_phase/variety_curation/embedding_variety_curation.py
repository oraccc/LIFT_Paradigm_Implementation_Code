from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import json


def get_pca_95_percent_n_compoents(data, embedding_size):
    pca = PCA(n_components=embedding_size)
    pca.fit(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = explained_variance_ratio.cumsum()
    n_components_95_percent = np.argmax(cumulative_variance_ratio >= 0.95)
    return n_components_95_percent


def get_data_variance(data, n_components):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    row_variances = np.var(reduced_data, axis=1)
    return row_variances


if __name__ == "__main__":
    data = np.load("embedding_data_ada_002.npy")["embedding"]

    # Variety Curation Method: Normalized PCA + Variance
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    print(get_pca_95_percent_n_compoents(data=normalized_data, embedding_size=1536)) # 60k output: 555;

    # adjust input parameters based on output from `get_pca_95_percent_n_compoents`
    row_variances = get_data_variance(normalized_data, 555)
    threshold = np.percentile(row_variances, 80)
    with open("path/to/expanded/dataset.json") as file:
        original_dataset = json.load(file)

    processed_dataset = []
    above_threshold_count = 0
    for idx, item in enumerate(row_variances):
        if item > threshold:
            above_threshold_count += 1
            processed_dataset.append(
                {
                    "instruction": original_dataset[idx]["instruction"],
                    "input": original_dataset[idx]["input"],
                    "output": original_dataset[idx]["output"],
                }
            )
            
    print(f"total {above_threshold_count} items selected")
    with open("path/to/variety/curated/dataset.json", "w") as f:
        json.dump(processed_dataset, f, indent=4)

