import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def find_similar_datasets(current_meta, meta_dataset, k=1):
    names = []
    features = []

    for name, entry in meta_dataset.items():
        names.append(name)
        features.append(list(entry["meta_features"].values()))

    if not features:
        return []  # no warm start available yet

    features = np.array(features)
    current_vec = np.array(list(current_meta.values())).reshape(1, -1)

    distances = euclidean_distances(current_vec, features)[0]
    nearest_indices = np.argsort(distances)[:k]

    return [names[i] for i in nearest_indices]
