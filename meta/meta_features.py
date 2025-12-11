import numpy as np
import pandas as pd
from scipy.stats import entropy

def compute_meta_features(X, y):
    meta = {}

    meta["n_samples"] = X.shape[0]
    meta["n_features"] = X.shape[1]

    # numeric / categorical count
    meta["n_numeric"] = X.select_dtypes(include=np.number).shape[1]
    meta["n_categorical"] = X.select_dtypes(exclude=np.number).shape[1]

    # class info
    class_counts = y.value_counts()
    meta["n_classes"] = len(class_counts)
    meta["class_imbalance"] = class_counts.max() / class_counts.min()
    meta["class_entropy"] = entropy(class_counts)

    # missing value ratio
    meta["missing_ratio"] = X.isna().mean().mean()

    # simple correlation difficulty (numeric only)
    if meta["n_numeric"] > 1:
        meta["mean_corr"] = X.select_dtypes(include=np.number).corr().abs().mean().mean()
    else:
        meta["mean_corr"] = 0

    return meta
