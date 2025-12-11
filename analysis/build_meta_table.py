import sys
from pathlib import Path

# Add project root to Python path BEFORE imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from meta.meta_features import compute_meta_features

DATASETS = {
    "adult": "../data/adult_income.csv",
    "iris": "../data/iris.csv",
    "wine": "../data/wine_quality.csv",
}

TARGETS = {
    "adult": "class",
    "iris": "class",
    "wine": "quality",
}

rows = []

for name, path in DATASETS.items():
    df = pd.read_csv(path)

    target = TARGETS[name]
    X = df.drop(columns=[target])
    y = df[target]

    meta = compute_meta_features(X, y)
    meta["dataset"] = name

    # Remove nested dicts (they break DataFrame)
    flat_meta = {
        k: v
        for k, v in meta.items()
        if not isinstance(v, dict) and not isinstance(v, list)
    }

    rows.append(flat_meta)

# Build DataFrame
meta_df = pd.DataFrame(rows)

# Difficulty proxy
if "class_imbalance" in meta_df.columns:
    meta_df["difficulty"] = meta_df["class_imbalance"]
else:
    meta_df["difficulty"] = meta_df.get("target_variance", 0)

# Save
Path("analysis").mkdir(exist_ok=True)
meta_df.to_csv("analysis/meta_table.csv", index=False)

print("\n=== Meta Table ===")
print(meta_df)
