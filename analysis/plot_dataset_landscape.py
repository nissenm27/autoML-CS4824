import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

meta_df = pd.read_csv("analysis/meta_table.csv")

# Select numeric meta-features
feature_cols = [c for c in meta_df.columns if c not in ["dataset", "difficulty"]]
X = meta_df[feature_cols]

# Scale features
X_scaled = StandardScaler().fit_transform(X)

# PCA 2D
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

meta_df["PC1"] = coords[:,0]
meta_df["PC2"] = coords[:,1]

# Beautiful plot
plt.figure(figsize=(10,7))
scatter = plt.scatter(
    meta_df["PC1"],
    meta_df["PC2"],
    c=meta_df["difficulty"],
    cmap="viridis",
    s=300,
    alpha=0.85,
    edgecolors="white",
    linewidth=2
)

for i, row in meta_df.iterrows():
    plt.text(row.PC1+0.02, row.PC2+0.02, row.dataset, fontsize=14)

plt.colorbar(scatter, label="Dataset Difficulty")
plt.title("Landscape of Datasets via Meta-Features (Slide 1)", fontsize=18)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("dataset_landscape.png", dpi=300)
plt.show()
