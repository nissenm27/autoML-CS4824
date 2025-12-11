import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("/Users/mattn/Documents/VT_Undergrad/Fall_25:26/CS_4824/final_project/autoML-CS4824/results")


sns.set(style="whitegrid", context="talk")

MODEL = "random_forest"    # choose one model to illustrate

def load_history(model, search_type):
    file = RESULTS_DIR / f"{model}_{search_type}_history.json"
    if not file.exists():
        print(f"Missing {file}, skipping.")
        return None
    
    with open(file, "r") as f:
        hist = json.load(f)
        
    df = pd.DataFrame(hist)
    df["search_type"] = search_type
    df["best_so_far"] = df["score"].cummax()
    return df

# Load all search types
dfs = []
for s in ["grid", "random", "bayes"]:
    df = load_history(MODEL, s)
    if df is not None:
        dfs.append(df)

df = pd.concat(dfs)

# Plot
plt.figure(figsize=(14, 7))

palette = {
    "grid": "#CC0000",
    "random": "#0066CC",
    "bayes": "#228B22"
}

# plot raw scores
for s_type in df["search_type"].unique():
    subset = df[df["search_type"] == s_type]
    plt.scatter(subset["iteration"], subset["score"],
                alpha=0.4, s=60, color=palette[s_type], label=f"{s_type} trials")

# best-so-far lines
for s_type in df["search_type"].unique():
    subset = df[df["search_type"] == s_type]
    plt.plot(subset["iteration"], subset["best_so_far"],
             linewidth=3, color=palette[s_type], label=f"{s_type} best-so-far")

plt.title(f"Search Trajectories for {MODEL.replace('_',' ').title()}")
plt.xlabel("Iteration")
plt.ylabel("Cross-Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig("search_trajectory_plot.png", dpi=300)
plt.show()
