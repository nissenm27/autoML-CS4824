import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import ast

# -------------------------------------------
# Helper: estimate model size (number params)
# -------------------------------------------
def estimate_model_size(model_name, best_params):
    """Rough model size metric for bubble scaling."""
    
    if model_name == "logistic_regression":
        return 10_000

    if model_name == "ridge":
        return 10_000

    if model_name == "decision_tree":
        return best_params.get("model__max_depth", 5) * 500

    if model_name == "random_forest":
        n = best_params.get("model__n_estimators", 100)
        depth = best_params.get("model__max_depth", 10)
        return n * depth * 300

    if model_name == "gradient_boosting":
        n = best_params.get("model__n_estimators", 100)
        depth = best_params.get("model__max_depth", 3)
        return n * depth * 400

    if model_name == "neural_net":
        layers = best_params.get("model__hidden_layer_sizes", (32,))
        return sum(layers) * 300

    if model_name == "tiny_nas":
        layers = best_params.get("hidden_layer_sizes", (32,))
        return sum(layers) * 500

    if "ensemble" in model_name:
        return 50_000

    return 20_000


# -------------------------------------------
# Load model results
# -------------------------------------------
df = pd.read_csv("../results/model_zoo_summary.csv")

# Convert best_params string → dict safely
def parse_params(x):
    if isinstance(x, str) and x.strip() != "":
        try:
            return ast.literal_eval(x)
        except:
            return {}
    return {}

df["best_params"] = df["best_params"].apply(parse_params)

# Handle None runtimes
df["runtime_sec"] = df["runtime_sec"].fillna(9999)

# Add model size
df["model_size"] = df.apply(
    lambda row: estimate_model_size(row["model"], row["best_params"]),
    axis=1
)

# -------------------------------------------
# Compute Pareto frontier (accuracy↑ runtime↓)
# -------------------------------------------
df_sorted = df.sort_values("runtime_sec")
pareto = []
best_acc = -1

for _, row in df_sorted.iterrows():
    if row["test_score"] > best_acc:
        pareto.append(row)
        best_acc = row["test_score"]

pareto = pd.DataFrame(pareto)

# -------------------------------------------
# Bubble Plot
# -------------------------------------------
plt.figure(figsize=(12, 8))

colors = {
    "logistic_regression": "blue",
    "ridge": "blue",
    "decision_tree": "green",
    "random_forest": "green",
    "gradient_boosting": "green",
    "neural_net": "red",
    "tiny_nas": "orange",
    "soft_voting_ensemble": "purple",
    "stacking_ensemble": "purple"
}

for _, row in df.iterrows():
    plt.scatter(
        row["runtime_sec"],
        row["test_score"],
        s=row["model_size"] / 40,
        alpha=0.6,
        color=colors.get(row["model"], "gray"),
        edgecolor="black",
        linewidth=0.8
    )
    plt.text(
        row["runtime_sec"] * 1.02,
        row["test_score"],
        row["model"],
        fontsize=9
    )

# Draw Pareto frontier
plt.plot(
    pareto["runtime_sec"],
    pareto["test_score"],
    linestyle="--",
    linewidth=2,
    color="black",
    label="Pareto Frontier"
)

plt.xscale("log")
plt.xlabel("Runtime (seconds, log scale)", fontsize=14)
plt.ylabel("Test Accuracy", fontsize=14)
plt.title("Model Zoo: Accuracy vs Runtime vs Model Size", fontsize=18)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("../results/model_zoo_bubble_plot.png", dpi=300)
plt.show()
