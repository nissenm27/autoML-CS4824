import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =======================
# LOAD MODEL SUMMARY
# =======================
summary_path = Path("../results/model_zoo_summary.csv")
df = pd.read_csv(summary_path)

# Extract best standalone model (not an ensemble)
standalone = df[~df["model"].isin(["soft_voting_ensemble", "stacking_ensemble"])]

best_standalone = standalone.loc[standalone["test_score"].idxmax()]
soft = df[df["model"] == "soft_voting_ensemble"].iloc[0]
stack = df[df["model"] == "stacking_ensemble"].iloc[0]

# Optional NAS
nas_row = df[df["model"] == "tiny_nas"]
nas = nas_row.iloc[0] if len(nas_row) else None

# =======================
# LOAD BAYES HISTORY (Warm-Start Bayesian Search)
# =======================
def load_history(model_name):
    base = Path("../results")
    path = base / f"{model_name}_bayes_history.json"
    try:
        with open(path, "r") as f:
            hist = json.load(f)
    except:
        hist = []
    return hist

target_model = "gradient_boosting"
hist = load_history(target_model)

scores = [h["value"] for h in hist]
if len(scores) == 0:
    scores = [0]

# =======================
# PLOT — COMPOSITE FIGURE
# =======================
plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# =======================
# LEFT: Ensemble Accuracy Boost
# =======================
ax = axes[0]

labels = [
    f"Best Model\n({best_standalone['model']})",
    "Soft Voting\nEnsemble",
    "Stacking\nEnsemble"
]

scores_bar = [
    best_standalone["test_score"],
    soft["test_score"],
    stack["test_score"]
]

if nas is not None:
    labels.append("TinyNAS")
    scores_bar.append(nas["test_score"])

bars = ax.bar(labels, scores_bar,
              color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(scores_bar)])

for i in [1, 2]:  # highlight ensembles
    bars[i].set_edgecolor("gold")
    bars[i].set_linewidth(3)

ax.set_title("Ensemble Accuracy Boost", fontsize=16, weight="bold")
ax.set_ylabel("Accuracy", fontsize=14)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.002,
            f"{height:.3f}", ha="center", fontsize=12)

# =======================
# RIGHT: Bayesian Convergence Plot (Warm-Start Enabled)
# =======================
ax2 = axes[1]

ax2.plot(scores, linewidth=3, color="#4C72B0")
ax2.set_title(f"Bayesian Search Convergence\n(Warm-Start Enabled: {target_model.replace('_',' ').title()})",
              fontsize=16, weight="bold")
ax2.set_xlabel("Trial Number")
ax2.set_ylabel("Cross-Validated Score")

# Annotation for early lift
if len(scores) > 2:
    ax2.annotate("Early improvement",
                 xy=(2, scores[2]),
                 xytext=(4, scores[2] + 0.02),
                 arrowprops=dict(arrowstyle="->", color="black"))

# =======================
# SAVE FIGURE
# =======================
out_path = Path("analysis/slide4_ensemble_and_warmstart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()

print(f"\n[Saved] {out_path}\n")
