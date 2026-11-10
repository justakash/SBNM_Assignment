
"""
Generate final Node2Vec parameter comparison plots.

Reads the completed parameter sweep results and creates:
1. ROC-AUC comparison plot
2. Average Precision comparison plot

No model training is performed.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = RESULTS_DIR / "node2vec_parameter_sweep.csv"

# --------------------------------------------------
# 2. Load experimental results
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Results file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# Create readable labels for each parameter configuration.
df["configuration"] = df.apply(
    lambda row: f"p={row['p']:g}, q={row['q']:g}",
    axis=1
)

# Sort configurations by ROC-AUC.
df = df.sort_values("roc_auc", ascending=False)

print("\nParameter comparison:")
print(df[["configuration", "roc_auc", "average_precision"]].to_string(index=False))

# --------------------------------------------------
# 3. Plot ROC-AUC comparison
# --------------------------------------------------

plt.figure(figsize=(9, 5))

bars = plt.bar(
    df["configuration"],
    df["roc_auc"]
)

plt.title("Node2Vec Parameter Comparison - ROC-AUC")
plt.xlabel("Parameter Configuration")
plt.ylabel("ROC-AUC")
plt.ylim(0.80, 0.90)
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.001,
        f"{height:.4f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "roc_auc_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 4. Plot Average Precision comparison
# --------------------------------------------------

plt.figure(figsize=(9, 5))

bars = plt.bar(
    df["configuration"],
    df["average_precision"]
)

plt.title("Node2Vec Parameter Comparison - Average Precision")
plt.xlabel("Parameter Configuration")
plt.ylabel("Average Precision")
plt.ylim(0.80, 0.92)
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.001,
        f"{height:.4f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "average_precision_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 5. Confirm output
# --------------------------------------------------

print("\nPlots generated successfully:")
print(RESULTS_DIR / "roc_auc_comparison.png")
print(RESULTS_DIR / "average_precision_comparison.png")
