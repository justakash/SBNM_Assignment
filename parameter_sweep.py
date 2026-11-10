
"""
Node2Vec p/q parameter sensitivity experiment on BioGRID.

Uses the same training graph, positive/negative samples,
embedding dimensions, walk settings, and classifier across
all configurations.

The existing p=1, q=1 evaluation is reused as baseline.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --------------------------------------------------
# 1. Paths and experiment settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

DIMENSIONS = 64
WALK_LENGTH = 10
NUM_WALKS = 5
SEED = 42

# Four additional configurations.
# The p=1, q=1 baseline is already available.
PARAMETER_CONFIGS = [
    (0.25, 0.25),
    (0.25, 4),
    (4, 0.25),
    (4, 4)
]

# --------------------------------------------------
# 2. Load training graph and evaluation samples
# --------------------------------------------------

G_train = nx.read_edgelist(
    RESULTS_DIR / "train_edges.tsv",
    delimiter="\t",
    nodetype=int,
    create_using=nx.Graph()
)

def load_edges(filename):
    """Load protein pairs from a TSV file."""
    return pd.read_csv(
        RESULTS_DIR / filename,
        sep="\t",
        header=None,
        names=["protein_a", "protein_b"]
    )


train_positive = load_edges("train_positive.tsv")
train_negative = load_edges("train_negative.tsv")
test_positive = load_edges("test_positive.tsv")
test_negative = load_edges("test_negative.tsv")

print("Training graph:", G_train.number_of_nodes(), "proteins,",
      G_train.number_of_edges(), "interactions")

# --------------------------------------------------
# 3. Prepare labels
# --------------------------------------------------

y_train = np.concatenate([
    np.ones(len(train_positive)),
    np.zeros(len(train_negative))
])

y_test = np.concatenate([
    np.ones(len(test_positive)),
    np.zeros(len(test_negative))
])


# --------------------------------------------------
# 4. Define Hadamard edge feature extraction
# --------------------------------------------------

def create_edge_features(edge_data, model):
    """
    Multiply the embeddings of two proteins element-wise.
    This creates one feature vector per protein pair.
    """
    features = []

    for u, v in edge_data.itertuples(index=False, name=None):
        vector_u = model.wv[str(u)]
        vector_v = model.wv[str(v)]
        features.append(vector_u * vector_v)

    return np.asarray(features)


# --------------------------------------------------
# 5. Load existing baseline result
# --------------------------------------------------

baseline_file = RESULTS_DIR / "node2vec_baseline_metrics.csv"

if not baseline_file.exists():
    raise FileNotFoundError(
        "Baseline metrics file not found. "
        "Run evaluate_node2vec.py first."
    )

baseline = pd.read_csv(baseline_file)

baseline = baseline[
    (baseline["p"] == 1) & (baseline["q"] == 1)
]

if baseline.empty:
    raise ValueError("The p=1, q=1 baseline was not found.")

results = baseline.to_dict("records")

# Save baseline before starting additional experiments.
output_file = RESULTS_DIR / "node2vec_parameter_sweep.csv"

pd.DataFrame(results).to_csv(output_file, index=False)

# --------------------------------------------------
# 6. Run parameter configurations
# --------------------------------------------------

for experiment_number, (p_value, q_value) in enumerate(
    PARAMETER_CONFIGS, start=1
):

    print("\n" + "=" * 55)
    print(f"Experiment {experiment_number}/4")
    print(f"Testing p={p_value}, q={q_value}")
    print("=" * 55)

    # Generate biased random walks and train embeddings.
    node2vec = Node2Vec(
        G_train,
        dimensions=DIMENSIONS,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        p=p_value,
        q=q_value,
        workers=2,
        seed=SEED,
        quiet=False
    )

    model = node2vec.fit(
        window=10,
        min_count=1,
        batch_words=128,
        sg=1,
        workers=2,
        seed=SEED
    )

    print("Embedding training completed.")

    # Extract edge features.
    X_train_positive = create_edge_features(
        train_positive, model
    )
    X_train_negative = create_edge_features(
        train_negative, model
    )
    X_test_positive = create_edge_features(
        test_positive, model
    )
    X_test_negative = create_edge_features(
        test_negative, model
    )

    X_train = np.vstack([
        X_train_positive,
        X_train_negative
    ])

    X_test = np.vstack([
        X_test_positive,
        X_test_negative
    ])

    # Train the same classifier for every configuration.
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=SEED)
    )

    classifier.fit(X_train, y_train)

    # Evaluate on held-out test pairs.
    probabilities = classifier.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)
    average_precision = average_precision_score(
        y_test, probabilities
    )

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")

    # Store results immediately so completed experiments
    # are preserved even if a later experiment is interrupted.
    results.append({
        "method": "Node2Vec",
        "p": p_value,
        "q": q_value,
        "dimensions": DIMENSIONS,
        "walk_length": WALK_LENGTH,
        "num_walks": NUM_WALKS,
        "roc_auc": roc_auc,
        "average_precision": average_precision
    })

    pd.DataFrame(results).to_csv(output_file, index=False)

    print("Experiment result saved.")

    # Release memory before the next configuration.
    del node2vec, model, classifier
    del X_train, X_test


# --------------------------------------------------
# 7. Display final comparison
# --------------------------------------------------

comparison = pd.DataFrame(results).sort_values(
    by="roc_auc",
    ascending=False
)

print("\nFINAL PARAMETER COMPARISON")
print(comparison.to_string(index=False))

print("\nResults saved to:", output_file)
print("Parameter sweep completed.")
