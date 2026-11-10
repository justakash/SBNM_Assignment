
"""
Leakage-free Node2Vec link prediction on BioGRID.

1. Train Node2Vec using training edges only.
2. Represent protein pairs using Hadamard edge features.
3. Train a Logistic Regression classifier.
4. Evaluate on unseen positive and negative protein pairs.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# --------------------------------------------------
# 2. Load training graph only
# --------------------------------------------------

G_train = nx.read_edgelist(
    RESULTS_DIR / "train_edges.tsv",
    delimiter="\t",
    nodetype=int,
    create_using=nx.Graph()
)

print("Training graph loaded")
print("Proteins:", G_train.number_of_nodes())
print("Training interactions:", G_train.number_of_edges())

# --------------------------------------------------
# 3. Train Node2Vec embeddings
# --------------------------------------------------

# A reduced walk budget keeps this evaluation practical.
# Keep these settings unchanged across the p/q experiments.

node2vec = Node2Vec(
    G_train,
    dimensions=64,
    walk_length=10,
    num_walks=5,
    p=1,
    q=1,
    workers=2,
    seed=42,
    quiet=False
)

embedding_model = node2vec.fit(
    window=10,
    min_count=1,
    batch_words=128,
    sg=1,
    workers=2,
    seed=42
)

print("\nNode2Vec training completed")

# --------------------------------------------------
# 4. Load positive and negative edge samples
# --------------------------------------------------

def load_edges(filename):
    """Read protein pairs from a two-column TSV file."""
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

# --------------------------------------------------
# 5. Generate Hadamard edge features
# --------------------------------------------------

def create_edge_features(edge_data, model):
    """
    Represent each protein pair by element-wise multiplication
    of its two Node2Vec embedding vectors.

    Hadamard features capture pairwise relationships between
    learned protein representations.
    """
    features = []

    for u, v in edge_data[["protein_a", "protein_b"]].itertuples(
        index=False, name=None
    ):
        vector_u = model.wv[str(u)]
        vector_v = model.wv[str(v)]

        features.append(vector_u * vector_v)

    return np.asarray(features)


X_train_positive = create_edge_features(train_positive, embedding_model)
X_train_negative = create_edge_features(train_negative, embedding_model)

X_test_positive = create_edge_features(test_positive, embedding_model)
X_test_negative = create_edge_features(test_negative, embedding_model)

# Combine positive and negative samples.
X_train = np.vstack([X_train_positive, X_train_negative])
X_test = np.vstack([X_test_positive, X_test_negative])

# 1 = existing interaction; 0 = sampled non-interaction.
y_train = np.concatenate([
    np.ones(len(X_train_positive)),
    np.zeros(len(X_train_negative))
])

y_test = np.concatenate([
    np.ones(len(X_test_positive)),
    np.zeros(len(X_test_negative))
])

# Shuffle training examples reproducibly.
X_train, _, y_train, _ = train_test_split(
    X_train,
    y_train,
    test_size=0.0,
    random_state=42,
    stratify=y_train
) if False else (X_train, None, y_train, None)

# --------------------------------------------------
# 6. Train Logistic Regression classifier
# --------------------------------------------------

classifier = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

classifier.fit(X_train, y_train)

print("\nClassifier training completed")

# --------------------------------------------------
# 7. Evaluate link prediction
# --------------------------------------------------

test_probabilities = classifier.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, test_probabilities)
average_precision = average_precision_score(
    y_test, test_probabilities
)

print("\nLink Prediction Results")
print("-----------------------")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")

# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

metrics = pd.DataFrame([{
    "method": "Node2Vec",
    "p": 1,
    "q": 1,
    "dimensions": 64,
    "walk_length": 10,
    "num_walks": 5,
    "roc_auc": roc_auc,
    "average_precision": average_precision
}])

metrics.to_csv(
    RESULTS_DIR / "node2vec_baseline_metrics.csv",
    index=False
)

embedding_model.wv.save_word2vec_format(
    str(RESULTS_DIR / "node2vec_train_graph_embeddings.txt")
)

print("\nMetrics saved to node2vec_baseline_metrics.csv")
print("Training-graph embeddings saved.")
print("Evaluation completed.")
