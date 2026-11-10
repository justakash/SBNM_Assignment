
"""
Prepare leakage-free train/test edge sets for Node2Vec link prediction.

- Hold out 20% of edges for testing.
- Preserve at least one training edge for every protein.
- Sample non-interacting protein pairs as negative examples.
- Exclude all known BioGRID interactions from negative sampling.
"""

from pathlib import Path
import random
import networkx as nx

# --------------------------------------------------
# 1. Paths and reproducibility
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ppi_edges_clean.tsv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(42)

# --------------------------------------------------
# 2. Load the complete BioGRID network
# --------------------------------------------------

G = nx.read_edgelist(
    DATA_FILE,
    delimiter="\t",
    nodetype=int,
    create_using=nx.Graph()
)

print("Original network:", G.number_of_nodes(), "proteins,",
      G.number_of_edges(), "interactions")

# --------------------------------------------------
# 3. Hold out 20% of edges
# --------------------------------------------------

edges = list(G.edges())
random.shuffle(edges)

target_test_size = int(0.20 * len(edges))
train_graph = G.copy()
test_edges = []

# Remove an edge only if both endpoints retain at least
# one other training interaction.
for u, v in edges:
    if len(test_edges) >= target_test_size:
        break

    if train_graph.degree(u) > 1 and train_graph.degree(v) > 1:
        train_graph.remove_edge(u, v)
        test_edges.append((u, v))

train_edges = list(train_graph.edges())

print("\nTraining interactions:", len(train_edges))
print("Testing interactions:", len(test_edges))
print("Training proteins:", train_graph.number_of_nodes())

# --------------------------------------------------
# 4. Sample negative edges
# --------------------------------------------------

# Negative edges are pairs that do not occur anywhere
# in the original BioGRID network.
# This prevents held-out positive edges becoming negatives.

def sample_negative_edges(graph, number_to_sample, seed=42):
    rng = random.Random(seed)
    negatives = set()
    nodes = list(graph.nodes())

    while len(negatives) < number_to_sample:
        u, v = rng.sample(nodes, 2)
        edge = (min(u, v), max(u, v))

        if not G.has_edge(*edge) and edge not in negatives:
            negatives.add(edge)

    return list(negatives)


# Use manageable, balanced samples for classifier training.
train_positive_sample = random.sample(
    train_edges, min(50000, len(train_edges))
)

test_positive_sample = random.sample(
    test_edges, min(10000, len(test_edges))
)

train_negative = sample_negative_edges(
    G, len(train_positive_sample), seed=42
)

test_negative = sample_negative_edges(
    G, len(test_positive_sample), seed=43
)

# --------------------------------------------------
# 5. Save datasets
# --------------------------------------------------

def save_edges(filename, edge_list):
    with open(RESULTS_DIR / filename, "w") as file:
        for u, v in edge_list:
            file.write(f"{u}\t{v}\n")


save_edges("train_edges.tsv", train_edges)
save_edges("test_edges.tsv", test_edges)
save_edges("train_positive.tsv", train_positive_sample)
save_edges("train_negative.tsv", train_negative)
save_edges("test_positive.tsv", test_positive_sample)
save_edges("test_negative.tsv", test_negative)

print("\nLink-prediction datasets saved in:", RESULTS_DIR)
print("Training classifier samples:", len(train_positive_sample),
      "positive +", len(train_negative), "negative")
print("Testing samples:", len(test_positive_sample),
      "positive +", len(test_negative), "negative")
print("Preparation completed.")
