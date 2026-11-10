
"""
Node2Vec implementation on BioGRID Human Physical PPI Network

Purpose:
1. Load the cleaned BioGRID protein-protein interaction network.
2. Generate biased random walks using Node2Vec.
3. Train Skip-gram embeddings.
4. Save learned protein representations for downstream evaluation.
"""

from pathlib import Path
import networkx as nx
from node2vec import Node2Vec

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ppi_edges_clean.tsv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. Load the cleaned PPI network
# --------------------------------------------------

# Each row represents one undirected protein interaction.
# Entrez Gene IDs are used as protein identifiers.

G = nx.read_edgelist(
    DATA_FILE,
    delimiter="\t",
    nodetype=int,
    create_using=nx.Graph()
)

print("BioGRID PPI Network Loaded")
print("Proteins:", G.number_of_nodes())
print("Interactions:", G.number_of_edges())

# --------------------------------------------------
# 3. Configure Node2Vec
# --------------------------------------------------

# p = return parameter
# q = in-out parameter
# These values define our baseline experiment.

node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=20,
    num_walks=10,
    p=1,
    q=1,
    workers=2,
    seed=42,
    quiet=False
)

# --------------------------------------------------
# 4. Train Skip-gram model
# --------------------------------------------------

# The model learns a vector representation for each protein
# based on its sampled random-walk neighbourhoods.

model = node2vec.fit(
    window=10,
    min_count=1,
    batch_words=128,
    sg=1,
    workers=2,
    seed=42
)

# --------------------------------------------------
# 5. Save learned protein embeddings
# --------------------------------------------------

# Each row contains a protein ID followed by its 64-dimensional
# learned feature representation.

embedding_file = RESULTS_DIR / "node2vec_baseline_embeddings.txt"

model.wv.save_word2vec_format(str(embedding_file))

print("\nBaseline Node2Vec Training Completed")
print("Embedding dimensions:", model.vector_size)
print("Number of embedded proteins:", len(model.wv))
print("Saved embeddings:", embedding_file)
