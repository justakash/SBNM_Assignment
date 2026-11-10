
"""
Network analysis of the BioGRID human physical
protein-protein interaction network.

1. Load the cleaned PPI edge list.
2. Calculate network topology statistics.
3. Identify connected components and hub proteins.
4. Generate and save the degree distribution plot.
"""

from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ppi_edges_clean.tsv"
RESULTS_DIR = BASE_DIR / "results"

# Create results directory if it does not exist.
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load the cleaned PPI network
# --------------------------------------------------

G = nx.read_edgelist(
    DATA_FILE,
    delimiter="\t",
    nodetype=int,
    create_using=nx.Graph()
)

print("Network Analysis — BioGRID Human Physical PPI")
print("-" * 50)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Is directed:", G.is_directed())


# --------------------------------------------------
# 3. Connected component analysis
# --------------------------------------------------

components = list(nx.connected_components(G))

print("\nConnected components:", len(components))

# Identify the largest connected component.
largest_component = max(components, key=len)

print("Largest component size:", len(largest_component))
print(
    "Largest component percentage:",
    round(
        len(largest_component) / G.number_of_nodes() * 100,
        2
    ),
    "%"
)

# Sort components by size.
components_sorted = sorted(
    components,
    key=len,
    reverse=True
)

print("\nConnected Component Details")
print("-" * 50)

for i, component in enumerate(components_sorted, start=1):
    print(f"Component {i}: {len(component)} nodes")

    # Display protein IDs for small components.
    if len(component) <= 10:
        print("Protein IDs:", sorted(component))


# --------------------------------------------------
# 4. Network density and isolated nodes
# --------------------------------------------------

print("\nNetwork Properties")
print("-" * 50)

print("Network density:", round(nx.density(G), 6))

isolated_nodes = list(nx.isolates(G))
print("Isolated nodes:", len(isolated_nodes))


# --------------------------------------------------
# 5. Degree distribution and statistics
# --------------------------------------------------

degrees = dict(G.degree())
degree_values = list(degrees.values())

print("\nDegree Statistics")
print("-" * 50)

print(
    "Average degree:",
    round(sum(degree_values) / len(degree_values), 2)
)

print("Minimum degree:", min(degree_values))
print("Maximum degree:", max(degree_values))


# --------------------------------------------------
# 6. Identify top 10 hub proteins
# --------------------------------------------------

top_nodes = sorted(
    degrees.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nTop 10 Hub Proteins (Entrez Gene IDs)")

for node, degree in top_nodes:
    print(node, degree)


# --------------------------------------------------
# 7. Generate degree distribution plot
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(degree_values, bins=50)

plt.xlabel("Node Degree")
plt.ylabel("Number of Proteins")
plt.title("Degree Distribution — BioGRID Human Physical PPI")

# Log scale highlights the highly skewed distribution.
plt.yscale("log")

plt.tight_layout()

# Save plot inside the results directory.
plot_file = RESULTS_DIR / "degree_distribution.png"

plt.savefig(plot_file, dpi=300)

print("\nDegree distribution plot saved to:")
print(plot_file)

plt.show()
