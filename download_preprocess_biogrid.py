
"""
Preprocess BioGRID 5.0.263 human physical
protein-protein interaction data.

Input: Existing extracted BioGRID Tab 3 dataset.
Output: ppi_edges_clean.tsv

Filtering:
- Physical interactions only
- Homo sapiens only (Taxonomy ID 9606)
- Entrez Gene identifiers
- Unique undirected edges
- Self-loops removed
"""

import csv
from pathlib import Path


# --------------------------------------------------
# 1. File paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    Path.home()
    / "Downloads"
    / "BIOGRID-ALL-5.0.263.tab3.txt"
)

OUTPUT_FILE = BASE_DIR / "ppi_edges_clean.tsv"

HUMAN_TAXON_ID = "9606"


# --------------------------------------------------
# 2. Preprocess BioGRID dataset
# --------------------------------------------------

def preprocess_biogrid():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"BioGRID dataset not found: {INPUT_FILE}"
        )

    print("Reading BioGRID dataset:", INPUT_FILE)

    unique_edges = set()

    physical_interactions = 0
    valid_human_interactions = 0
    self_loops_removed = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.reader(file, delimiter="\t")

        header = next(reader)

        required_columns = [
            "Entrez Gene Interactor A",
            "Entrez Gene Interactor B",
            "Experimental System Type",
            "Organism ID Interactor A",
            "Organism ID Interactor B"
        ]

        column_indices = {
            column: header.index(column)
            for column in required_columns
        }

        for row in reader:

            # Retain physical interactions.
            if row[column_indices["Experimental System Type"]] != "physical":
                continue

            physical_interactions += 1

            # Retain human interactions only.
            organism_a = row[
                column_indices["Organism ID Interactor A"]
            ]

            organism_b = row[
                column_indices["Organism ID Interactor B"]
            ]

            if (
                organism_a != HUMAN_TAXON_ID
                or organism_b != HUMAN_TAXON_ID
            ):
                continue

            # Extract Entrez Gene identifiers.
            protein_a = row[
                column_indices["Entrez Gene Interactor A"]
            ]

            protein_b = row[
                column_indices["Entrez Gene Interactor B"]
            ]

            if not protein_a.isdigit() or not protein_b.isdigit():
                continue

            protein_a = int(protein_a)
            protein_b = int(protein_b)

            valid_human_interactions += 1

            # Remove self-interactions.
            if protein_a == protein_b:
                self_loops_removed += 1
                continue

            # Create an undirected edge.
            edge = tuple(sorted((protein_a, protein_b)))

            # Set removes duplicate edges.
            unique_edges.add(edge)

    # --------------------------------------------------
    # 3. Save cleaned edge list
    # --------------------------------------------------

    unique_proteins = set()

    with open(OUTPUT_FILE, "w") as output:

        for protein_a, protein_b in sorted(unique_edges):

            output.write(f"{protein_a}\t{protein_b}\n")

            unique_proteins.add(protein_a)
            unique_proteins.add(protein_b)

    # --------------------------------------------------
    # 4. Print preprocessing summary
    # --------------------------------------------------

    print("\nBIOGRID PREPROCESSING SUMMARY")
    print("-" * 45)

    print("BioGRID version: 5.0.263")
    print("Physical interactions:", physical_interactions)
    print("Valid human interactions:", valid_human_interactions)
    print("Self-loops removed:", self_loops_removed)
    print("Unique undirected edges:", len(unique_edges))
    print("Unique proteins:", len(unique_proteins))

    print("\nCleaned dataset saved to:")
    print(OUTPUT_FILE)


# --------------------------------------------------
# 5. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    preprocess_biogrid()

    print("\nBioGRID preprocessing completed successfully.")
