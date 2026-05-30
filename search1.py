# __define-ocg__ SEMANTIC SEARCH SYSTEM

import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# =========================
# Load Embedding Model
# =========================

print("Loading model...")

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

print("Model loaded!")

# =========================
# Load FAISS Index
# =========================

print("\nLoading FAISS index...")

index = faiss.read_index(
    "document_index.faiss"
)

print("FAISS index loaded!")

# =========================
# Load Metadata
# =========================

print("\nLoading metadata...")

with open(
    "metadata.pkl",
    "rb"
) as f:

    metadata = pickle.load(f)

print("Metadata loaded!")

# =========================
# Search Loop
# =========================

while True:

    print("\n=========================")

    query = input(
        "\nEnter your search query: "
    )

    # Exit option
    if query.lower() == "exit":

        print("\nExiting search system...")

        break

    # =========================
    # Convert Query to Embedding
    # =========================

    query_embedding = model.encode(
        query
    )

    query_embedding = np.array(
        [query_embedding]
    ).astype("float32")

    # =========================
    # Search Top Results
    # =========================

    top_k = 20

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    print("\nTop Results:\n")


    # Show Result
    shown_files = set()
    rank = 1
    for idx in indices[0]:
        result = metadata[idx]
        file_name = result["file"]

        # Skip Duplicate Files
        if file_name in shown_files:
            continue

        shown_files.add(file_name)
        print(f"\nRank: {rank}")
        print(f"File: {file_name}")
        print(f"\nMatched Text: \n")
        print(result["text"][:500])
        print("\n---------------------------------")
        rank += 1

        if rank > 10:
            break 
