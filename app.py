# =========================
# Flask App
# =========================

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

import os
import re
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi

# =========================
# Flask App
# =========================

app = Flask(__name__)

# =========================
# Documents Folder
# =========================

DOCUMENTS_FOLDER = "sample"

# =========================
# Load Model
# =========================

print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

print("Model loaded!")

# =========================
# Load FAISS
# =========================

print("Loading FAISS index...")

index = faiss.read_index(
    "document_index.faiss"
)

faiss.omp_set_num_threads(4)

print("FAISS loaded!")

# =========================
# Load Metadata
# =========================

print("Loading metadata...")

with open("metadata.pkl", "rb") as f:

    metadata = pickle.load(f)

print("Metadata loaded!")

# =========================
# Vocabulary
# =========================

print("Creating vocabulary...")

vocab = set()

for item in metadata:

    text = item["text"].lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    words = text.split()

    for word in words:

        word = word.strip()

        if len(word) > 2:

            vocab.add(word)

vocab = list(vocab)

print(f"Vocabulary Size: {len(vocab)}")

# =========================
# Typo Correction
# =========================

def correct_query(query):

    corrected_words = []

    for word in query.lower().split():

        if len(word) <= 2:

            corrected_words.append(word)

            continue

        match = process.extractOne(

            word,

            vocab,

            scorer=fuzz.ratio

        )

        if match:

            matched_word = match[0]

            score = match[1]

            if score >= 75:

                corrected_words.append(
                    matched_word
                )

            else:

                corrected_words.append(
                    word
                )

        else:

            corrected_words.append(
                word
            )

    return " ".join(corrected_words)

# =========================
# Suggestions
# =========================

print("Creating suggestions...")

suggestions = []

for item in metadata:

    text = item["text"].lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    words = text.split()

    limit = min(
        len(words) - 3,
        15
    )

    for i in range(limit):

        phrase = " ".join(
            words[i:i+4]
        ).strip()

        if len(phrase) > 10:

            suggestions.append({

                "text": phrase,

                "file": item["file"]

            })

# =========================
# Remove Duplicates
# =========================

unique_suggestions = {}

for item in suggestions:

    text = item["text"]

    if text not in unique_suggestions:

        unique_suggestions[text] = item

suggestions = list(
    unique_suggestions.values()
)

# Limit suggestions
suggestions = suggestions[:50000]

print(f"Suggestions: {len(suggestions)}")

# =========================
# Suggestion Embeddings
# =========================

print("Creating suggestion embeddings...")

suggestion_texts = [

    item["text"]

    for item in suggestions
]

suggestion_embeddings = model.encode(

    suggestion_texts,

    batch_size=256,

    convert_to_numpy=True,

    show_progress_bar=True

).astype("float32")

suggestion_embeddings = normalize(
    suggestion_embeddings
)

print("Suggestion embeddings ready!")

# =========================
# Query Expansion
# =========================

def expand_query(query):

    query_embedding = model.encode(

        query,

        convert_to_numpy=True

    ).astype("float32")

    query_embedding = normalize(
        [query_embedding]
    )[0]

    similarities = np.dot(

        suggestion_embeddings,

        query_embedding

    )

    top_indices = np.argsort(
        similarities
    )[-5:][::-1]

    expanded_parts = [query]

    for idx in top_indices:

        expanded_parts.append(
            suggestion_texts[idx]
        )

    expanded_parts = list(
        dict.fromkeys(expanded_parts)
    )

    expanded_query = " ".join(
        expanded_parts
    )

    return expanded_query

# =========================
# BM25 Corpus
# =========================

print("Creating BM25 corpus...")

bm25_corpus = []

for item in metadata:

    text = item["text"].lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    tokens = text.split()

    bm25_corpus.append(tokens)

print("BM25 corpus ready!")

# =========================
# BM25 Index
# =========================

print("Creating BM25 index...")

bm25 = BM25Okapi(
    bm25_corpus
)

print("BM25 ready!")

# =========================
# Home Route
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    results = []

    corrected_query = ""
    expanded_query = ""
    original_query = ""

    if request.method == "POST":

        query = request.form["query"]

        original_query = query

        # =========================
        # Typo Correction
        # =========================

        corrected_query = correct_query(
            query
        )

        print(
            "Corrected:",
            corrected_query
        )

        # =========================
        # Query Expansion
        # =========================

        expanded_query = expand_query(
            corrected_query
        )

        print(
            "Expanded:",
            expanded_query
        )

        final_query = expanded_query

        # =========================
        # Embedding
        # =========================

        query_embedding = model.encode(

            final_query,

            convert_to_numpy=True

        ).astype("float32")

        query_embedding = np.array(
            [query_embedding]
        )

        
        faiss.normalize_L2(
            query_embedding
        )

        # =========================
        # FAISS Search
        # =========================

        semantic_top_k = 50

        distances, indices = index.search(

            query_embedding,

            semantic_top_k

        )

        # =========================
        # BM25 Scores
        # =========================

        tokenized_query = final_query.lower().split()

        bm25_scores = bm25.get_scores(
            tokenized_query
        )

        # =========================
        # Score Normalization
        # =========================

        semantic_scores = distances[0]

        semantic_min = semantic_scores.min()
        semantic_max = semantic_scores.max()

        bm25_selected = []

        for idx in indices[0]:

            bm25_selected.append(
                bm25_scores[idx]
            )

        bm25_selected = np.array(
            bm25_selected
        )

        bm25_min = bm25_selected.min()
        bm25_max = bm25_selected.max()

        # =========================
        # Hybrid Ranking
        # =========================

        hybrid_results = []

        for i, idx in enumerate(indices[0]):

            if idx >= len(metadata):

                continue

            semantic_score = (
                semantic_scores[i] - semantic_min
            ) / (
                semantic_max - semantic_min + 1e-8
            )

            keyword_score = (
                bm25_scores[idx] - bm25_min
            ) / (
                bm25_max - bm25_min + 1e-8
            )

            hybrid_score = (

                semantic_score * 0.7 +

                keyword_score * 0.3

            )

            hybrid_results.append({

                "idx": idx,

                "score": hybrid_score

            })

        # Sort
        hybrid_results = sorted(

            hybrid_results,

            key=lambda x: x["score"],

            reverse=True

        )

        # =========================
        # Remove Duplicate Files
        # =========================

        shown_files = set()

        rank = 1

        for item in hybrid_results:

            idx = item["idx"]

            result = metadata[idx]

            file_name = result["file"]

            if file_name in shown_files:

                continue

            shown_files.add(
                file_name
            )

            results.append({

                "rank": rank,

                "file": file_name,

                "text": result["text"][:500],

                "score": round(
                    item["score"],
                    4
                )

            })

            rank += 1

            if rank > 20:

                break

    return render_template(

        "index.html",

        results=results,

        corrected_query=corrected_query,

        expanded_query=expanded_query,

        original_query=original_query
    )

# =========================
# Autocomplete API
# =========================

@app.route("/autocomplete")
def autocomplete():

    query = request.args.get(
        "q",
        ""
    ).lower()

    if len(query) < 2:

        return jsonify({
            "suggestions": []
        })

    matches = process.extract(

        query,

        suggestion_texts,

        limit=10
    )

    results = []

    for match in matches:

        matched_text = match[0]

        for item in suggestions:

            if item["text"] == matched_text:

                results.append({

                    "text": item["text"],

                    "file": item["file"]

                })

                break

    return jsonify({
        "suggestions": results
    })

# =========================
# Open Document Route
# =========================

@app.route("/open/<path:filename>")
def open_document(filename):

    print("Opening:", filename)

    return send_from_directory(

        os.path.abspath(
            DOCUMENTS_FOLDER
        ),

        filename,

        as_attachment=False
    )

# =========================
# Run App
# =========================

if __name__ == "__main__":

    app.run(
        debug=True
    )

