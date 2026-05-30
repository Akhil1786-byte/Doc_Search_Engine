from flask import Flask, render_template, request, jsonify, send_from_directory
import os, re, pickle, numpy as np, faiss, traceback
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

DOCUMENTS_FOLDER = "sample"
RESULT_PER_PAGE = 20

# =========================
# LOAD EMBEDDING MODEL (LIGHT & FAST)
# =========================
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("Model loaded")

# =========================
# LOAD FAISS
# =========================
index = faiss.read_index("document_index.faiss")
faiss.omp_set_num_threads(2)

# =========================
# LOAD METADATA
# =========================
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# FILE TYPE DETECTION
# =========================
def get_file_type(filename):
    filename = filename.lower()
    if filename.endswith(".pdf"):
        return "pdf"
    elif filename.endswith((".xlsx", ".xls")):
        return "xlsx"
    elif filename.endswith(".txt"):
        return "txt"
    elif filename.endswith((".docx", ".doc")):
        return "docx"
    return "other"

# =========================
# DOCUMENT LIST (FAST)
# =========================
all_documents = []
seen = set()

for item in metadata:
    f = item["file"]
    if f not in seen:
        seen.add(f)
        all_documents.append({
            "file": f,
            "type": get_file_type(f)
        })

# =========================
# VOCAB (FOR TYPOS)
# =========================
vocab = set()
for item in metadata:
    for w in clean_text(item["text"]).split():
        if len(w) > 2:
            vocab.add(w)
vocab = list(vocab)

# =========================
# TYPO FIX
# =========================
def correct_query(query):
    words = []
    for w in query.lower().split():
        if len(w) <= 2:
            words.append(w)
            continue

        match = process.extractOne(w, vocab, scorer=fuzz.ratio)
        if match and match[1] > 80:
            words.append(match[0])
        else:
            words.append(w)

    return " ".join(words)

# =========================
# BM25
# =========================
bm25_corpus = [clean_text(m["text"]).split() for m in metadata]
bm25 = BM25Okapi(bm25_corpus)

# =========================
# SIMPLE FAST QUERY EXPANSION (FIXED)
# =========================
def expand_query(query):

    original_words = set(query.lower().split())

    expanded = []
    seen = set()

    # Limit scan for speed
    for item in metadata[:300]:

        words = clean_text(item["text"]).split()

        for w in words:

            # Skip short words
            if len(w) < 4:
                continue

            # Skip if already in original query
            if w in original_words:
                continue

            # Skip duplicates in expansion
            if w in seen:
                continue

            seen.add(w)
            expanded.append(w)

            # limit size
            if len(expanded) >= 15:
                break

        if len(expanded) >= 15:
            break

    # Final merge: original + clean expansion
    final = list(dict.fromkeys(query.split() + expanded))

    return "".join([f"{term}|" for term in final])
# =========================
# SEARCH ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():

    selected_type = request.args.get("type", "all")

    # =========================
    # SHOW FILE LIST ONLY (SIDEBAR FIX)
    # =========================
    if request.method == "GET":
        docs = [
            d for d in all_documents
            if selected_type == "all" or d["type"] == selected_type
        ]

        return render_template(
            "index.html",
            documents=docs,
            results=[],
            rag_answer="",
            corrected_query="",
            expanded_query="",
            original_query="",
            selected_type=selected_type
        )

    # =========================
    # SEARCH MODE
    # =========================
    query = request.form["query"]

    corrected_query = correct_query(query)
    expanded_query = expand_query(corrected_query)

    final_query = expanded_query

    # EMBEDDING
    q_emb = model.encode(final_query, convert_to_numpy=True).astype("float32")
    q_emb = np.array([q_emb])
    faiss.normalize_L2(q_emb)

    # FAISS SEARCH
    scores, idxs = index.search(q_emb, 200)

    scores = scores[0]
    idxs = idxs[0]

    # BM25
    bm25_scores = bm25.get_scores(clean_text(final_query).split())

    results = []
    seen_files = set()

    for i, idx in enumerate(idxs):

        if idx >= len(metadata):
            continue

        item = metadata[idx]
        file_name = item["file"]

        if file_name in seen_files:
            continue

        seen_files.add(file_name)

        hybrid = scores[i] * 0.7 + bm25_scores[idx] * 0.3

        results.append({
            "file": file_name,
            "text": item["text"][:500],
            "score": round(float(hybrid), 4),
            "type": get_file_type(file_name)
        })

        if len(results) >= RESULT_PER_PAGE:
            break

    return render_template(
        "index.html",
        documents=[],
        results=results,
        rag_answer="",   # GROQ REMOVED (FAST MODE)
        corrected_query=corrected_query,
        expanded_query=expanded_query,
        original_query=query,
        selected_type=selected_type
    )

# =========================
# AUTOCOMPLETE (FAST FIX)
# =========================
@app.route("/autocomplete")
def autocomplete():

    q = request.args.get("q", "").lower()
    if len(q) < 2:
        return jsonify({"suggestions": []})

    matches = process.extract(q, vocab, limit=8)

    return jsonify({
        "suggestions": [{"text": m[0]} for m in matches]
    })

# =========================
# OPEN FILE
# =========================
@app.route("/open/<path:filename>")
def open_document(filename):
    return send_from_directory(
        os.path.abspath(DOCUMENTS_FOLDER),
        filename,
        as_attachment=False
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
