# =========================================
# FULL DOCUMENT CHUNK VIEWER + EMBEDDING SYSTEM
# =========================================

import os
import re
import pickle
import numpy as np
import pandas as pd
import faiss

from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from charset_normalizer import from_path


# =========================================
# LOAD MODEL
# =========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model loaded!")

# =========================================
# DOCUMENT FOLDER
# =========================================

DOCUMENT_FOLDER = "sample"

# =========================================
# STORAGE
# =========================================

all_embeddings = []

# =========================================
# CLEAN TEXT
# =========================================
def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

# =========================================
# READ PDF
# =========================================

def read_pdf(path):

    text = ""

    try:

        reader = PdfReader(path)

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    except Exception as e:

        print(f"[ERROR] PDF Failed: {path}")
        print(e)

    return clean_text(text)


# =========================================
# READ DOCX
# =========================================

def read_docx(path):

    try:

        doc = Document(path)

        text =  "\n".join(
            p.text for p in doc.paragraphs
        )
        return clean_text(text)

    except Exception as e:

        print(f"[ERROR] DOCX Failed: {path}")
        print(e)

        return ""


# =========================================
# READ TXT
# =========================================

def read_txt(path):

    try:

        result = from_path(path).best()

        encoding = (
            result.encoding
            if result
            else "utf-8"
        )

        with open(
            path,
            "r",
            encoding=encoding,
            errors="ignore"
        ) as f:

            text = f.read()
        return clean_text(text)

    except Exception as e:

        print(f"[ERROR] TXT Failed: {path}")
        print(e)

        return ""


# =========================================
# READ XLSX
# =========================================

def read_xlsx(path):

    try:

        df = pd.read_excel(
            path,
            engine="openpyxl"
        )

        text = df.astype(str).to_string()
        return clean_text(text)

    except Exception as e:

        print(f"[ERROR] XLSX Failed: {path}")
        print(e)

        return ""


# =========================================
# OVERLAPPING CHUNK FUNCTION
# =========================================

def chunk_text(
    text,
    chunk_size=300,
    overlap=50
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )
        
        if len(chunk.strip())>30:
            chunks.append(chunk)

        start += (chunk_size - overlap)

    return chunks


# =========================================
# PROCESS DOCUMENTS
# =========================================

print("\nProcessing documents...")

for file in os.listdir(DOCUMENT_FOLDER):

    path = os.path.join(
        DOCUMENT_FOLDER,
        file
    )

    if not os.path.isfile(path):
        continue

    content = ""

    print(f"\n===================================")
    print(f"Processing File: {file}")
    print("===================================")

    # =========================================
    # PDF
    # =========================================

    if file.endswith(".pdf"):

        content = read_pdf(path)

    # =========================================
    # DOCX
    # =========================================

    elif file.endswith(".docx"):

        content = read_docx(path)

    # =========================================
    # TXT
    # =========================================

    elif file.endswith(".txt"):

        content = read_txt(path)

    # =========================================
    # XLSX
    # =========================================

    elif file.endswith(".xlsx"):

        content = read_xlsx(path)

    else:

        print("[SKIPPED] Unsupported file")
        continue

    # =========================================
    # EMPTY CHECK
    # =========================================

    if not content.strip():

        print("[WARNING] Empty content")
        continue

    # =========================================
    # CREATE CHUNKS
    # =========================================

    chunks = chunk_text(
        content,
        chunk_size=300,
        overlap=50
    )

    print(f"\nChunks created: {len(chunks)}")

    # =========================================
    # SHOW CHUNKS
    # =========================================

    print("\n========== GENERATED CHUNKS ==========")

    for i, chunk in enumerate(chunks):

        word_count = len(chunk.split())

        print(f"\nChunk {i+1}")
        print(f"Words: {word_count}")

        print("\nPreview:\n")

        print(chunk[:500])

        print("\n-----------------------------------")

    # =========================================
    # SAVE CHUNKS TO DEBUG FILE
    # =========================================

    debug_file = f"{file}_chunks.txt"

    with open(debug_file, "w") as f:

        for i, chunk in enumerate(chunks):

            f.write(
                f"\n========== Chunk {i+1} ==========\n"
            )

            f.write(chunk)

            f.write("\n\n")

    print(f"\nChunk debug saved: {debug_file}")

    # =========================================
    # CREATE EMBEDDINGS
    # =========================================

    print("\nCreating embeddings...")

    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy = True
    )

    # =========================================
    # STORE EMBEDDINGS
    # =========================================

    for i, (chunk, embedding) in enumerate(

        zip(chunks, embeddings)

    ):

        all_embeddings.append({

            "file": file,

            "chunk_id": i + 1,

            "text": chunk,

            "embedding": embedding

        })

    print("Embeddings created!")

# =========================================
# EMPTY CHECK
# =========================================

if len(all_embeddings) == 0:

    print("\nNo embeddings created!")
    exit()

# =========================================
# NUMPY CONVERSION
# =========================================

embedding_vectors = np.array(

    [
        item["embedding"]
        for item in all_embeddings
    ]

).astype("float32")

print("\n===================================")
print("Embedding Matrix Shape")
print("===================================")

print(embedding_vectors.shape)

# =========================================
# NORMALIZE
# =========================================
faiss.normalize_L2(embedding_vectors)

# =========================================
# CREATE FAISS INDEX
# =========================================

dimension = embedding_vectors.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embedding_vectors
)

print("\nFAISS index created!")

# =========================================
# SAVE FAISS INDEX
# =========================================

faiss.write_index(

    index,

    "document_index.faiss"

)

print("FAISS index saved!")

# =========================================
# SAVE METADATA
# =========================================

with open(
    "metadata.pkl",
    "wb"
) as f:

    pickle.dump(
        all_embeddings,
        f
    )

print("Metadata saved!")

# =========================================
# DONE
# =========================================

print("\n===================================")
print("DONE SUCCESSFULLY!")
print("===================================")
