# =========================
# OCR
# =========================

import os
import re
import pickle
import numpy as np
import faiss
import fitz
import cv2
import pytesseract

from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

DOCUMENTS_FOLDER = "sample"

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100

# =========================
# Load Embedding Model
# =========================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

print("Model loaded!")

# =========================
# OCR FUNCTION
# =========================

def extract_text_from_scanned_pdf(pdf_path):

    full_text = ""

    try:

        print(f"OCR Processing: {pdf_path}")

        pages = convert_from_path(

            pdf_path,

            dpi=200
        )

        for page_num, page in enumerate(pages):

            # PIL -> OpenCV
            image = np.array(page)

            image = cv2.cvtColor(

                image,

                cv2.COLOR_RGB2BGR
            )

            # =========================
            # Preprocessing
            # =========================

            gray = cv2.cvtColor(

                image,

                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.GaussianBlur(

                gray,

                (5, 5),

                0
            )

            thresh = cv2.threshold(

                gray,

                0,

                255,

                cv2.THRESH_BINARY + cv2.THRESH_OTSU

            )[1]

            # =========================
            # OCR
            # =========================

            text = pytesseract.image_to_string(
                thresh
            )

            full_text += "\n" + text

            print(
                f"OCR Page {page_num + 1} completed"
            )

    except Exception as e:

        print(f"OCR Error: {e}")

    return full_text

# =========================
# PDF TEXT EXTRACTION
# =========================

def extract_pdf_text(pdf_path):

    text = ""

    try:

        doc = fitz.open(pdf_path)

        for page in doc:

            page_text = page.get_text()

            text += page_text

        doc.close()

    except Exception as e:

        print(f"PDF Read Error: {e}")

    # =========================
    # OCR Fallback
    # =========================

    if len(text.strip()) < 50:

        print(
            f"Scanned PDF detected: {pdf_path}"
        )

        text = extract_text_from_scanned_pdf(
            pdf_path
        )

    return text

# =========================
# CLEAN TEXT
# =========================

def clean_text(text):

    text = re.sub(

        r"\s+",

        " ",

        text

    )

    return text.strip()

# =========================
# CHUNK TEXT
# =========================

def chunk_text(text):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk = words[start:end]

        chunk = " ".join(chunk)

        chunks.append(chunk)

        start += (
            CHUNK_SIZE - CHUNK_OVERLAP
        )

    return chunks

# =========================
# CREATE DATA
# =========================

documents = []

metadata = []

print("Reading PDF files...")

# =========================
# READ DOCUMENTS
# =========================

for filename in os.listdir(
    DOCUMENTS_FOLDER
):

    if not filename.lower().endswith(
        ".pdf"
    ):

        continue

    pdf_path = os.path.join(

        DOCUMENTS_FOLDER,

        filename
    )

    print(f"\nProcessing: {filename}")

    # =========================
    # Extract Text
    # =========================

    text = extract_pdf_text(
        pdf_path
    )

    text = clean_text(text)

    if len(text) == 0:

        print(
            f"Skipping empty file: {filename}"
        )

        continue

    print(
        f"Text Length: {len(text)}"
    )

    # =========================
    # Chunking
    # =========================

    chunks = chunk_text(
        text
    )

    print(
        f"Total Chunks: {len(chunks)}"
    )

    # =========================
    # Save Chunks
    # =========================

    for chunk in chunks:

        documents.append(
            chunk
        )

        metadata.append({

            "file": filename,

            "text": chunk

        })

# =========================
# Create Embeddings
# =========================

print("\nCreating embeddings...")

embeddings = model.encode(

    documents,

    batch_size=64,

    show_progress_bar=True,

    convert_to_numpy=True

).astype("float32")

print("Embeddings created!")

# =========================
# Normalize Embeddings
# =========================

faiss.normalize_L2(
    embeddings
)

# =========================
# Create FAISS Index
# =========================

dimension = embeddings.shape[1]

print("Creating FAISS index...")

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print(
    f"Total vectors indexed: {index.ntotal}"
)

# =========================
# Save FAISS Index
# =========================

faiss.write_index(

    index,

    "document_index.faiss"
)

print("FAISS index saved!")

# =========================
# Save Metadata
# =========================

with open(

    "metadata.pkl",

    "wb"

) as f:

    pickle.dump(
        metadata,
        f
    )

print("Metadata saved!")

# =========================
# COMPLETE
# =========================

print("\n=================================")

print("INDEX BUILD COMPLETED SUCCESSFULLY")

print("=================================")
