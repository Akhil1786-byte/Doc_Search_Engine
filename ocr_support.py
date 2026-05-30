# =========================
# OCR
# =========================

import os
import re
import pickle
import faiss
import fitz
import cv2
import pytesseract
import numpy as np

from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================
# FOLDERS
# =========================

DOCUMENTS_FOLDER = "sample"

# =========================
# LOAD MODEL
# =========================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

print("Model loaded!")

# =========================
# CLEAN OCR TEXT
# =========================
def clean_ocr_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.replace("-\n", "")
    text = text.strip()
    return text

# =========================
# OCR PDF TEXT EXTRACTOR
# =========================

def extract_text_from_pdf(pdf_path):

    full_text = ""

    try:

        doc = fitz.open(pdf_path)

        print(f"\nReading: {pdf_path}")

        for page_num in range(len(doc)):

            page = doc[page_num]

            # =========================
            # NORMAL TEXT EXTRACTION
            # =========================

            text = page.get_text()

            # If normal PDF text exists
            if text and len(text.strip())>30:

                full_text += text + "\n"

            else:

                print(
                    f"OCR Page: {page_num + 1}"
                )

                # =========================
                # CONVERT PAGE TO IMAGE
                # =========================

                zoom = 2.5
                mat = fitz.Matrix(zoom, zoom)

                pix = page.get_pixmap(matrix = mat)

                img = Image.frombytes(

                    "RGB",

                    [pix.width, pix.height],

                    pix.samples
                )

                img_np = np.array(img)

                # =========================
                # IMAGE PREPROCESSING
                # =========================

                gray = cv2.cvtColor(

                    img_np,

                    cv2.COLOR_RGB2GRAY
                )

                # Enlarge Image
                gray = cv2.resize(

                    gray,
                    None,
                    fx=2,
                    fy=2,
                    interpolation=cv2.INTER_CUBIC
                )

                # Denoise
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

                # Thresholding
                gray = cv2.threshold(

                    gray,

                    0,

                    255,

                    cv2.THRESH_BINARY + cv2.THRESH_OTSU

                )[1]

                # =========================
                # OCR
                # =========================

                ocr_text = pytesseract.image_to_string(
                    gray,
                    conifg = "--oem 3 --psm 11"
                )
                ocr_text = clean_ocr_text(ocr_text)

                full_text += (ocr_text + "\n")

        doc.close()

    except Exception as e:

        print(
            "ERROR:",
            pdf_path,
            e
        )

    return full_text

# =========================
# SPLIT TEXT INTO CHUNKS
# =========================

def split_text(text, chunk_size=700):

    words = text.split("\n")

    chunks = []

    for w in words:

        w = w.strip()

        if not w:
            continue        

        if len(current_chunk) + len(w) < chunk_size:
            current_chunk += (w + "\n")

        else:
            chunks.append(current_chunk.strip())
            current_chunk = (w + "\n")
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# =========================
# READ DOCUMENTS
# =========================

all_chunks = []

metadata = []

print("Scanning documents...")

for filename in tqdm(

    os.listdir(DOCUMENTS_FOLDER)

):

    if not filename.lower().endswith(".pdf"):

        continue

    pdf_path = os.path.join(

        DOCUMENTS_FOLDER,

        filename
    )

    # =========================
    # EXTRACT TEXT
    # =========================

    text = extract_text_from_pdf(
        pdf_path
    )

    text = clean_ocr_text(text)

    if len(text.strip())<20:

        print(
            f"No text found: {filename}"
        )

        continue

    # =========================
    # SPLIT INTO CHUNKS
    # =========================

    chunks = split_text(
        text,
        chunk_size=700
    )

    for chunk in chunks:

        if len(chunk.strip()) < 20:

            continue

        all_chunks.append(chunk)

        metadata.append({

            "file": filename,

            "text": chunk

        })

print(
    f"Total Chunks: {len(all_chunks)}"
)

# =========================
# CREATE EMBEDDINGS
# =========================

print("Creating embeddings...")

embeddings = model.encode(

    all_chunks,

    batch_size=32,

    show_progress_bar=True,

    convert_to_numpy=True
)

embeddings = embeddings.astype(
    "float32"
)

# =========================
# NORMALIZE EMBEDDINGS
# =========================

faiss.normalize_L2(
    embeddings
)

# =========================
# CREATE FAISS INDEX
# =========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print(
    f"Indexed Vectors: {index.ntotal}"
)

# =========================
# SAVE INDEX
# =========================

faiss.write_index(

    index,

    "document_index.faiss"
)

# =========================
# SAVE METADATA
# =========================

with open(

    "metadata.pkl",

    "wb"

) as f:

    pickle.dump(
        metadata,
        f
    )

print("Index saved successfully!")
