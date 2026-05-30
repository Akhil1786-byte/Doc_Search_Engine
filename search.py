from flask import Flask, render_template, request
import os
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)

DOCUMENT_FOLDER = "sample"

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(path):
    text = ""
    reader = PdfReader(path)

    for page in reader.pages:
        text += page.extract_text() or ""

    return text

def read_docx(path):
    doc = Document(path)

    return "\n".join([p.text for p in doc.paragraphs])

@app.route("/", methods=["GET", "POST"])
def index():

    results = []

    if request.method == "POST":

        keyword = request.form["keyword"].lower()

        for file in os.listdir(DOCUMENT_FOLDER):

            path = os.path.join(DOCUMENT_FOLDER, file)

            content = ""

            if file.endswith(".txt"):
                content = read_txt(path)

            elif file.endswith(".pdf"):
                content = read_pdf(path)

            elif file.endswith(".docx"):
                content = read_docx(path)

            if keyword in content.lower():
                results.append(file)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
