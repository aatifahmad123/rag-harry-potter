import os
from pypdf import PdfReader

DOCS_PATH = "../data/docs"

def load_pdfs():
    documents = []

    for file in os.listdir(DOCS_PATH):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(DOCS_PATH, file))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            documents.append({
                "source": file,
                "text": text
            })

    return documents


if __name__ == "__main__":
    docs = load_pdfs()
    print(f"Loaded {len(docs)} documents")
