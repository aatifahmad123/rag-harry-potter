import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from load_docs import load_pdfs
from chunk_docs import chunk_documents


INDEX_DIR = "../faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "metadata.pkl")


def create_vector_store():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Loading documents...")
    documents = load_pdfs()

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    texts = [chunk["text"] for chunk in chunks]

    print("Loading sentence-transformer model (CPU)...")
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,               # Lightning CPU can handle this
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Saving FAISS index and metadata...")
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Stored {len(chunks)} vectors successfully.")


if __name__ == "__main__":
    create_vector_store()
