import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index", "index.faiss")
META_PATH = os.path.join(BASE_DIR, "faiss_index", "metadata.pkl")


class RAGEngine:
    def __init__(self):
        self.index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    def retrieve(self, query, k=5):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"]
            })

        return results
