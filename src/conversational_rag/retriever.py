"""Simple vector retriever using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._documents: list[str] = []
        self._embeddings: np.ndarray | None = None

    def index(self, documents: list[str]) -> None:
        self._documents = list(documents)
        self._embeddings = self._model.encode(documents, show_progress_bar=False)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._documents or self._embeddings is None:
            return []

        query_embedding = self._model.encode([query], show_progress_bar=False)
        similarities = self._cosine_similarity(query_embedding, self._embeddings)[0]

        top_k = min(top_k, len(self._documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self._documents[i], float(similarities[i])) for i in top_indices]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return a_norm @ b_norm.T
