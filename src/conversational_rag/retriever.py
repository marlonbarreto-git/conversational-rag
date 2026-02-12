"""Simple vector retriever using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


class Retriever:
    """Embeds documents with sentence-transformers and retrieves the most similar ones via cosine similarity."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)
        self._documents: list[str] = []
        self._embeddings: np.ndarray | None = None

    def index(self, documents: list[str]) -> None:
        """Encode and store document embeddings for later retrieval.

        Args:
            documents: List of document texts to index.
        """
        self._documents = list(documents)
        self._embeddings = self._model.encode(documents, show_progress_bar=False)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[str, float]]:
        """Find the top-k most similar documents to the query.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.

        Returns:
            List of (document_text, similarity_score) tuples sorted by descending similarity.
        """
        if not self._documents or self._embeddings is None:
            return []

        query_embedding = self._model.encode([query], show_progress_bar=False)
        similarities = self._cosine_similarity(query_embedding, self._embeddings)[0]

        top_k = min(top_k, len(self._documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self._documents[i], float(similarities[i])) for i in top_indices]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of vectors.

        Args:
            a: First matrix of row vectors.
            b: Second matrix of row vectors.

        Returns:
            Similarity matrix of shape (len(a), len(b)).
        """
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return a_norm @ b_norm.T
