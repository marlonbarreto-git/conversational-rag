"""Tests for the Retriever module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conversational_rag.retriever import Retriever


class TestRetrieverInit:
    def test_default_model_name(self):
        with patch("conversational_rag.retriever.SentenceTransformer") as mock_st:
            retriever = Retriever()
            mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    def test_custom_model_name(self):
        with patch("conversational_rag.retriever.SentenceTransformer") as mock_st:
            retriever = Retriever(model_name="custom-model")
            mock_st.assert_called_once_with("custom-model")

    def test_initial_state_empty(self):
        with patch("conversational_rag.retriever.SentenceTransformer"):
            retriever = Retriever()
            assert retriever._documents == []
            assert retriever._embeddings is None


class TestRetrieverIndex:
    def test_index_stores_documents(self):
        with patch("conversational_rag.retriever.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_st.return_value = mock_model

            retriever = Retriever()
            docs = ["doc one", "doc two"]
            retriever.index(docs)

            assert retriever._documents == docs
            mock_model.encode.assert_called_once_with(docs, show_progress_bar=False)
            np.testing.assert_array_equal(
                retriever._embeddings, np.array([[0.1, 0.2], [0.3, 0.4]])
            )

    def test_index_replaces_previous_documents(self):
        with patch("conversational_rag.retriever.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.side_effect = [
                np.array([[0.1, 0.2]]),
                np.array([[0.3, 0.4], [0.5, 0.6]]),
            ]
            mock_st.return_value = mock_model

            retriever = Retriever()
            retriever.index(["first"])
            retriever.index(["second", "third"])

            assert retriever._documents == ["second", "third"]


class TestRetrieverSearch:
    def _make_retriever_with_docs(self, docs, embeddings, query_embedding):
        with patch("conversational_rag.retriever.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.side_effect = [
                embeddings,  # index call
                query_embedding,  # search call
            ]
            mock_st.return_value = mock_model

            retriever = Retriever()
            retriever.index(docs)
            return retriever

    def test_search_returns_results_sorted_by_score_descending(self):
        docs = ["low relevance", "high relevance", "medium relevance"]
        # Embeddings designed so "high relevance" is closest to query
        doc_embeddings = np.array([
            [0.0, 1.0],  # low
            [1.0, 0.0],  # high (same direction as query)
            [0.5, 0.5],  # medium
        ])
        query_embedding = np.array([[1.0, 0.0]])

        retriever = self._make_retriever_with_docs(docs, doc_embeddings, query_embedding)
        results = retriever.search("query", top_k=3)

        assert len(results) == 3
        assert results[0][0] == "high relevance"
        assert results[1][0] == "medium relevance"
        assert results[2][0] == "low relevance"
        # Scores should be descending
        assert results[0][1] >= results[1][1] >= results[2][1]

    def test_search_respects_top_k(self):
        docs = ["a", "b", "c", "d"]
        doc_embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.5, 0.5],
            [0.0, 1.0],
        ])
        query_embedding = np.array([[1.0, 0.0]])

        retriever = self._make_retriever_with_docs(docs, doc_embeddings, query_embedding)
        results = retriever.search("query", top_k=2)

        assert len(results) == 2

    def test_search_on_empty_index_returns_empty_list(self):
        with patch("conversational_rag.retriever.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            retriever = Retriever()
            results = retriever.search("anything")

            assert results == []

    def test_search_returns_tuples_of_text_and_float_score(self):
        docs = ["only doc"]
        doc_embeddings = np.array([[1.0, 0.0]])
        query_embedding = np.array([[1.0, 0.0]])

        retriever = self._make_retriever_with_docs(docs, doc_embeddings, query_embedding)
        results = retriever.search("query", top_k=1)

        assert len(results) == 1
        text, score = results[0]
        assert isinstance(text, str)
        assert isinstance(score, float)

    def test_search_top_k_larger_than_index_returns_all(self):
        docs = ["a", "b"]
        doc_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        query_embedding = np.array([[1.0, 0.0]])

        retriever = self._make_retriever_with_docs(docs, doc_embeddings, query_embedding)
        results = retriever.search("query", top_k=10)

        assert len(results) == 2
