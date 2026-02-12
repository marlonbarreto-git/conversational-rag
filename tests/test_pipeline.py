"""Tests for the ConversationalRAG pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conversational_rag.models import ConversationTurn, Message
from conversational_rag.pipeline import ConversationalRAG


@pytest.fixture
def mock_dependencies():
    """Mock SentenceTransformer to avoid loading real models."""
    with patch("conversational_rag.retriever.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_st.return_value = mock_model
        yield mock_model


@pytest.fixture
def rag(mock_dependencies):
    return ConversationalRAG()


class TestConversationalRAGInit:
    def test_initialization(self, rag):
        assert rag.memory is not None
        assert rag.reformulator is not None
        assert rag.retriever is not None

    def test_initialization_creates_empty_state(self, rag):
        assert rag.get_history() == []


class TestConversationalRAGIndex:
    def test_index_stores_documents(self, rag, mock_dependencies):
        mock_dependencies.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        rag.index(["doc one", "doc two"])
        assert rag.retriever._documents == ["doc one", "doc two"]


class TestConversationalRAGQuery:
    def test_query_returns_conversation_turn(self, rag, mock_dependencies):
        mock_dependencies.encode.side_effect = [
            np.array([[1.0, 0.0], [0.0, 1.0]]),  # index
            np.array([[0.5, 0.5]]),  # reformulator encode query
            np.array([[1.0, 0.0]]),  # search encode query
        ]
        rag.index(["relevant doc", "other doc"])

        # Reset side_effect for query phase
        mock_dependencies.encode.side_effect = [
            np.array([[0.5, 0.5]]),  # reformulator
            np.array([[1.0, 0.0]]),  # search
        ]
        result = rag.query("test question")

        assert isinstance(result, ConversationTurn)
        assert result.user_query == "test question"
        assert isinstance(result.reformulated_query, str)
        assert isinstance(result.response, str)
        assert isinstance(result.sources, list)

    def test_query_stores_messages_in_memory(self, rag, mock_dependencies):
        mock_dependencies.encode.return_value = np.array([[0.1, 0.2]])
        rag.index(["some doc"])

        mock_dependencies.encode.return_value = np.array([[0.1, 0.2]])
        rag.query("hello")

        history = rag.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "hello"
        assert history[1].role == "assistant"

    def test_query_on_empty_index_returns_no_info_response(self, rag, mock_dependencies):
        mock_dependencies.encode.return_value = np.array([[0.1, 0.2]])
        result = rag.query("question with no docs")

        assert result.response == "No relevant information found."
        assert result.sources == []


class TestConversationalRAGSequentialQueries:
    def test_sequential_queries_maintain_history(self, rag, mock_dependencies):
        mock_dependencies.encode.return_value = np.array([[0.1, 0.2]])
        rag.index(["some doc"])

        mock_dependencies.encode.return_value = np.array([[0.1, 0.2]])
        rag.query("first question")
        rag.query("second question")

        history = rag.get_history()
        assert len(history) == 4  # 2 user + 2 assistant messages
        assert history[0].content == "first question"
        assert history[2].content == "second question"


class TestConversationalRAGGetHistory:
    def test_get_history_returns_messages(self, rag, mock_dependencies):
        mock_dependencies.encode.return_value = np.array([[0.1, 0.2]])
        rag.index(["doc"])

        mock_dependencies.encode.return_value = np.array([[0.1, 0.2]])
        rag.query("q1")

        history = rag.get_history()
        assert len(history) >= 2
        assert all(isinstance(m, Message) for m in history)


class TestConversationalRAGReset:
    def test_reset_clears_memory(self, rag, mock_dependencies):
        mock_dependencies.encode.return_value = np.array([[0.1, 0.2]])
        rag.index(["doc"])

        mock_dependencies.encode.return_value = np.array([[0.1, 0.2]])
        rag.query("question")

        assert len(rag.get_history()) > 0
        rag.reset()
        assert rag.get_history() == []
