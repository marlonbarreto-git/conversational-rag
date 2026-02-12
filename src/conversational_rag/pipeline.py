"""Conversational RAG pipeline tying memory, reformulation, and retrieval together."""

from conversational_rag.memory import ConversationMemory
from conversational_rag.models import ConversationTurn, Message
from conversational_rag.reformulator import QueryReformulator
from conversational_rag.retriever import Retriever

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
CONTEXT_WINDOW_SIZE = 10
MAX_RESPONSE_SOURCES = 2
NO_RESULTS_MESSAGE = "No relevant information found."


class ConversationalRAG:
    """End-to-end conversational RAG pipeline combining memory, query reformulation, and retrieval."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.memory = ConversationMemory()
        self.reformulator = QueryReformulator(model_name=model_name)
        self.retriever = Retriever(model_name=model_name)

    def index(self, documents: list[str]) -> None:
        """Index a collection of documents for retrieval.

        Args:
            documents: List of document texts to embed and index.
        """
        self.retriever.index(documents)

    def query(self, user_query: str, top_k: int = DEFAULT_TOP_K) -> ConversationTurn:
        """Process a user query through reformulation, retrieval, and response generation.

        Args:
            user_query: The raw user question.
            top_k: Number of top documents to retrieve.

        Returns:
            A ConversationTurn with the query, reformulated query, response, and sources.
        """
        history = self.memory.get_context_window(n=CONTEXT_WINDOW_SIZE)
        reformulated = self.reformulator.reformulate(user_query, history)
        results = self.retriever.search(reformulated, top_k=top_k)
        sources = [text for text, _ in results]
        response = " ".join(sources[:MAX_RESPONSE_SOURCES]) if sources else NO_RESULTS_MESSAGE

        self.memory.add_message("user", user_query)
        self.memory.add_message("assistant", response)

        return ConversationTurn(
            user_query=user_query,
            reformulated_query=reformulated,
            response=response,
            sources=sources,
        )

    def get_history(self) -> list[Message]:
        """Return the full conversation history.

        Returns:
            List of all messages in chronological order.
        """
        return self.memory.get_history()

    def reset(self) -> None:
        """Clear conversation history and reset the pipeline state."""
        self.memory.clear()
