"""Conversational RAG pipeline tying memory, reformulation, and retrieval together."""

from conversational_rag.memory import ConversationMemory
from conversational_rag.models import ConversationTurn
from conversational_rag.reformulator import QueryReformulator
from conversational_rag.retriever import Retriever


class ConversationalRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.memory = ConversationMemory()
        self.reformulator = QueryReformulator(model_name=model_name)
        self.retriever = Retriever(model_name=model_name)

    def index(self, documents: list[str]) -> None:
        self.retriever.index(documents)

    def query(self, user_query: str, top_k: int = 3) -> ConversationTurn:
        history = self.memory.get_context_window(n=10)
        reformulated = self.reformulator.reformulate(user_query, history)
        results = self.retriever.search(reformulated, top_k=top_k)
        sources = [text for text, _ in results]
        response = " ".join(sources[:2]) if sources else "No relevant information found."

        self.memory.add_message("user", user_query)
        self.memory.add_message("assistant", response)

        return ConversationTurn(
            user_query=user_query,
            reformulated_query=reformulated,
            response=response,
            sources=sources,
        )

    def get_history(self):
        return self.memory.get_history()

    def reset(self):
        self.memory.clear()
