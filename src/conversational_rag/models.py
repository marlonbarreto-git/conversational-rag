from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """A single message in a conversation with role, content, and timestamp."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class ConversationTurn:
    """Result of a single RAG query including the original, reformulated query, response, and sources."""

    user_query: str
    reformulated_query: str
    response: str
    sources: list[str] = field(default_factory=list)
