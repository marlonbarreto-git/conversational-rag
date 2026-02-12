from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class ConversationTurn:
    user_query: str
    reformulated_query: str
    response: str
    sources: list[str] = field(default_factory=list)
