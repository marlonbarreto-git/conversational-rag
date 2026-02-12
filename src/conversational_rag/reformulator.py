import re

from conversational_rag.models import Message

PRONOUN_PATTERN = re.compile(
    r"\b(it|they|this|that|these|those|its|their|them|he|she)\b",
    re.IGNORECASE,
)


class QueryReformulator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def reformulate(self, query: str, history: list[Message]) -> str:
        if not history:
            return query

        if not PRONOUN_PATTERN.search(query):
            return query

        last_user_topic = self._get_last_user_topic(history)
        if not last_user_topic:
            return query

        return f"Regarding {last_user_topic}: {query}"

    def _get_last_user_topic(self, history: list[Message]) -> str | None:
        for msg in reversed(history):
            if msg.role == "user":
                return msg.content[:100]
        return None
