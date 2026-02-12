import re

from conversational_rag.models import Message

PRONOUN_PATTERN = re.compile(
    r"\b(it|they|this|that|these|those|its|their|them|he|she)\b",
    re.IGNORECASE,
)

MAX_TOPIC_LENGTH = 100


class QueryReformulator:
    """Rewrites user queries by resolving pronouns using conversation history."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def reformulate(self, query: str, history: list[Message]) -> str:
        """Reformulate a query by prepending context when pronouns are detected.

        Args:
            query: The raw user query.
            history: Recent conversation messages for context.

        Returns:
            The original query if no reformulation is needed, or a contextualised version.
        """
        if not history:
            return query

        if not PRONOUN_PATTERN.search(query):
            return query

        last_user_topic = self._get_last_user_topic(history)
        if not last_user_topic:
            return query

        return f"Regarding {last_user_topic}: {query}"

    def _get_last_user_topic(self, history: list[Message]) -> str | None:
        """Extract the most recent user message content as the topic.

        Args:
            history: Conversation messages to search through.

        Returns:
            Truncated content of the last user message, or None if no user messages exist.
        """
        for msg in reversed(history):
            if msg.role == "user":
                return msg.content[:MAX_TOPIC_LENGTH]
        return None
