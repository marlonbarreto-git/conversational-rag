from conversational_rag.models import Message

DEFAULT_MAX_HISTORY = 50
DEFAULT_CONTEXT_WINDOW = 10


class ConversationMemory:
    """Stores and manages conversation message history with a configurable size limit."""

    def __init__(self, max_history: int = DEFAULT_MAX_HISTORY) -> None:
        self.max_history = max_history
        self._messages: list[Message] = []

    def add_message(self, role: str, content: str) -> None:
        """Append a message and trim history if it exceeds the maximum.

        Args:
            role: The message role, either "user" or "assistant".
            content: The text content of the message.
        """
        self._messages.append(Message(role=role, content=content))
        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history :]

    def get_history(self, max_messages: int | None = None) -> list[Message]:
        """Return conversation history, optionally limited to the last N messages.

        Args:
            max_messages: Maximum number of recent messages to return. Returns all if None.

        Returns:
            List of messages in chronological order.
        """
        if max_messages is None:
            return list(self._messages)
        return list(self._messages[-max_messages:])

    def get_context_window(self, n: int = DEFAULT_CONTEXT_WINDOW) -> list[Message]:
        """Return the last N messages as context for query reformulation.

        Args:
            n: Number of recent messages to include.

        Returns:
            List of the most recent messages.
        """
        return list(self._messages[-n:])

    def clear(self) -> None:
        """Remove all messages from history."""
        self._messages.clear()

    def summarize_history(self) -> str:
        """Produce a plain-text summary of the conversation history.

        Returns:
            Newline-separated string of "Role: content" lines, or empty string if no history.
        """
        if not self._messages:
            return ""
        lines = []
        for msg in self._messages:
            role_label = msg.role.capitalize()
            lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)
