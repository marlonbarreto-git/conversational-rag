from conversational_rag.models import Message


class ConversationMemory:
    def __init__(self, max_history: int = 50) -> None:
        self.max_history = max_history
        self._messages: list[Message] = []

    def add_message(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))
        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history :]

    def get_history(self, max_messages: int | None = None) -> list[Message]:
        if max_messages is None:
            return list(self._messages)
        return list(self._messages[-max_messages:])

    def get_context_window(self, n: int = 10) -> list[Message]:
        return list(self._messages[-n:])

    def clear(self) -> None:
        self._messages.clear()

    def summarize_history(self) -> str:
        if not self._messages:
            return ""
        lines = []
        for msg in self._messages:
            role_label = msg.role.capitalize()
            lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)
