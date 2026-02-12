import time

from conversational_rag.memory import ConversationMemory
from conversational_rag.models import Message


class TestConversationMemoryInit:
    def test_initializes_with_empty_history(self):
        memory = ConversationMemory()
        assert memory.get_history() == []

    def test_initializes_with_default_max_history(self):
        memory = ConversationMemory()
        assert memory.max_history == 50

    def test_initializes_with_custom_max_history(self):
        memory = ConversationMemory(max_history=10)
        assert memory.max_history == 10


class TestAddMessage:
    def test_stores_single_message(self):
        memory = ConversationMemory()
        memory.add_message("user", "Hello")
        history = memory.get_history()
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == "Hello"

    def test_stores_multiple_messages(self):
        memory = ConversationMemory()
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there")
        history = memory.get_history()
        assert len(history) == 2

    def test_message_has_timestamp(self):
        memory = ConversationMemory()
        before = time.time()
        memory.add_message("user", "Hello")
        after = time.time()
        msg = memory.get_history()[0]
        assert before - 0.01 <= msg.timestamp <= after + 0.01


class TestGetHistory:
    def test_returns_messages_in_order(self):
        memory = ConversationMemory()
        memory.add_message("user", "First")
        memory.add_message("assistant", "Second")
        memory.add_message("user", "Third")
        history = memory.get_history()
        assert [m.content for m in history] == ["First", "Second", "Third"]

    def test_respects_max_messages_parameter(self):
        memory = ConversationMemory()
        memory.add_message("user", "First")
        memory.add_message("assistant", "Second")
        memory.add_message("user", "Third")
        history = memory.get_history(max_messages=2)
        assert len(history) == 2
        assert history[0].content == "Second"
        assert history[1].content == "Third"

    def test_max_messages_none_returns_all(self):
        memory = ConversationMemory()
        for i in range(5):
            memory.add_message("user", f"Message {i}")
        assert len(memory.get_history(max_messages=None)) == 5

    def test_max_messages_larger_than_history_returns_all(self):
        memory = ConversationMemory()
        memory.add_message("user", "Only one")
        assert len(memory.get_history(max_messages=100)) == 1


class TestGetContextWindow:
    def test_returns_recent_n_messages(self):
        memory = ConversationMemory()
        for i in range(20):
            memory.add_message("user", f"Message {i}")
        window = memory.get_context_window(n=5)
        assert len(window) == 5
        assert window[0].content == "Message 15"
        assert window[-1].content == "Message 19"

    def test_default_n_is_10(self):
        memory = ConversationMemory()
        for i in range(20):
            memory.add_message("user", f"Message {i}")
        window = memory.get_context_window()
        assert len(window) == 10

    def test_returns_all_if_fewer_than_n(self):
        memory = ConversationMemory()
        memory.add_message("user", "Only one")
        window = memory.get_context_window(n=10)
        assert len(window) == 1


class TestClear:
    def test_empties_history(self):
        memory = ConversationMemory()
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi")
        memory.clear()
        assert memory.get_history() == []

    def test_can_add_messages_after_clear(self):
        memory = ConversationMemory()
        memory.add_message("user", "Before clear")
        memory.clear()
        memory.add_message("user", "After clear")
        history = memory.get_history()
        assert len(history) == 1
        assert history[0].content == "After clear"


class TestSummarizeHistory:
    def test_produces_text_summary(self):
        memory = ConversationMemory()
        memory.add_message("user", "What is Python?")
        memory.add_message("assistant", "A programming language.")
        summary = memory.summarize_history()
        assert "User: What is Python?" in summary
        assert "Assistant: A programming language." in summary

    def test_empty_history_returns_empty_string(self):
        memory = ConversationMemory()
        assert memory.summarize_history() == ""

    def test_summary_format_newline_separated(self):
        memory = ConversationMemory()
        memory.add_message("user", "Q1")
        memory.add_message("assistant", "A1")
        memory.add_message("user", "Q2")
        summary = memory.summarize_history()
        assert summary == "User: Q1\nAssistant: A1\nUser: Q2"
