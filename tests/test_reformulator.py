from conversational_rag.models import Message
from conversational_rag.reformulator import QueryReformulator


class TestQueryReformulatorInit:
    def test_creates_instance(self):
        reformulator = QueryReformulator()
        assert reformulator is not None

    def test_accepts_custom_model_name(self):
        reformulator = QueryReformulator(model_name="all-MiniLM-L6-v2")
        assert reformulator.model_name == "all-MiniLM-L6-v2"


class TestReformulateNoHistory:
    def test_returns_original_query_with_empty_history(self):
        reformulator = QueryReformulator()
        result = reformulator.reformulate("What is Python?", [])
        assert result == "What is Python?"

    def test_returns_str_type(self):
        reformulator = QueryReformulator()
        result = reformulator.reformulate("Hello", [])
        assert isinstance(result, str)


class TestReformulateWithHistory:
    def test_expands_pronoun_it(self):
        reformulator = QueryReformulator()
        history = [
            Message(role="user", content="Tell me about Python"),
            Message(role="assistant", content="Python is a programming language."),
        ]
        result = reformulator.reformulate("What can it do?", history)
        assert "Tell me about Python" in result
        assert "What can it do?" in result

    def test_expands_pronoun_they(self):
        reformulator = QueryReformulator()
        history = [
            Message(role="user", content="List machine learning frameworks"),
            Message(role="assistant", content="TensorFlow, PyTorch, etc."),
        ]
        result = reformulator.reformulate("How do they compare?", history)
        assert "List machine learning frameworks" in result

    def test_expands_pronoun_this(self):
        reformulator = QueryReformulator()
        history = [
            Message(role="user", content="Show me the error traceback"),
            Message(role="assistant", content="Here is the traceback..."),
        ]
        result = reformulator.reformulate("How do I fix this?", history)
        assert "Show me the error traceback" in result

    def test_explicit_query_no_pronouns_returns_similar(self):
        reformulator = QueryReformulator()
        history = [
            Message(role="user", content="Tell me about Python"),
            Message(role="assistant", content="Python is great."),
        ]
        result = reformulator.reformulate("What is JavaScript?", history)
        assert result == "What is JavaScript?"

    def test_uses_last_user_message_as_topic(self):
        reformulator = QueryReformulator()
        history = [
            Message(role="user", content="First topic"),
            Message(role="assistant", content="Response 1"),
            Message(role="user", content="Second topic about databases"),
            Message(role="assistant", content="Response 2"),
        ]
        result = reformulator.reformulate("Tell me more about it", history)
        assert "Second topic about databases" in result
        assert "First topic" not in result

    def test_truncates_long_topic_to_100_chars(self):
        reformulator = QueryReformulator()
        long_content = "A" * 200
        history = [
            Message(role="user", content=long_content),
            Message(role="assistant", content="OK"),
        ]
        result = reformulator.reformulate("What about it?", history)
        # The topic portion should be at most 100 chars from the original
        assert long_content[:100] in result
        assert long_content[:101] not in result
