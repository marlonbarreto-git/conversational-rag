# conversational-rag

RAG with conversational memory and query reformulation. Maintains context across follow-up questions by detecting pronouns and expanding queries with conversation history.

## Features

- **Conversation memory**: Sliding window of recent messages with configurable history size
- **Query reformulation**: Detects pronouns (it, they, this, etc.) and expands with context from history
- **Semantic retrieval**: Sentence-transformer embeddings for document search
- **Full pipeline**: Single `ConversationalRAG` class combining memory + reformulation + retrieval

## Architecture

```
conversational_rag/
├── models.py         # Message, ConversationTurn dataclasses
├── memory.py         # ConversationMemory with sliding window
├── reformulator.py   # Pronoun-aware query expansion
├── retriever.py      # Semantic vector retriever
└── pipeline.py       # ConversationalRAG combining all components
```

## Quick Start

```python
from conversational_rag.pipeline import ConversationalRAG

rag = ConversationalRAG()
rag.index([
    "Python is a programming language for data science",
    "TensorFlow is a deep learning framework by Google",
    "PyTorch is developed by Meta for neural networks",
])

# First query
turn1 = rag.query("What is TensorFlow?")
print(turn1.response)

# Follow-up with pronoun — automatically reformulated
turn2 = rag.query("Who created it?")
print(f"Reformulated: {turn2.reformulated_query}")
print(turn2.response)
```

## Development

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest tests/ -v
```

## Roadmap

- **v2**: Sliding window + summary memory hybrid, LLM-powered query rewriting
- **v3**: Multi-session memory (users), conversation branching, export

## License

MIT
