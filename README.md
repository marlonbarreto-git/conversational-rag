# Conversational RAG

RAG with conversational memory and query reformulation for follow-up questions.

## Overview

Conversational RAG extends traditional retrieval-augmented generation by maintaining conversation history and automatically reformulating follow-up queries. When a user asks a vague follow-up like "What about its performance?", the system detects pronouns, resolves them against recent conversation context, and rewrites the query before retrieval. This enables natural multi-turn conversations over a document corpus.

## Architecture

```
User Query
    |
    v
+-------------------+     +---------------------+
| ConversationMemory|<--->| QueryReformulator    |
| (sliding window)  |     | (pronoun detection + |
+-------------------+     |  context injection)  |
                          +---------------------+
                                    |
                            reformulated query
                                    |
                                    v
                          +---------------------+
                          |     Retriever        |
                          | (sentence-transformers|
                          |  cosine similarity)  |
                          +---------------------+
                                    |
                                    v
                          +---------------------+
                          | ConversationalRAG    |
                          |   (pipeline)         |
                          +---------------------+
```

## Features

- Sliding-window conversation memory with configurable history size
- Pronoun-based query reformulation for follow-up questions
- Vector retrieval using sentence-transformers embeddings
- Cosine similarity search with top-k results
- Full pipeline orchestrating memory, reformulation, and retrieval
- Conversation history summarization

## Tech Stack

- Python 3.11+
- sentence-transformers (embedding model)
- NumPy (vector operations)
- Pydantic (data validation)

## Quick Start

```bash
git clone https://github.com/marlonbarreto-git/conversational-rag.git
cd conversational-rag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Project Structure

```
src/conversational_rag/
  __init__.py
  models.py          # Message and ConversationTurn dataclasses
  memory.py          # ConversationMemory with sliding window
  reformulator.py    # QueryReformulator with pronoun detection
  retriever.py       # Vector retriever with sentence-transformers
  pipeline.py        # ConversationalRAG pipeline orchestrator
tests/
  test_memory.py
  test_reformulator.py
  test_retriever.py
  test_pipeline.py
```

## Testing

```bash
pytest -v --cov=src/conversational_rag
```

47 tests covering memory management, query reformulation, vector retrieval, and end-to-end pipeline behavior.

## License

MIT
