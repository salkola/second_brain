# Shared defaults (shipped with the package). Override in ./config.py via CONFIG dict.

from typing import Any

CONFIG: dict[str, Any] = {
    "ollama_base_url": "http://127.0.0.1:11434",
    "ollama_chat_model": "llama3.2",
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "chroma_persist_directory": "~/.brain/chroma",
    "index_state_db": "~/.brain/index_state.sqlite",
    "notes_roots": [],
    "pdf_roots": [],
    "screenshot_roots": [],
    "mail_library": "~/Library/Mail",
    "mail_message_limit": 10000,
    "mail_since_days": None,
    # Only ingest messages whose mailbox URL looks like an account Inbox (Envelope Index
    # must expose messages.mailbox → mailboxes.url). If unavailable, ingest falls back to all
    # mail with a warning.
    "mail_only_inbox": True,
    "chunk_size": 1200,
    "chunk_overlap": 200,
    "retrieve_k": 10000,
    # Retrieval quality knobs (accuracy / fewer hallucinations).
    "hybrid_retrieval": True,
    "hybrid_bm25_top_k": 30,
    "hybrid_bm25_max_chunks": 12_000,
    "rerank": True,
    "rerank_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rerank_top_k": 12,
    "parent_context": True,
    "parent_context_window": 1,
    "parent_context_max_chunks_per_source": 750,
    "todos_path": "~/.brain/todos.json",
    "notes_write_allowlist": [],
    "open_calendar_enabled": False,
    "calendar_url": None,
    "log_level": "INFO",
}
