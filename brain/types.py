"""Shared typing helpers (structural types at I/O boundaries)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.documents import Document


class RetrieverLike(Protocol):
    """Minimal protocol for LangChain-style retrievers used by tools."""

    def invoke(self, query: str, /) -> Sequence[Document]:
        """Return ranked documents for the query."""
        ...
