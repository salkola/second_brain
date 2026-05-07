from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from brain.app_config import AppConfig
from brain.chunking import sanitize_doc_metadata, split_text
from brain.ingest.fs import sorted_files_under_roots

_NOTE_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".org"}


def iter_note_files(roots: list[Path]) -> list[Path]:
    return sorted_files_under_roots(roots, _NOTE_SUFFIXES)


def iter_note_documents(cfg: AppConfig, path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source = str(path.resolve())
    chunks = split_text(
        text,
        source=source,
        source_type="note",
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        base_metadata={"source": source, "source_type": "note"},
    )
    return [
        Document(page_content=c.text, metadata=sanitize_doc_metadata(c.metadata)) for c in chunks
    ]
