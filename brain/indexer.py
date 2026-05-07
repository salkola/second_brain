from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from brain.app_config import AppConfig
from brain.index_state import clear_all, kv_get, kv_set, mark_indexed, needs_reindex
from brain.ingest.apple_mail import iter_mail_documents
from brain.ingest.notes import iter_note_documents, iter_note_files
from brain.ingest.pdf import iter_pdf_documents, iter_pdf_files
from brain.ingest.screenshots import (
    iter_screenshot_documents,
    iter_screenshot_files,
    tesseract_available,
)
from brain.mail_fingerprint import envelope_index_fingerprint
from brain.vectorstore import add_documents_batched, delete_by_source, get_vectorstore

log = logging.getLogger(__name__)

_MAIL_FP_KEY = "mail_envelope_fingerprint"


def _index_file_docs(
    store: Chroma,
    cfg: AppConfig,
    path: Path,
    builder: Callable[[AppConfig, Path], list[Document]],
    *,
    full: bool,
) -> int:
    source = str(path.resolve())
    if not full and not needs_reindex(cfg.index_state_db, path):
        return 0
    delete_by_source(store, source)
    docs = builder(cfg, path)
    if docs:
        add_documents_batched(store, docs)
    # Only record success when we stored chunks; otherwise the next run retries (e.g. OCR was empty
    # until Tesseract was installed).
    if path.is_file() and docs:
        mark_indexed(cfg.index_state_db, path)
    return len(docs)


def run_index(cfg: AppConfig, *, full: bool = False) -> dict[str, int]:
    if full:
        clear_all(cfg.index_state_db)

    store = get_vectorstore(cfg.chroma_persist_directory, cfg.embedding_model_name)
    counts: dict[str, int] = {
        "notes": 0,
        "pdfs": 0,
        "screenshots": 0,
        "screenshot_files": 0,
        "email_chunks": 0,
        "files_unchanged": 0,
    }

    for p in iter_note_files(cfg.notes_roots):
        if not full and not needs_reindex(cfg.index_state_db, p):
            counts["files_unchanged"] += 1
            continue
        counts["notes"] += _index_file_docs(store, cfg, p, iter_note_documents, full=full)

    for p in iter_pdf_files(cfg.pdf_roots):
        if not full and not needs_reindex(cfg.index_state_db, p):
            counts["files_unchanged"] += 1
            continue
        counts["pdfs"] += _index_file_docs(store, cfg, p, iter_pdf_documents, full=full)

    shot_paths = iter_screenshot_files(cfg.screenshot_roots)
    counts["screenshot_files"] = len(shot_paths)
    if shot_paths and not tesseract_available():
        log.warning(
            "Screenshot folder(s) have images but Tesseract is not on PATH — "
            "no screenshot text/chunks will be indexed (e.g. brew install tesseract).",
        )
    for p in shot_paths:
        if not full and not needs_reindex(cfg.index_state_db, p):
            counts["files_unchanged"] += 1
            continue
        counts["screenshots"] += _index_file_docs(
            store,
            cfg,
            p,
            iter_screenshot_documents,
            full=full,
        )

    fp = envelope_index_fingerprint(cfg.mail_library)
    prev = kv_get(cfg.index_state_db, _MAIL_FP_KEY)
    if full or prev != fp:
        mail_docs = iter_mail_documents(cfg)
        if mail_docs:
            try:
                data = store.get(where={"source_type": "email"}, include=["metadatas"])
                ids = data.get("ids") or []
                if ids:
                    store.delete(ids=ids)
            except Exception as e:
                log.warning("Could not clear old mail vectors: %s", e)
            add_documents_batched(store, mail_docs)
        counts["email_chunks"] = len(mail_docs)
        kv_set(cfg.index_state_db, _MAIL_FP_KEY, fp)
    else:
        counts["email_chunks"] = 0

    return counts
