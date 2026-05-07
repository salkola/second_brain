from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from brain.app_config import AppConfig
from brain.chunking import sanitize_doc_metadata, split_text
from brain.ingest.fs import sorted_files_under_roots

_PDF_SUFFIX = frozenset({".pdf"})


def iter_pdf_files(roots: list[Path]) -> list[Path]:
    return sorted_files_under_roots(roots, _PDF_SUFFIX)


def _extract_pdf_text(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append((i + 1, t))
    return pages


def iter_pdf_documents(cfg: AppConfig, path: Path) -> list[Document]:
    source = str(path.resolve())
    docs: list[Document] = []
    for page_num, text in _extract_pdf_text(path):
        if not text.strip():
            continue
        chunks = split_text(
            text,
            source=source,
            source_type="pdf",
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            base_metadata={
                "source": source,
                "source_type": "pdf",
                "page": page_num,
            },
        )
        for c in chunks:
            docs.append(
                Document(page_content=c.text, metadata=sanitize_doc_metadata(c.metadata)),
            )
    return docs
