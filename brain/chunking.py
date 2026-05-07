from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


def sanitize_doc_metadata(
    meta: dict[str, str | int | float | None],
) -> dict[str, str | int | float | bool]:
    """Chroma metadata must be primitives; drop None and stringify the rest."""
    out: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, int | float):
            out[k] = v
        else:
            out[k] = str(v)
    return out


@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_id: str
    metadata: dict[str, str | int | float | None]


def stable_doc_id(source: str, extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\0")
    h.update(extra.encode("utf-8"))
    return h.hexdigest()[:24]


def split_text(
    text: str,
    *,
    source: str,
    source_type: str,
    chunk_size: int,
    chunk_overlap: int,
    base_metadata: dict[str, str | int | float | None] | None = None,
) -> list[TextChunk]:
    base = dict(base_metadata or {})
    base.setdefault("source", source)
    base.setdefault("source_type", source_type)
    doc_id = stable_doc_id(source, str(base.get("page", "")))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    parts = splitter.split_text(text)
    out: list[TextChunk] = []
    for i, part in enumerate(parts):
        cid = f"{doc_id}:{i}"
        meta = {**base, "chunk_index": i}
        out.append(TextChunk(text=part, chunk_id=cid, metadata=meta))
    return out
