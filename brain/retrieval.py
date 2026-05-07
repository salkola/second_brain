from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.documents import Document

from brain.app_config import AppConfig

log = logging.getLogger(__name__)


class VectorStoreLike(Protocol):
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]: ...

    def get(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RetrievedDoc:
    doc: Document
    score: float


_WORD_RE = re.compile(r"[A-Za-z0-9_@.-]{2,}")


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _dedupe_by_chunk_id(docs: Iterable[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        cid = str(d.metadata.get("chunk_id") or "")
        key = cid or (str(d.metadata.get("source") or ""), str(d.metadata.get("chunk_index") or ""))
        k = "|".join(key) if isinstance(key, tuple) else key
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out


def bm25_like_keyword_rank(
    docs: list[Document],
    query: str,
    *,
    top_k: int,
) -> list[RetrievedDoc]:
    """Cheap lexical ranker over a provided candidate set (not a global BM25 index)."""
    q = _tokenize(query)
    if not q:
        return []
    qset = set(q)
    scored: list[RetrievedDoc] = []
    for d in docs:
        t = _tokenize(d.page_content)
        if not t:
            continue
        # Simple overlap + term frequency proxy.
        overlap = sum(1 for tok in t if tok in qset)
        uniq = len(set(t) & qset)
        score = float(uniq) * 2.0 + float(overlap) * 0.05
        if score > 0:
            scored.append(RetrievedDoc(doc=d, score=score))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[: max(0, int(top_k))]


def _load_sample_docs_for_keyword_search(
    store: VectorStoreLike,
    *,
    filt: dict[str, Any] | None,
    max_chunks: int,
) -> list[Document]:
    """Sample up to max_chunks documents from the vector store for lexical ranking."""
    max_chunks = max(0, int(max_chunks))
    if max_chunks == 0:
        return []
    take = min(8000, max_chunks)
    offset = 0
    out: list[Document] = []
    where = filt or {}
    while offset < max_chunks:
        data = store.get(
            where=where,
            limit=take,
            offset=offset,
            include=["documents", "metadatas"],
        )
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        if not docs or not metas:
            break
        for text, meta in zip(docs, metas, strict=False):
            if not text or not meta:
                continue
            out.append(Document(page_content=str(text), metadata=dict(meta)))
        n = len(docs)
        offset += n
        if n < take:
            break
        take = min(8000, max_chunks - offset)
    return out


def maybe_rerank(
    docs: list[Document],
    query: str,
    *,
    cfg: AppConfig,
    ctx: Any,
) -> list[Document]:
    """Rerank with a cross-encoder if available; otherwise return docs unchanged."""
    if not cfg.rerank or not docs:
        return docs

    # Lazy cache in ctx (ToolContext) without making it a hard dependency.
    model = getattr(ctx, "_reranker", None)
    if model is None:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            log.debug("CrossEncoder import failed, skipping rerank: %r", exc)
            ctx._reranker = False
            return docs
        try:
            model = CrossEncoder(cfg.rerank_model_name)
        except Exception as exc:
            log.warning("Could not load reranker model %s: %r", cfg.rerank_model_name, exc)
            ctx._reranker = False
            return docs
        ctx._reranker = model
    if model is False:
        return docs

    top_k = max(1, int(cfg.rerank_top_k))
    cand = docs[: max(top_k, len(docs))]
    pairs = [(query, d.page_content) for d in cand]
    try:
        scores = model.predict(pairs)
    except Exception as exc:
        log.warning("Reranker failed, skipping: %r", exc)
        return docs
    scored = list(zip(cand, scores, strict=False))
    scored.sort(key=lambda x: float(x[1]), reverse=True)
    return [d for d, _s in scored]


def expand_parent_context(
    store: VectorStoreLike,
    docs: list[Document],
    *,
    cfg: AppConfig,
) -> list[Document]:
    """Expand each doc with neighbor chunks from the same source (parent/child context)."""
    if not cfg.parent_context or not docs:
        return docs
    window = max(0, int(cfg.parent_context_window))
    if window == 0:
        return docs
    limit_per_source = max(50, int(cfg.parent_context_max_chunks_per_source))

    by_source: dict[str, list[Document]] = {}
    for d in docs:
        src = str(d.metadata.get("source") or "")
        if src:
            by_source.setdefault(src, [])

    # Fetch chunks per source once.
    source_chunks: dict[str, list[Document]] = {}
    for src in by_source:
        try:
            data = store.get(
                where={"source": src},
                limit=limit_per_source,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            log.debug("parent context fetch failed for %s: %r", src, exc)
            continue
        texts = data.get("documents") or []
        metas = data.get("metadatas") or []
        chunks: list[Document] = []
        for text, meta in zip(texts, metas, strict=False):
            if not text or not meta:
                continue
            chunks.append(Document(page_content=str(text), metadata=dict(meta)))
        chunks.sort(key=lambda x: int(x.metadata.get("chunk_index") or 0))
        source_chunks[src] = chunks

    expanded: list[Document] = []
    for d in docs:
        src = str(d.metadata.get("source") or "")
        idx = d.metadata.get("chunk_index")
        if not src or idx is None or src not in source_chunks:
            expanded.append(d)
            continue
        try:
            i = int(idx)
        except (TypeError, ValueError):
            expanded.append(d)
            continue
        chunks = source_chunks[src]
        if not chunks:
            expanded.append(d)
            continue
        start = max(0, i - window)
        end = i + window
        picked = [c for c in chunks if start <= int(c.metadata.get("chunk_index") or 0) <= end]
        if not picked:
            expanded.append(d)
            continue
        joined = "\n\n---\n\n".join(c.page_content for c in picked)
        expanded.append(Document(page_content=joined, metadata=d.metadata))
    return expanded


def hybrid_retrieve(
    store: VectorStoreLike,
    query: str,
    *,
    cfg: AppConfig,
    ctx: Any,
    filt: dict[str, Any] | None = None,
) -> list[Document]:
    """Vector + sampled keyword retrieval + optional rerank + parent context expansion."""
    # Vector candidates
    vec_k = max(1, int(cfg.retrieve_k))
    vec_k = min(vec_k, 400)  # keep things bounded; rerank/parent-context operate on top results
    vec = store.similarity_search(query, k=vec_k, filter=filt)

    if not cfg.hybrid_retrieval:
        docs = _dedupe_by_chunk_id(vec)
        docs = maybe_rerank(docs, query, cfg=cfg, ctx=ctx)
        return expand_parent_context(store, docs[: max(1, int(cfg.rerank_top_k))], cfg=cfg)

    # Keyword candidates (sampled)
    try:
        sample = _load_sample_docs_for_keyword_search(
            store,
            filt=filt,
            max_chunks=int(cfg.hybrid_bm25_max_chunks),
        )
    except Exception as exc:
        log.debug("keyword sample failed: %r", exc)
        sample = []
    kw_ranked = bm25_like_keyword_rank(sample, query, top_k=int(cfg.hybrid_bm25_top_k))
    kw = [r.doc for r in kw_ranked]

    merged = _dedupe_by_chunk_id([*vec, *kw])
    merged = maybe_rerank(merged, query, cfg=cfg, ctx=ctx)
    top = merged[: max(1, int(cfg.rerank_top_k))]
    return expand_parent_context(store, top, cfg=cfg)
