from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from brain.types import RetrieverLike

COLLECTION_NAME = "brain"
# Chroma + sentence-transformers reject a single embed/add above ~5.4k documents
# (version-dependent).
_CHROMA_ADD_BATCH = 4096


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def get_vectorstore(
    persist_directory: Path,
    embedding_model_name: str,
) -> Chroma:
    persist_directory.mkdir(parents=True, exist_ok=True)
    emb = get_embeddings(embedding_model_name)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=str(persist_directory),
    )


def add_documents_batched(store: Chroma, docs: list[Document]) -> None:
    """Add documents in chunks so embedding/Chroma stay under max batch size."""
    if not docs:
        return
    for start in range(0, len(docs), _CHROMA_ADD_BATCH):
        store.add_documents(docs[start : start + _CHROMA_ADD_BATCH])


def delete_by_source(store: Chroma, source: str) -> None:
    """Remove chunks whose metadata.source matches (for re-index)."""
    try:
        data = store.get(where={"source": source})
    except Exception:
        return
    ids = data.get("ids") or []
    if ids:
        store.delete(ids=ids)


def as_retriever(store: Chroma, k: int) -> RetrieverLike:
    return store.as_retriever(search_kwargs={"k": k})
