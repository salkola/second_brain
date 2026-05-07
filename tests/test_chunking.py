from __future__ import annotations

from brain.chunking import split_text, stable_doc_id


def test_stable_doc_id_deterministic() -> None:
    a = stable_doc_id("/tmp/foo.md", "1")
    b = stable_doc_id("/tmp/foo.md", "1")
    assert a == b
    assert stable_doc_id("/tmp/foo.md", "2") != a


def test_split_text_metadata() -> None:
    text = "para1\n\n" + ("word " * 500)
    chunks = split_text(
        text,
        source="/x.md",
        source_type="note",
        chunk_size=200,
        chunk_overlap=20,
        base_metadata={"source": "/x.md"},
    )
    assert len(chunks) >= 2
    assert all(c.metadata.get("source") == "/x.md" for c in chunks)
    assert chunks[0].chunk_id != chunks[1].chunk_id
