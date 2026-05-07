from __future__ import annotations

from pathlib import Path

from brain.index_state import kv_get, kv_set, needs_reindex


def test_kv_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    assert kv_get(db, "k") is None
    kv_set(db, "k", "v")
    assert kv_get(db, "k") == "v"


def test_needs_reindex_new_file(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    assert needs_reindex(db, f) is True
