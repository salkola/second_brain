from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


def _ensure_kv(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kv_store (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS indexed_files (
            path TEXT PRIMARY KEY,
            mtime_ns INTEGER NOT NULL,
            size INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    conn.commit()
    return conn


def file_fingerprint(path: Path) -> tuple[int, int, str]:
    st = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return int(st.st_mtime_ns), st.st_size, h.hexdigest()


def needs_reindex(state_db: Path, path: Path) -> bool:
    if not path.is_file():
        return False
    mtime_ns, size, chash = file_fingerprint(path)
    conn = _connect(state_db)
    try:
        row = conn.execute(
            "SELECT mtime_ns, size, content_hash FROM indexed_files WHERE path = ?",
            (str(path.resolve()),),
        ).fetchone()
        if row is None:
            return True
        rm, rs, rh = row
        return (rm, rs, rh) != (mtime_ns, size, chash)
    finally:
        conn.close()


def mark_indexed(state_db: Path, path: Path) -> None:
    mtime_ns, size, chash = file_fingerprint(path)
    conn = _connect(state_db)
    try:
        conn.execute(
            """
            INSERT INTO indexed_files (path, mtime_ns, size, content_hash)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                mtime_ns = excluded.mtime_ns,
                size = excluded.size,
                content_hash = excluded.content_hash,
                updated_at = excluded.updated_at
            """,
            (str(path.resolve()), mtime_ns, size, chash),
        )
        conn.commit()
    finally:
        conn.close()


def clear_all(state_db: Path) -> None:
    conn = _connect(state_db)
    try:
        conn.execute("DELETE FROM indexed_files")
        _ensure_kv(conn)
        conn.execute("DELETE FROM kv_store")
        conn.commit()
    finally:
        conn.close()


def kv_get(state_db: Path, key: str) -> str | None:
    conn = _connect(state_db)
    try:
        _ensure_kv(conn)
        row = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def kv_set(state_db: Path, key: str, value: str) -> None:
    conn = _connect(state_db)
    try:
        _ensure_kv(conn)
        conn.execute(
            """
            INSERT INTO kv_store (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()
