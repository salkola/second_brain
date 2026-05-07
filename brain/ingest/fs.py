from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def sorted_files_under_roots(roots: list[Path], suffixes: Iterable[str]) -> list[Path]:
    """Recursively collect files under roots whose extension (case-insensitive) is in suffixes."""
    allowed = {s.lower() for s in suffixes}
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in allowed:
                out.append(p)
    return sorted(out)
