from __future__ import annotations

import hashlib
from pathlib import Path

from brain.ingest.apple_mail import discover_envelope_indices


def envelope_index_fingerprint(mail_library: Path) -> str:
    paths = discover_envelope_indices(mail_library)
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.resolve()).encode("utf-8"))
        try:
            st = p.stat()
            h.update(str(st.st_mtime_ns).encode("ascii"))
            h.update(str(st.st_size).encode("ascii"))
        except OSError:
            h.update(b"0")
    return h.hexdigest()
