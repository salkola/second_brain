from __future__ import annotations

import logging
import plistlib
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from email import message_from_bytes
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from brain.app_config import AppConfig
from brain.chunking import sanitize_doc_metadata, split_text

log = logging.getLogger(__name__)

# Mail Envelope Index often stores dates as CFAbsoluteTime (seconds since 2001-01-01 UTC).
_APPLE_TO_UNIX = datetime(2001, 1, 1, tzinfo=UTC).timestamp()


def discover_envelope_indices(mail_library: Path) -> list[Path]:
    if not mail_library.is_dir():
        return []
    found: list[Path] = []
    for p in mail_library.rglob("Envelope Index"):
        if p.is_file():
            found.append(p)
    return sorted(set(found))


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=5.0)


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]


def _column_type(conn: sqlite3.Connection, table: str, col: str) -> str | None:
    for row in conn.execute(f"PRAGMA table_info({table})").fetchall():
        if row[1] == col:
            return str(row[2] or "").upper()
    return None


def _pick(cols: list[str], *candidates: str) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _format_envelope_date(raw: Any) -> str:
    """Human-readable date for Mail envelope timestamps (unix or Apple CFAbsoluteTime)."""
    if raw is None:
        return ""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return str(raw)
    # Large values are almost certainly unix seconds (Mail may also store unix).
    if v >= 1_000_000_000:
        ts_unix = v
    else:
        ts_unix = v + _APPLE_TO_UNIX
    try:
        return datetime.fromtimestamp(ts_unix, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    except OSError:
        return str(raw)


def _envelope_date_unix(raw: Any) -> float | None:
    """Unix timestamp for Mail envelope timestamps (unix or Apple CFAbsoluteTime)."""
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v >= 1_000_000_000:
        return v
    return v + _APPLE_TO_UNIX


def _strip_accidental_plist_xml_prefix(text: str) -> str:
    """Drop Apple plist preamble if it leaked into the body (bad .emlx length or snippet noise)."""
    t = text.lstrip()
    if not t.startswith("<?xml"):
        return text
    head = t[:1200].lower()
    if "plist-1.0.dtd" not in head and "<plist" not in head:
        return text
    idx = t.find("</plist>")
    if idx == -1:
        return text
    return t[idx + len("</plist>") :].lstrip("\n\r \t")


class _HTMLToPlainText(HTMLParser):
    """Minimal HTML → plain text for text/html bodies when text/plain is missing."""

    _BLOCK = frozenset({"br", "p", "div", "tr", "li", "h1", "h2", "h3", "table"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip = True
        elif tag in self._BLOCK:
            self._out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False
        elif tag in self._BLOCK:
            self._out.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._out.append(data)

    def plain(self) -> str:
        raw = "".join(self._out)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _as_bytes_or_str(payload: object) -> bytes | str | None:
    if isinstance(payload, bytes | str):
        return payload
    return None


def _html_to_plain(payload: bytes | str) -> str:
    if isinstance(payload, bytes):
        html = payload.decode("utf-8", errors="replace")
    else:
        html = payload
    parser = _HTMLToPlainText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return ""
    return parser.plain()


def _part_plain_payload(part: Message) -> str | None:
    try:
        raw = part.get_payload(decode=True)
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def parse_emlx(path: Path) -> tuple[dict[str, Any], bytes]:
    """Parse .emlx: first line is plist byte length, then plist, then raw RFC822."""
    data = path.read_bytes()
    if not data:
        return {}, b""
    try:
        first_nl = data.index(b"\n")
    except ValueError:
        return {}, b""
    try:
        plist_len = int(data[:first_nl].decode("ascii").strip())
    except ValueError:
        return {}, b""
    start = first_nl + 1
    plist_bytes = data[start : start + plist_len]
    rfc822 = data[start + plist_len :]
    try:
        plist = plistlib.loads(plist_bytes)
    except Exception:
        plist = {}
    if not isinstance(plist, dict):
        plist = {}
    return plist, rfc822


def rfc822_plain_body(rfc822: bytes) -> str:
    if not rfc822.strip():
        return ""
    try:
        msg = message_from_bytes(rfc822)
    except Exception:
        return ""
    plain_chunks: list[str] = []
    html_chunks: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if part.get_content_maintype() == "multipart":
                continue
            if ctype == "text/plain":
                t = _part_plain_payload(part)
                if t and t.strip():
                    plain_chunks.append(t.strip())
            elif ctype == "text/html":
                raw = _as_bytes_or_str(part.get_payload(decode=True))
                if raw:
                    ht = _html_to_plain(raw)
                    if ht:
                        html_chunks.append(ht)
        joined_plain = "\n\n".join(plain_chunks).strip()
        if joined_plain:
            return joined_plain
        joined_html = "\n\n".join(html_chunks).strip()
        return joined_html
    ctype = msg.get_content_type()
    if ctype == "text/plain":
        t = _part_plain_payload(msg)
        return (t or "").strip()
    if ctype == "text/html":
        raw = _as_bytes_or_str(msg.get_payload(decode=True))
        return _html_to_plain(raw) if raw else ""
    try:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace").strip()
        if isinstance(payload, str):
            return payload.strip()
    except Exception:
        pass
    return ""


def _find_emlx_for_rowid(mail_library: Path, rowid: int) -> Path | None:
    for name in (f"{rowid}.emlx", f"{rowid}.partial.emlx"):
        for p in mail_library.rglob(name):
            if p.is_file():
                return p
    return None


def _mailbox_url_looks_like_inbox(url_col_sql: str) -> str:
    """WHERE fragment matching Mail.app folder URLs for INBOX (case-insensitive)."""
    return f"(LOWER({url_col_sql}) LIKE '%/inbox' OR LOWER({url_col_sql}) LIKE '%/inbox/%')"


def _since_clause(
    date_col: str,
    cutoff_unix: float,
    cutoff_apple: float,
) -> tuple[str, list[float]]:
    """WHERE fragment matching either unix-style or Apple CFAbsoluteTime envelope dates."""
    sql = (
        f'((m."{date_col}" >= ? AND m."{date_col}" >= 1000000000) OR '
        f'(m."{date_col}" >= ? AND m."{date_col}" < 1000000000))'
    )
    return sql, [cutoff_unix, cutoff_apple]


def _messages_from_db(
    cfg: AppConfig,
    mail_root: Path,
    db_path: Path,
) -> list[Document]:
    try:
        conn = _connect_readonly(db_path)
    except sqlite3.Error:
        return []
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            ).fetchall()
        ]
        if "messages" not in tables:
            return []
        cols = _table_columns(conn, "messages")
        subj_col = _pick(cols, "subject", "normalized_subject")
        sender_col = _pick(cols, "sender", "sender_address")
        date_col = _pick(cols, "date_sent", "date_received", "received_date")
        snippet_col = _pick(cols, "snippet", "preview")
        summary_fk = _pick(cols, "summary")

        subjects_tbl = "subjects" in tables
        addresses_tbl = "addresses" in tables
        summaries_tbl = "summaries" in tables

        subj_integer_fk = (
            subjects_tbl
            and subj_col is not None
            and (_column_type(conn, "messages", subj_col) or "") == "INTEGER"
        )
        sender_integer_fk = (
            addresses_tbl
            and sender_col is not None
            and (_column_type(conn, "messages", sender_col) or "") == "INTEGER"
        )
        use_joins = bool(subj_integer_fk and sender_integer_fk)
        joins: list[str] = []
        sel: list[str]

        if (
            use_joins
            and subj_col
            and sender_col
            and (subj_txt := _pick(_table_columns(conn, "subjects"), "subject", "string"))
            and (addr_txt := _pick(_table_columns(conn, "addresses"), "address", "email", "string"))
        ):
            sel = [
                "m.ROWID AS rowid",
                f'subs."{subj_txt}" AS subj',
                f'addr."{addr_txt}" AS snd',
            ]
            joins.append(f'LEFT JOIN subjects subs ON m."{subj_col}" = subs.ROWID')
            joins.append(f'LEFT JOIN addresses addr ON m."{sender_col}" = addr.ROWID')
        else:
            sel = ["m.ROWID AS rowid"]
            if subj_col:
                sel.append(f'm."{subj_col}" AS subj')
            else:
                sel.append("NULL AS subj")
            if sender_col:
                sel.append(f'm."{sender_col}" AS snd')
            else:
                sel.append("NULL AS snd")

        if date_col:
            sel.append(f'm."{date_col}" AS dts')
        else:
            sel.append("NULL AS dts")

        snip_via_summary = False
        if summaries_tbl and summary_fk:
            summ_cols = _table_columns(conn, "summaries")
            summ_txt = _pick(summ_cols, "snippet", "summary", "string")
            if summ_txt:
                sel.append(f'summ."{summ_txt}" AS snip')
                joins.append(f'LEFT JOIN summaries summ ON m."{summary_fk}" = summ.ROWID')
                snip_via_summary = True
        if not snip_via_summary:
            if snippet_col:
                sel.append(f'm."{snippet_col}" AS snip')
            else:
                sel.append("NULL AS snip")

        where_clauses: list[str] = []
        params: list[Any] = []
        if cfg.mail_since_days is not None and date_col:
            cutoff = datetime.now(UTC) - timedelta(days=cfg.mail_since_days)
            cu = cutoff.timestamp()
            wc, pr = _since_clause(date_col, cu, cu - _APPLE_TO_UNIX)
            where_clauses.append(wc)
            params.extend(pr)

        if cfg.mail_only_inbox:
            mb_col = _pick(cols, "mailbox", "mailbox_id")
            if "mailboxes" in tables and mb_col is not None:
                mb_cols = _table_columns(conn, "mailboxes")
                url_col = _pick(mb_cols, "url", "URL")
                if url_col:
                    joins.append(f'INNER JOIN mailboxes mb ON m."{mb_col}" = mb.ROWID')
                    url_sql = f'mb."{url_col}"'
                    where_clauses.append(_mailbox_url_looks_like_inbox(url_sql))
                else:
                    log.warning(
                        "mail_only_inbox is set but mailboxes table has no url column in %s; "
                        "indexing all messages from this DB.",
                        db_path.name,
                    )
            else:
                log.warning(
                    "mail_only_inbox is set but messages.mailbox → mailboxes link not found in %s; "
                    "indexing all messages from this DB.",
                    db_path.name,
                )

        join_sql = " " + " ".join(joins) if joins else ""
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        # Prefer real envelope timestamps so mail_message_limit keeps the newest mail, not ROWID
        # order (ROWID does not match received date after reindexing / imports).
        if date_col:
            dq = f'm."{date_col}"'
            order_sql = f"ORDER BY ({dq} IS NULL) ASC, {dq} DESC, m.ROWID DESC"
        else:
            order_sql = "ORDER BY m.ROWID DESC"
        sql = f"SELECT {', '.join(sel)} FROM messages m{join_sql} {where_sql} {order_sql} LIMIT ?"
        params.append(cfg.mail_message_limit)

        cur = conn.execute(sql, params)
        colnames = [c[0] for c in cur.description]
        rows = cur.fetchall()
    finally:
        conn.close()

    all_docs: list[Document] = []
    for row in rows:
        mapping = dict(zip(colnames, row, strict=False))
        rid = int(mapping.get("rowid") or 0)
        subject = str(mapping.get("subj") or "").strip()
        sender = str(mapping.get("snd") or "").strip()
        date_raw = mapping.get("dts")
        snippet = str(mapping.get("snip") or "")

        body_text = snippet
        emlx = _find_emlx_for_rowid(mail_root, rid)
        if emlx and emlx.is_file():
            _plist, rfc = parse_emlx(emlx)
            extracted = rfc822_plain_body(rfc)
            if extracted.strip():
                body_text = extracted

        body_text = _strip_accidental_plist_xml_prefix(body_text)

        date_line = _format_envelope_date(date_raw)
        date_unix = _envelope_date_unix(date_raw)
        header = f"Subject: {subject}\nFrom: {sender}\nDate: {date_line}\n"
        full = header + "\n" + body_text
        if not full.strip():
            continue
        source = f"mail:{db_path.name}:{rid}"
        chunks = split_text(
            full,
            source=source,
            source_type="email",
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            base_metadata={
                "source": source,
                "source_type": "email",
                "message_rowid": rid,
                "subject": subject[:500],
                "sender": sender[:500],
                "date_unix": date_unix,
                "mailbox_db": str(db_path.resolve()),
            },
        )
        for c in chunks:
            all_docs.append(
                Document(page_content=c.text, metadata=sanitize_doc_metadata(c.metadata)),
            )
    return all_docs


def iter_mail_documents(cfg: AppConfig, envelope_db: Path | None = None) -> list[Document]:
    mail_root = cfg.mail_library
    indices = [envelope_db] if envelope_db else discover_envelope_indices(mail_root)
    out: list[Document] = []
    for db_path in indices:
        out.extend(_messages_from_db(cfg, mail_root, db_path))
    return out
