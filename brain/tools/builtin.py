from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from brain.app_config import AppConfig
from brain.retrieval import hybrid_retrieve
from brain.types import RetrieverLike


@dataclass
class ToolContext:
    """Mutable context shared across tools (e.g. last retrieval citations)."""

    last_retrieval: str = ""
    last_sources: list[str] = field(default_factory=list)


class _SimilaritySearchStore(Protocol):
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]: ...

    def get(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


def _under_allowlist(path: Path, roots: list[Path]) -> bool:
    rp = path.resolve()
    for root in roots:
        try:
            rp.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _format_retrieval(ctx: ToolContext, docs: Sequence[Document]) -> str:
    lines: list[str] = []
    ctx.last_sources.clear()
    any_mail = False
    for d in docs:
        src = str(d.metadata.get("source", ""))
        ctx.last_sources.append(src)
        st = str(d.metadata.get("source_type") or "")
        if st == "email":
            any_mail = True
            lines.append(
                f"Mail excerpt (citation id for /sources only: {src})\n{d.page_content}",
            )
        else:
            lines.append(f"Source: {src}\n{d.page_content}")
    body = "\n\n---\n\n".join(lines) if lines else "(no matching chunks)"
    if any_mail:
        legend = (
            "The following includes Apple Mail excerpts. Each block is human-readable email "
            "(Subject / From / Date, then body or preview). Ignore the opaque citation id prefix "
            "when summarizing — it is not a database row you must decode. "
            "Produce the user's requested summary or categorization from this content directly.\n\n"
        )
    else:
        legend = ""
    out = legend + body
    ctx.last_retrieval = out
    return out


def _parse_relative_days(query: str) -> int | None:
    q = query.lower()
    m = re.search(r"\blast\s+(\d{1,3})\s+days?\b", q)
    if m:
        return int(m.group(1))
    m = re.search(r"\blast\s+(\d{1,3})\s+weeks?\b", q)
    if m:
        return int(m.group(1)) * 7
    m = re.search(r"\blast\s+(\d{1,3})\s+months?\b", q)
    if m:
        return int(m.group(1)) * 30
    return None


def compute_mail_sender_breakdown(
    store: _SimilaritySearchStore,
    sample_chunks: int = 12_000,
    *,
    include_tool_instruction: bool = True,
) -> str:
    """Scan Chroma email metadatas and return ranked sender counts (same logic as the tool)."""
    cap = max(200, min(int(sample_chunks), 120_000))
    batch = 8000
    by_src: dict[str, str] = {}
    seen_chunks = 0
    offset = 0
    filt: dict[str, str] = {"source_type": "email"}
    try:
        while seen_chunks < cap:
            take = min(batch, cap - seen_chunks)
            data = store.get(
                where=filt,
                limit=take,
                offset=offset,
                include=["metadatas"],
            )
            metas = data.get("metadatas") or []
            if not metas:
                break
            for m in metas:
                if not m:
                    continue
                src = str(m.get("source") or "")
                snd = str(m.get("sender") or "").strip()
                if not src:
                    continue
                if snd:
                    by_src[src] = snd
                else:
                    by_src.setdefault(src, "")
            n = len(metas)
            seen_chunks += n
            offset += n
            if n < take:
                break
    except Exception as exc:
        return f"Could not read mail metadata from Chroma: {exc!r}"

    unknown = sum(1 for snd in by_src.values() if not snd)
    counts = Counter(s for s in by_src.values() if s)
    total_msgs = len(by_src)
    if total_msgs == 0:
        return "No indexed mail chunks matched (empty index or filter). Try `brain index --full`."

    lines = [
        "Indexed mail — senders (deduped per message `source` in scanned chunks):",
        f"- Distinct messages in sample: **{total_msgs}** (from **{seen_chunks}** chunks scanned)",
        f"- Senders with parsed metadata: **{total_msgs - unknown}** messages",
    ]
    if unknown:
        lines.append(
            f"- Messages missing `sender` metadata in sample: **{unknown}** "
            "(re-run `brain index --full` after upgrading).",
        )
    lines.append("")
    lines.append("**By sender** (message counts, highest first):")
    if not counts:
        lines.append("- _(No sender metadata in these chunks — run `brain index --full`.)_")
    else:
        ranked = counts.most_common(75)
        for sender, n_msg in ranked:
            lines.append(f"- `{sender}` — **{n_msg}** message(s)")
        tail = len(counts) - len(ranked)
        if tail > 0:
            lines.append(
                f"- … **{tail}** more sender address(es) not listed (raise ``sample_chunks``).",
            )
    lines.append("")
    if include_tool_instruction:
        lines.append(
            "Present this breakdown to the user; optionally group themes under top senders "
            "using `search_knowledge` + `source_scope` mail.",
        )
    return "\n".join(lines)


def wants_mail_sender_breakdown_query(text: str) -> bool:
    """Heuristic for chat preflight: user wants aggregate sender stats from indexed mail."""
    t = text.strip().lower()
    if len(t) < 6:
        return False
    mail_ctx = (
        any(
            p in t
            for p in (
                "email",
                "e-mail",
                "mailbox",
                "inbox",
                "gmail",
                "imap",
                "mail app",
                "apple mail",
                "indexed mail",
            )
        )
        or " mail " in f" {t} "
    )
    sender_ctx = any(
        p in t
        for p in (
            "sender",
            "senders",
            "from address",
            "from field",
            "who wrote",
            "who emailed",
            "who sent",
            "who mails",
        )
    )
    aggregate = any(
        p in t
        for p in (
            "categorize",
            "categorise",
            "categorization",
            "categorisation",
            "category ",
            "categories",
            "group ",
            "group,",
            "grouped",
            "rank",
            "sort",
            "count",
            "breakdown",
            "histogram",
            "distribution",
            "list ",
            "list,",
            "list all",
            "every ",
            "each ",
            "all my ",
            "top ",
            "most frequent",
            "how many",
        )
    )
    if sender_ctx and mail_ctx and aggregate:
        return True
    if sender_ctx and mail_ctx and ("who " in t or "whom " in t):
        return True
    return False


def _metadata_filter_for_scope(source_scope: str) -> dict[str, Any] | None:
    """Narrow Chroma search by chunk metadata.source_type; None = search everything."""
    key = source_scope.strip().lower()
    mapping: dict[str, dict[str, Any]] = {
        "screenshots": {"source_type": "screenshot"},
        "screenshot": {"source_type": "screenshot"},
        "mail": {"source_type": "email"},
        "email": {"source_type": "email"},
        "notes": {"source_type": "note"},
        "note": {"source_type": "note"},
        "pdfs": {"source_type": "pdf"},
        "pdf": {"source_type": "pdf"},
    }
    return mapping.get(key)


def build_tools(
    cfg: AppConfig,
    ctx: ToolContext,
    retriever: RetrieverLike,
    store: _SimilaritySearchStore,
) -> list[BaseTool]:
    @tool
    def mail_sender_breakdown(sample_chunks: int = 12_000) -> str:
        """REQUIRED when the user wants indexed mail grouped or counted by From/sender/address.

        Use for: categorize email senders, list who emailed most, sender breakdown, rank senders.
        Prefer this over semantic search for exact per-sender message counts in the sample.

        Dedupes by metadata ``source`` (one message). Returns markdown bullets ``sender — N``.
        """
        return compute_mail_sender_breakdown(store, sample_chunks)

    @tool
    def search_knowledge(query: str, source_scope: str = "any") -> str:
        """Search indexed notes, PDFs, screenshots, and mail.

        Use source_scope to limit where to search: any (default), screenshots, mail, notes,
        pdfs. For screenshot questions use source_scope screenshots. For email or inbox
        questions use source_scope mail — results are plain email headers and body text, not
        arbitrary key/value data. For counting or grouping mail **by sender / From**, call
        mail_sender_breakdown instead of only this tool.
        """
        filt = _metadata_filter_for_scope(source_scope)
        # If the user asked for a relative time window and we're searching mail, add a numeric
        # constraint so retrieval can't pull old messages.
        if filt and filt.get("source_type") == "email":
            days = _parse_relative_days(query)
            if days is not None and days > 0:
                cutoff = datetime.now(UTC).timestamp() - (days * 24 * 60 * 60)
                # Chroma expects a single operator node when mixing constraints.
                filt = {"$and": [{"source_type": "email"}, {"date_unix": {"$gte": cutoff}}]}
        docs = hybrid_retrieve(store, query, cfg=cfg, ctx=ctx, filt=filt)
        return _format_retrieval(ctx, docs)

    @tool
    def summarize_meeting(text_or_path: str) -> str:
        """Summarize meeting text; if input is an existing file path, read that file."""
        p = Path(text_or_path).expanduser()
        if p.is_file():
            content = p.read_text(encoding="utf-8", errors="replace")
        else:
            content = text_or_path
        if not content.strip():
            return "No content to summarize."
        bullets = [ln.strip() for ln in content.splitlines() if ln.strip()]
        head = bullets[:40]
        return "Key lines (pass to model for final summary):\n" + "\n".join(f"- {b}" for b in head)

    @tool
    def draft_reply(intent: str, thread_hint: str = "") -> str:
        """Email reply scaffold; combine with search_knowledge for thread context."""
        return (
            f"Intent: {intent}\nThread hint: {thread_hint}\n"
            "Use search_knowledge with the thread hint to pull relevant prior messages, "
            "then compose the reply in your final answer."
        )

    @tool
    def create_todo(title: str, details: str = "") -> str:
        """Append a todo item to the local todo list."""
        path = cfg.todos_path
        path.parent.mkdir(parents=True, exist_ok=True)
        items: list[dict[str, str]] = []
        if path.is_file():
            try:
                items = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                items = []
        items.append({"title": title, "details": details})
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")
        return f"Todo added ({len(items)} total)."

    @tool
    def list_todos() -> str:
        """List todos from the local store."""
        path = cfg.todos_path
        if not path.is_file():
            return "(no todos yet)"
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "(invalid todos file)"
        if not items:
            return "(empty todo list)"
        lines = [f"{i + 1}. {it.get('title', '')}" for i, it in enumerate(items)]
        return "\n".join(lines)

    @tool
    def write_note(relative_path: str, content: str) -> str:
        """Write a note file under an allowed directory from config notes_write_allowlist."""
        if not cfg.notes_write_allowlist:
            return "notes_write_allowlist is empty in config; refusing to write."
        target = Path(relative_path).expanduser()
        if target.is_absolute():
            return "relative_path must be relative."
        wrote: Path | None = None
        for root in cfg.notes_write_allowlist:
            dest = (root / target).resolve()
            if not _under_allowlist(dest, cfg.notes_write_allowlist):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            wrote = dest
            break
        if wrote is None:
            return "Path not under any allowed root."
        return f"Wrote {wrote}"

    @tool
    def open_calendar() -> str:
        """Open Calendar.app; requires tools.open_calendar_enabled in config."""
        if not cfg.open_calendar_enabled:
            return "open_calendar is disabled in config (tools.open_calendar_enabled)."
        script = 'tell application "Calendar" to activate'
        try:
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=30)
        except subprocess.CalledProcessError as e:
            return f"osascript failed: {e.stderr.decode(errors='replace')}"
        if cfg.calendar_url:
            try:
                subprocess.run(["open", cfg.calendar_url], check=False, timeout=30)
            except OSError as e:
                return f"Calendar activated but open URL failed: {e}"
        return "Calendar.app activated."

    tools: list[BaseTool] = [
        mail_sender_breakdown,
        search_knowledge,
        summarize_meeting,
        draft_reply,
        create_todo,
        list_todos,
        write_note,
        open_calendar,
    ]
    return tools
