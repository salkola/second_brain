from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from brain.tools.builtin import ToolContext, build_tools


class _FakeRetriever:
    def invoke(self, _query: str):
        return []


class _FakeChroma:
    _mail_metas = [
        {"source": "mail:e:1", "sender": "lists@apache.org"},
        {"source": "mail:e:1", "sender": "lists@apache.org"},
        {"source": "mail:e:2", "sender": "bot@test.dev"},
    ]

    def similarity_search(self, *_args, **_kwargs):
        return []

    def get(self, *_args, **kwargs):
        limit = kwargs.get("limit") or 100
        offset = kwargs.get("offset") or 0
        sl = self._mail_metas[offset : offset + limit]
        return {"metadatas": sl}


def test_mail_sender_breakdown_dedupes_by_message(app_config) -> None:
    ctx = ToolContext()
    tools = {t.name: t for t in build_tools(app_config, ctx, _FakeRetriever(), _FakeChroma())}
    out = tools["mail_sender_breakdown"].invoke({"sample_chunks": 500})
    assert "lists@apache.org" in out
    assert "bot@test.dev" in out
    assert "`lists@apache.org` — **1** message(s)" in out
    assert "`bot@test.dev` — **1** message(s)" in out
    assert "Distinct messages in sample: **2**" in out
    assert "from **3** chunks scanned" in out


def test_write_note_blocked_without_allowlist(app_config) -> None:
    cfg = replace(app_config, notes_write_allowlist=[])
    ctx = ToolContext()
    tools = {t.name: t for t in build_tools(cfg, ctx, _FakeRetriever(), _FakeChroma())}
    out = tools["write_note"].invoke({"relative_path": "x.md", "content": "hi"})
    assert "empty" in out.lower()


def test_write_note_allowed(app_config, tmp_path: Path) -> None:
    allow = tmp_path / "vault"
    allow.mkdir()
    cfg = replace(app_config, notes_write_allowlist=[allow])
    ctx = ToolContext()
    tools = {t.name: t for t in build_tools(cfg, ctx, _FakeRetriever(), _FakeChroma())}
    out = tools["write_note"].invoke({"relative_path": "n.md", "content": "# t"})
    assert "Wrote" in out
    assert (allow / "n.md").read_text() == "# t"
