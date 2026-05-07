from __future__ import annotations

from pathlib import Path

from brain.app_config import load_config


def test_repo_default_config_loads(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.ollama_chat_model == "llama3.2"


def test_user_config_overrides_merge(monkeypatch, tmp_path: Path) -> None:
    import brain.default_config as dc

    monkeypatch.setitem(dc.CONFIG, "ollama_chat_model", "base-model")
    (tmp_path / "config.py").write_text(
        'CONFIG = {"ollama_chat_model": "merged"}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.ollama_chat_model == "merged"


def test_empty_config_module_means_no_overrides(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.py").write_text("# CONFIG omitted\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.ollama_chat_model == "llama3.2"
