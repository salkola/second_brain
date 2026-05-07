from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from brain.errors import ConfigError


def _expand_path(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


@dataclass
class AppConfig:
    """Typed runtime config from merged CONFIG dicts (defaults + ``./config.py``)."""

    ollama_base_url: str
    ollama_chat_model: str
    embedding_model_name: str
    chroma_persist_directory: Path
    index_state_db: Path
    notes_roots: list[Path]
    pdf_roots: list[Path]
    screenshot_roots: list[Path]
    mail_library: Path
    mail_message_limit: int
    mail_since_days: int | None
    mail_only_inbox: bool
    chunk_size: int
    chunk_overlap: int
    retrieve_k: int
    hybrid_retrieval: bool
    hybrid_bm25_top_k: int
    hybrid_bm25_max_chunks: int
    rerank: bool
    rerank_model_name: str
    rerank_top_k: int
    parent_context: bool
    parent_context_window: int
    parent_context_max_chunks_per_source: int
    todos_path: Path
    notes_write_allowlist: list[Path]
    open_calendar_enabled: bool
    calendar_url: str | None
    log_level: str


_CONFIG_KEYS = frozenset(f.name for f in fields(AppConfig))
_PATH_FIELDS = frozenset(
    {
        "chroma_persist_directory",
        "index_state_db",
        "mail_library",
        "todos_path",
    }
)
_LIST_PATH_FIELDS = frozenset(
    {
        "notes_roots",
        "pdf_roots",
        "screenshot_roots",
        "notes_write_allowlist",
    }
)


def _merge_config_dicts(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    unknown = frozenset(overrides) - _CONFIG_KEYS
    if unknown:
        raise ConfigError(f"Unknown config keys: {sorted(unknown)}")
    merged = {**defaults, **overrides}
    missing = _CONFIG_KEYS - frozenset(merged)
    if missing:
        raise ConfigError(f"Missing config keys after merge: {sorted(missing)}")
    return merged


def _dict_to_app_config(d: dict[str, Any]) -> AppConfig:
    out: dict[str, Any] = {}
    for name in _CONFIG_KEYS:
        v = d[name]
        if name in _PATH_FIELDS:
            out[name] = _expand_path(v)
        elif name in _LIST_PATH_FIELDS:
            out[name] = [_expand_path(x) for x in v]
        else:
            out[name] = v
    return AppConfig(**out)


def _exec_module(path: Path, module_name: str):
    if path.suffix != ".py":
        raise ConfigError(f"Configuration file must be .py, got: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"Cannot load configuration from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SyntaxError as exc:
        raise ConfigError(f"Invalid Python syntax in {path}: {exc}") from exc
    except Exception as exc:
        raise ConfigError(f"Error while loading {path}: {exc}") from exc
    return module


def _load_user_config_overrides() -> dict[str, Any]:
    path = (Path.cwd() / "config.py").resolve()
    if not path.is_file():
        return {}
    module = _exec_module(path, "_brain_user_config")
    if not hasattr(module, "CONFIG"):
        return {}
    cfg = module.CONFIG
    if not isinstance(cfg, dict):
        raise ConfigError(
            f"{path}: CONFIG must be a dict, got {type(cfg).__name__}",
        )
    return cfg


def load_config() -> AppConfig:
    """Merge ``brain.default_config.CONFIG`` with ``./config.py`` ``CONFIG`` if present."""
    from brain.default_config import CONFIG as defaults

    overrides = _load_user_config_overrides()
    merged = _merge_config_dicts(dict(defaults), overrides)
    return _dict_to_app_config(merged)
