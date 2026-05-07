from __future__ import annotations

import pytest
from brain.app_config import AppConfig, load_config


@pytest.fixture
def app_config(monkeypatch, tmp_path) -> AppConfig:
    """Defaults only: cwd has no ``config.py``, so nothing is merged from disk."""
    monkeypatch.chdir(tmp_path)
    return load_config()
