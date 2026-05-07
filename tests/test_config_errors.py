from __future__ import annotations

from pathlib import Path

import pytest
from brain.app_config import load_config
from brain.errors import ConfigError


def test_user_unknown_config_key(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.py").write_text(
        'CONFIG = {"not_a_real_key": True}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="Unknown config keys"):
        load_config()


def test_user_config_not_dict(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.py").write_text("CONFIG = ()\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="CONFIG must be a dict"):
        load_config()
