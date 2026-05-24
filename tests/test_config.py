"""Coverage of llmproxy/config.py — load, save, hot-reload, schema fallback."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from llmproxy import config as config_mod


def test_load_returns_defaults_when_file_missing(tmp_path: Path):
    path = tmp_path / "nope.json"
    cfg = config_mod.load_config(str(path), force_reload=True)
    assert cfg["providers"] == {}
    assert cfg["believed_free"] == []


def test_save_then_load_roundtrip(tmp_path: Path):
    path = tmp_path / "cfg.json"
    src = {
        "providers": {"p": {"base_url": "http://x", "api_key": "k", "model_filter": None}},
        "believed_free": ["p/m"],
    }
    assert config_mod.save_config(src, str(path)) is True
    cfg = config_mod.load_config(str(path), force_reload=True)
    assert cfg["providers"]["p"]["base_url"] == "http://x"
    assert cfg["believed_free"] == ["p/m"]


def test_hot_reload_picks_up_changes(tmp_path: Path):
    path = tmp_path / "cfg.json"
    config_mod.save_config({"providers": {}, "believed_free": ["a"]}, str(path))
    first = config_mod.load_config(str(path), force_reload=True)
    assert first["believed_free"] == ["a"]

    # Sleep ensures mtime advances on filesystems with 1s resolution.
    time.sleep(1.05)
    config_mod.save_config({"providers": {}, "believed_free": ["b"]}, str(path))
    second = config_mod.load_config(str(path))
    assert second["believed_free"] == ["b"]


def test_parse_model_string_basic():
    p, m = config_mod.parse_model_string("openai/gpt-4o")
    assert p == "openai"
    assert m == "gpt-4o"


def test_parse_model_string_nested_slashes():
    p, m = config_mod.parse_model_string("openrouter/foo/bar:free")
    assert p == "openrouter"
    assert m == "foo/bar:free"


def test_parse_model_string_no_slash_raises():
    with pytest.raises(ValueError):
        config_mod.parse_model_string("noslash")


def test_model_is_allowed_filter_semantics():
    assert config_mod.model_is_allowed({"model_filter": None}, "x") is True
    assert config_mod.model_is_allowed({"model_filter": []}, "x") is False
    assert config_mod.model_is_allowed({"model_filter": ["x"]}, "x") is True
    assert config_mod.model_is_allowed({"model_filter": ["y"]}, "x") is False


def test_get_provider_helper():
    cfg = {"providers": {"p": {"base_url": "u"}}}
    assert config_mod.get_provider(cfg, "p") == {"base_url": "u"}
    assert config_mod.get_provider(cfg, "missing") is None


def test_get_config_path_priority(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(tmp_path / "env.json"))
    # Explicit override wins over env var
    p = config_mod.get_config_path(str(tmp_path / "cli.json"))
    assert p == tmp_path / "cli.json"
    # Env var beats default
    p = config_mod.get_config_path(None)
    assert p == tmp_path / "env.json"
