"""Round-trip and atomic-write tests for config save/load, including that raw
${VAR} references survive a save->load cycle unresolved."""

from __future__ import annotations

import importlib
import json
from pathlib import Path


def _fresh_config_mod(monkeypatch, cfg_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(cfg_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    return config_mod


def test_save_then_load_roundtrip(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    config_mod = _fresh_config_mod(monkeypatch, cfg_path)
    data = {
        "providers": {"p": {"base_url": "https://${HOST}/v1", "api_key": "${KEY}", "model_filter": None}},
        "believed_free": ["p/x"],
        "model_reasoning": {}, "model_capabilities": {}, "free_limits": {},
        "server": dict(config_mod.DEFAULT_SERVER_CONFIG),
    }
    assert config_mod.save_config(data) is True
    loaded = config_mod.load_config(force_reload=True)
    # Raw env references must survive unresolved on disk and in load_config.
    assert loaded["providers"]["p"]["api_key"] == "${KEY}"
    assert loaded["providers"]["p"]["base_url"] == "https://${HOST}/v1"
    assert loaded["believed_free"] == ["p/x"]


def test_atomic_save_leaves_no_temp_files(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    config_mod = _fresh_config_mod(monkeypatch, cfg_path)
    config_mod.save_config({"providers": {}, "server": {}})
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "config.json"]
    assert leftovers == [], f"unexpected temp files left behind: {leftovers}"
    # File is valid JSON.
    json.loads(cfg_path.read_text())


def test_save_failure_preserves_existing_file(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    config_mod = _fresh_config_mod(monkeypatch, cfg_path)
    config_mod.save_config({"providers": {"keep": {"base_url": "http://x/v1"}}, "server": {}})
    original = cfg_path.read_text()

    # Force the JSON serialization to fail mid-write (non-serializable value).
    bad = {"providers": {"p": {"base_url": object()}}}
    assert config_mod.save_config(bad) is False
    # Original config is intact (atomic replace never happened).
    assert cfg_path.read_text() == original
