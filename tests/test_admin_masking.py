"""Tests that the admin API never leaks plaintext secrets and round-trips
env references verbatim."""

from __future__ import annotations

import importlib
import json

import pytest

CONFIG = {
    "providers": {
        "literal": {"base_url": "https://api.x/v1", "api_key": "sk-plaintext-abcdef123456", "model_filter": None},
        "envref": {"base_url": "https://${HOST}/v1", "api_key": "${OPENAI_API_KEY}", "model_filter": None},
        "nokey": {"base_url": "http://localhost:11434/v1", "model_filter": None},
    },
    "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR", "request_timeout": 5, "stream_timeout": 5},
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(CONFIG))
    monkeypatch.setenv("LLMPROXY_CONFIG", str(cfg_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    server_mod.app.config["TESTING"] = True
    return server_mod.app.test_client(), cfg_path


def test_plaintext_key_is_masked(client):
    c, _ = client
    body = c.get("/admin/api/providers").get_json()
    masked = body["literal"]["api_key"]
    assert masked != "sk-plaintext-abcdef123456"
    assert "plaintext" not in masked
    assert body["literal"]["api_key_set"] is True
    assert body["literal"]["api_key_is_env"] is False


def test_env_ref_returned_verbatim(client):
    c, _ = client
    body = c.get("/admin/api/providers").get_json()
    assert body["envref"]["api_key"] == "${OPENAI_API_KEY}"
    assert body["envref"]["api_key_is_env"] is True
    assert body["envref"]["base_url_is_env"] is True


def test_no_key_provider(client):
    c, _ = client
    body = c.get("/admin/api/providers").get_json()
    assert body["nokey"]["api_key"] == ""
    assert body["nokey"]["api_key_set"] is False


def test_full_config_never_contains_plaintext(client):
    c, _ = client
    raw = c.get("/admin/api/config").get_data(as_text=True)
    assert "sk-plaintext-abcdef123456" not in raw


def test_blank_submit_keeps_secret_on_disk(client):
    c, cfg_path = client
    resp = c.put("/admin/api/providers/literal", json={"base_url": "https://api.x/v2", "api_key": ""})
    assert resp.status_code == 200
    saved = json.loads(cfg_path.read_text())["providers"]["literal"]
    assert saved["api_key"] == "sk-plaintext-abcdef123456"  # untouched
    assert saved["base_url"] == "https://api.x/v2"
