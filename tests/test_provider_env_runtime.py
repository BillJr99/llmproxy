"""Verify the server resolves ${VAR} references at request time when building
upstream URLs and Authorization headers (not at config-load time)."""

from __future__ import annotations

import importlib
import json

import pytest


class _FakeResp:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    text = "{}"

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"id": "gpt-4o", "object": "model"}]}


@pytest.fixture
def server_mod(monkeypatch, tmp_path):
    cfg = {
        "providers": {
            "envprov": {
                "base_url": "https://${UPSTREAM_HOST}/v1",
                "api_key": "${UPSTREAM_KEY}",
                "model_filter": None,
            }
        },
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setenv("LLMPROXY_CONFIG", str(cfg_path))
    monkeypatch.setenv("UPSTREAM_HOST", "api.example.com")
    monkeypatch.setenv("UPSTREAM_KEY", "sk-runtime-secret")
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


def test_fetch_models_uses_resolved_url_and_key(server_mod, monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeResp()

    monkeypatch.setattr(server_mod.requests, "get", fake_get)

    cfg = server_mod.load_config()["providers"]["envprov"]
    models = server_mod._fetch_provider_models("envprov", cfg, timeout=5)

    assert captured["url"] == "https://api.example.com/v1/models"
    assert captured["headers"]["Authorization"] == "Bearer sk-runtime-secret"
    assert models and models[0]["id"].startswith("envprov")


def test_on_disk_config_still_holds_reference(server_mod):
    # The stored config must remain raw; only consumption resolves.
    cfg = server_mod.load_config()["providers"]["envprov"]
    assert cfg["base_url"] == "https://${UPSTREAM_HOST}/v1"
    assert cfg["api_key"] == "${UPSTREAM_KEY}"


def test_fetch_models_omits_auth_header_when_no_key(server_mod, monkeypatch):
    """A keyless (e.g. local) provider must not send 'Authorization: Bearer '."""
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["headers"] = headers or {}
        return _FakeResp()

    monkeypatch.setattr(server_mod.requests, "get", fake_get)
    cfg = {"base_url": "http://localhost:11434/v1", "model_filter": None}
    server_mod._fetch_provider_models("local", cfg, timeout=5)
    assert "Authorization" not in captured["headers"]


def test_upstream_headers_omit_auth_when_no_key(server_mod):
    with server_mod.app.test_request_context("/"):
        headers = server_mod._upstream_headers({"base_url": "http://x/v1"})
    assert "Authorization" not in headers

    with server_mod.app.test_request_context("/"):
        headers2 = server_mod._upstream_headers({"api_key": "k"})
    assert headers2["Authorization"] == "Bearer k"


def test_upstream_headers_resolve_env_key(server_mod, monkeypatch):
    monkeypatch.setenv("UPSTREAM_KEY", "sk-runtime-secret")
    with server_mod.app.test_request_context("/"):
        headers = server_mod._upstream_headers({"api_key": "${UPSTREAM_KEY}"})
    assert headers["Authorization"] == "Bearer sk-runtime-secret"
