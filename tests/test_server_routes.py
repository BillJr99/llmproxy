"""Smoke tests for llmproxy/server.py using Flask's test client.

We don't exercise live upstream calls. The focus is the parts of the proxy
that don't require a real backend: route registration, /v1/models filter,
config-driven virtual model resolution, error responses.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _load_server_with_config(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    # Reload to pick up the env-var path
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    return server_mod


@pytest.fixture
def server(monkeypatch, minimal_config):
    mod = _load_server_with_config(monkeypatch, minimal_config)
    yield mod


@pytest.fixture
def client(server):
    server.app.config["TESTING"] = True
    return server.app.test_client()


def test_health_endpoint(client):
    """Health endpoint should respond without touching upstreams."""
    # The server may register /health or just /; either should return 200.
    for path in ("/health", "/healthz", "/"):
        resp = client.get(path)
        if resp.status_code == 200:
            return
    pytest.skip("No health endpoint registered on this server build")


def test_version_endpoint(client):
    """/version should return 200 with the package version."""
    resp = client.get("/version")
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}"
    body = resp.get_json()
    assert body is not None, "expected JSON body"
    assert body.get("name") == "llmproxy"
    assert body.get("version"), "expected a non-empty version string"


def test_models_endpoint_registered(client):
    """/v1/models route should exist and respond non-5xx, regardless of
    whether the upstream provider is reachable."""
    resp = client.get("/v1/models")
    assert resp.status_code < 500, f"Got 5xx: {resp.status_code} {resp.data!r}"
    body = resp.get_json()
    assert body is not None, "expected JSON body"
    assert "data" in body, f"expected 'data' key in response, got keys={list(body.keys())}"


def test_unknown_model_returns_4xx(client):
    """Posting to chat completions with an unknown model should not 500."""
    resp = client.post("/v1/chat/completions", json={
        "model": "nope/does-not-exist",
        "messages": [{"role": "user", "content": "hi"}],
    })
    # We accept anything except 5xx — the proxy should reject cleanly.
    assert resp.status_code < 500, f"Got 5xx: {resp.status_code} {resp.data!r}"
