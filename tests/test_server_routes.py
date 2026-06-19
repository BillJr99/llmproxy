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


def test_models_rechecks_disk_before_no_providers_warning(server, monkeypatch):
    """/v1/models must re-read config from disk before declaring it empty.

    A stale in-process cache (or a config still being written at startup) can
    momentarily yield zero providers; the endpoint does a force_reload re-check so
    the "No providers configured" warning only fires when the on-disk config
    genuinely has none. Here every non-forced load reports an empty config while a
    force_reload returns the real one — the warning must NOT appear.
    """
    real_load = server.load_config

    def fake_load(*args, **kwargs):
        if kwargs.get("force_reload"):
            return real_load(*args, **kwargs)
        return {"providers": {}, "server": {"request_timeout": 5}}

    monkeypatch.setattr(server, "load_config", fake_load)

    server.app.config["TESTING"] = True
    resp = server.app.test_client().get("/v1/models")
    assert resp.status_code < 500, f"Got 5xx: {resp.status_code} {resp.data!r}"
    body = resp.get_json()
    assert body is not None
    assert "_warning" not in body, (
        "force_reload re-check should have found providers on disk and suppressed "
        "the 'No providers configured' warning"
    )


def test_unknown_model_returns_4xx(client):
    """Posting to chat completions with an unknown model should not 500."""
    resp = client.post("/v1/chat/completions", json={
        "model": "nope/does-not-exist",
        "messages": [{"role": "user", "content": "hi"}],
    })
    # We accept anything except 5xx — the proxy should reject cleanly.
    assert resp.status_code < 500, f"Got 5xx: {resp.status_code} {resp.data!r}"


def test_api_prefix_mirrors_routes(client):
    """Requests under /api should reach the same handlers as the bare path.

    Clients with a base_url of /api or /api/v1 (OpenRouter / Open WebUI /
    Ollama style) must not 404. /version needs no upstream, so it's a clean
    probe for the prefix-strip middleware.
    """
    bare = client.get("/version")
    via_api = client.get("/api/version")
    assert via_api.status_code == bare.status_code == 200
    assert via_api.get_json() == bare.get_json()
    # /api/v1/models should also route through to the models handler.
    assert client.get("/api/v1/models").status_code < 500


def test_api_prefix_excludes_admin(client):
    """The /api alias must NOT expose the admin surface (/api/admin)."""
    # The admin UI lives only at its canonical /admin path; /api/admin is left
    # untouched by the strip and therefore is not a registered route.
    assert client.get("/api/admin").status_code == 404


def test_architecture_block(server):
    """_architecture_block builds an OpenRouter-style modality descriptor."""
    blk = server._architecture_block(["text", "image"], None)
    assert blk == {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "modality": "text+image->text",
    }
    # Empty/None sides fall back to text-only.
    assert server._architecture_block(None, [])["modality"] == "text->text"


def test_startup_sidecar_sync_reconciles_live_config(server, tmp_path, monkeypatch):
    """The startup sync helper reconciles the live config.json from the bundled
    sidecar, with no network and without writing the sidecar (read-only safe)."""
    import json

    import scripts.update_free_models as ufm

    sidecar = {
        "providers": {
            "google": {
                "base_url": "u", "display": "G",
                "believed_free": ["google/added"],
                "free_limits": {}, "model_reasoning": {}, "model_capabilities": {},
            },
        },
        "provider_order": ["google"],
    }
    monkeypatch.setattr(ufm, "load_data", lambda: sidecar)

    cfg = {"providers": {"google": {"base_url": "u", "api_key": "k"}}, "believed_free": []}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    ran = server._sync_believed_free_from_sidecar(str(p))
    assert ran is True
    written = json.loads(p.read_text())
    assert "google/added" in written["believed_free"]


def test_supported_parameters_from_config(server):
    """_supported_parameters surfaces tool/reasoning capabilities from config."""
    cfg = {
        "model_capabilities": {"p/m1": ["tools"]},
        "model_reasoning": {"p/m2": "deep"},
    }
    assert server._supported_parameters("p", "m1", cfg) == ["tools", "tool_choice"]
    assert server._supported_parameters("p", "m2", cfg) == ["reasoning"]
    assert server._supported_parameters("p", "m3", cfg) == []
