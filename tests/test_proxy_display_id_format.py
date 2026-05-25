"""Pin the display-id format returned by /v1/models.

Regression guard for the case where the proxy briefly used "model (provider)",
which contains a space and parens and is rejected by strict clients like Hermes.
The current format is "<upstream_model_id>__<provider_name>".
"""

from __future__ import annotations

import importlib
import re

import pytest

_DISPLAY_ID_RE = re.compile(r"^[^\s()]+__[A-Za-z0-9_.\-]+$")


@pytest.fixture
def server(monkeypatch, minimal_config):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(minimal_config))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    return server_mod


def _stub_response(model_ids):
    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"data": [{"id": mid, "object": "model"} for mid in model_ids]}
    return _R()


def test_proxy_id_uses_double_underscore(server, monkeypatch):
    """Every non-virtual model id must use the new `model__provider` form —
    no spaces, no parens, no leading slash truncation surface."""
    captured = _stub_response([
        "qwen2.5vl:3b",
        "llama3.2-3b-instruct",
        "nested/path/model-x",
    ])
    monkeypatch.setattr(server.requests, "get", lambda *a, **kw: captured)

    models = server._fetch_provider_models(
        "ollama",
        {"base_url": "http://upstream.example/v1", "api_key": "x"},
        timeout=1,
    )

    assert models, "expected the stubbed upstream to yield models"
    for m in models:
        mid = m["id"]
        assert " " not in mid, f"display id contains a space: {mid!r}"
        assert "(" not in mid and ")" not in mid, f"display id contains parens: {mid!r}"
        assert _DISPLAY_ID_RE.match(mid), (
            f"display id {mid!r} does not match expected `model__provider` shape"
        )
        assert mid.endswith("__ollama"), f"missing provider suffix: {mid!r}"


def test_resolver_accepts_new_format(server, monkeypatch):
    """_resolve_provider's cold-cache fallback must parse `model__provider`."""
    # Clear the cache so we exercise the cold-cache fallback path.
    with server._model_route_cache_lock:
        server._model_route_cache.clear()

    provider_name, _provider_cfg, upstream_model, err = server._resolve_provider(
        "qwen2.5vl:3b__fakeprov"
    )
    assert err is None, f"unexpected error: {err}"
    assert provider_name == "fakeprov"
    assert upstream_model == "qwen2.5vl:3b"


def test_spaces_in_upstream_model_name_are_sanitized(server, monkeypatch):
    """If an upstream id contains spaces, they must be replaced with `_` so the
    display id stays whitespace-free."""
    captured = _stub_response(["My Cool Model v1"])
    monkeypatch.setattr(server.requests, "get", lambda *a, **kw: captured)

    models = server._fetch_provider_models(
        "ollama",
        {"base_url": "http://upstream.example/v1", "api_key": "x"},
        timeout=1,
    )
    assert models
    mid = models[0]["id"]
    assert " " not in mid
    assert mid == "My_Cool_Model_v1__ollama"
    # The original upstream id is preserved on the route so requests still
    # forward to the upstream under its true name.
    assert models[0]["_upstream_id"] == "My Cool Model v1"
    assert models[0]["_route"] == ("ollama", "My Cool Model v1")


def test_spaces_in_provider_name_are_sanitized(server, monkeypatch):
    """If a provider name contains spaces, the display id replaces them with `_`."""
    captured = _stub_response(["llama3"])
    monkeypatch.setattr(server.requests, "get", lambda *a, **kw: captured)

    models = server._fetch_provider_models(
        "my provider",
        {"base_url": "http://upstream.example/v1", "api_key": "x"},
        timeout=1,
    )
    assert models
    mid = models[0]["id"]
    assert " " not in mid
    assert mid.endswith("__my_provider")


def test_resolver_still_accepts_legacy_format(server, monkeypatch):
    """Legacy `model (provider)` ids must still resolve (backward compat)."""
    with server._model_route_cache_lock:
        server._model_route_cache.clear()

    provider_name, _provider_cfg, upstream_model, err = server._resolve_provider(
        "qwen2.5vl:3b (fakeprov)"
    )
    assert err is None, f"unexpected error: {err}"
    assert provider_name == "fakeprov"
    assert upstream_model == "qwen2.5vl:3b"
