"""Pin the display-id format returned by /v1/models.

Regression guards for:
- the original "model (provider)" form (rejected by Hermes for the spaces);
- the PR #27 "model__provider" form (correct chars, but provider on the wrong side);
- the current "provider__model" form (matches the canonical slash order).
All three legacy input forms must continue to resolve as input on chat/completions.
"""

from __future__ import annotations

import importlib
import re

import pytest

_DISPLAY_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+__[^\s()]+$")


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


def test_proxy_id_uses_provider_first_double_underscore(server, monkeypatch):
    """Every non-virtual model id must use the `provider__model` form —
    provider on the left, no spaces, no parens, no leading slash."""
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
            f"display id {mid!r} does not match expected `provider__model` shape"
        )
        assert mid.startswith("ollama__"), f"provider must come first: {mid!r}"


def test_resolver_accepts_new_format(server, monkeypatch):
    """_resolve_provider must parse the current `provider__model` form."""
    with server._model_route_cache_lock:
        server._model_route_cache.clear()

    provider_name, _provider_cfg, upstream_model, err = server._resolve_provider(
        "fakeprov__qwen2.5vl:3b"
    )
    assert err is None, f"unexpected error: {err}"
    assert provider_name == "fakeprov"
    assert upstream_model == "qwen2.5vl:3b"


def test_resolver_still_accepts_pr27_format(server, monkeypatch):
    """Legacy `model__provider` ids from PR #27 must still resolve."""
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
    assert mid == "ollama__My_Cool_Model_v1"
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
    assert mid.startswith("my_provider__")


def test_resolver_still_accepts_legacy_paren_format(server, monkeypatch):
    """Pre-PR #27 `model (provider)` ids must still resolve (backward compat)."""
    with server._model_route_cache_lock:
        server._model_route_cache.clear()

    provider_name, _provider_cfg, upstream_model, err = server._resolve_provider(
        "qwen2.5vl:3b (fakeprov)"
    )
    assert err is None, f"unexpected error: {err}"
    assert provider_name == "fakeprov"
    assert upstream_model == "qwen2.5vl:3b"


def test_virtual_ids_advertised_with_double_underscore(server):
    """Every virtual id in _VIRTUAL_MODELS that the server emits as primary
    (i.e. the NEW set) must start with `llmproxy__`, never `llmproxy/`."""
    new = server._NEW_VIRTUAL_MODELS
    assert new, "expected _NEW_VIRTUAL_MODELS to be non-empty"
    for vid in new:
        assert vid.startswith("llmproxy__"), f"virtual id should use __ prefix: {vid!r}"
        assert " " not in vid and "(" not in vid and ")" not in vid


def test_legacy_virtual_ids_still_in_membership_set(server):
    """The legacy `llmproxy/...` virtual ids must still resolve as input —
    they remain in _VIRTUAL_MODELS so chat/completions dispatches them to the
    virtual handler instead of treating them as provider/model pairs."""
    legacy = server._LEGACY_VIRTUAL_MODELS
    assert "llmproxy/free" in legacy
    assert "llmproxy/deep/free" in legacy
    # All legacy ids must be in the combined membership set.
    for vid in legacy:
        assert vid in server._VIRTUAL_MODELS


def test_virtual_candidates_dispatch_matches_for_legacy_and_new(server):
    """`_get_virtual_candidates("llmproxy/free")` and
    `_get_virtual_candidates("llmproxy__free")` must produce the same list."""
    new = server._get_virtual_candidates("llmproxy__free")
    legacy = server._get_virtual_candidates("llmproxy/free")
    assert new == legacy

    new_tiered = server._get_virtual_candidates("llmproxy__deep/free")
    legacy_tiered = server._get_virtual_candidates("llmproxy/deep/free")
    assert new_tiered == legacy_tiered
