"""Tests for per-provider virtual models: llmproxy__<provider>[/<dimension>].

These cycle/fail over within a SINGLE enabled, non-local, virtual-exposing
provider, e.g. llmproxy__visible/deep or the bare aggregator llmproxy__visible.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from flask import Response


def _load_server_with_config(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    return server_mod


def _seed_route_cache(server, routes: dict[str, tuple[str, str]]):
    with server._model_route_cache_lock:
        server._model_route_cache.clear()
        server._model_route_cache.update(routes)


@pytest.fixture
def pp_config(tmp_path: Path) -> Path:
    cfg = {
        "providers": {
            # Normal, eligible, non-local provider.
            "visible": {"base_url": "http://visible.example/v1", "api_key": "k", "model_filter": None},
            # Provider named exactly like a reasoning level -> precedence test.
            "standard": {"base_url": "http://standard.example/v1", "api_key": "k", "model_filter": None},
            # Opted out of virtual exposure.
            "hidden": {
                "base_url": "http://hidden.example/v1", "api_key": "k",
                "model_filter": None, "expose_to_virtual_models": False,
            },
            # Local provider -> never per-provider eligible.
            "localp": {"base_url": "http://localhost:11434/v1", "api_key": "k", "model_filter": None},
        },
        "believed_free": ["visible/free-model", "standard/free-model"],
        "model_reasoning": {
            "visible/fast": "exploratory",
            "visible/think": "deep",
            "standard/think": "deep",
        },
        "model_capabilities": {
            "visible/tool-model": ["tools"],
            "visible/vision-model": ["vision"],
        },
        "free_limits": {},
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5, "models_cache_ttl": 0,
                   "response_cache_ttl": 0},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def server(monkeypatch, pp_config):
    return _load_server_with_config(monkeypatch, pp_config)


# ── recognition & precedence ────────────────────────────────────────────────

def test_split_basic_forms(server):
    f = server._split_per_provider_virtual
    assert f("llmproxy__visible") == ("visible", "")
    assert f("llmproxy__visible/deep") == ("visible", "deep")
    assert f("llmproxy__visible/tools") == ("visible", "tools")
    assert f("llmproxy__visible/free") == ("visible", "free")
    # legacy single-slash prefix also recognised
    assert f("llmproxy/visible/deep") == ("visible", "deep")


def test_split_precedence_global_wins(server):
    """A provider named 'standard' does not shadow the global 'standard/free'."""
    f = server._split_per_provider_virtual
    # global <level>/free form wins -> not per-provider
    assert f("llmproxy__standard/free") is None
    assert f("llmproxy__standard") is None      # global bare 'standard' wins
    # but other dimensions of that provider still resolve per-provider
    assert f("llmproxy__standard/deep") == ("standard", "deep")
    assert f("llmproxy__standard/tools") == ("standard", "tools")


def test_split_rejects_ineligible(server):
    f = server._split_per_provider_virtual
    assert f("llmproxy__hidden/deep") is None        # opted out
    assert f("llmproxy__localp/deep") is None         # local provider
    assert f("llmproxy__unknown/deep") is None        # not configured
    assert f("llmproxy__visible/bogus") is None        # bad dimension
    assert f("llmproxy__visible/deep/free") is None    # multi-slash -> not per-provider
    assert f("visible/deep") is None                   # missing prefix


# ── membership helpers ──────────────────────────────────────────────────────

def test_membership_helpers(server):
    assert server._is_virtual_model("llmproxy__visible/deep") is True
    assert server._is_virtual_model("llmproxy__visible") is True
    assert server._is_virtual_model("llmproxy__free") is True        # global still works
    assert server._is_virtual_model("visible/deep") is False
    assert server._is_free_virtual_model("llmproxy__visible/free") is True
    assert server._is_free_virtual_model("llmproxy__visible/deep") is False
    assert server._is_free_virtual_model("llmproxy__free") is True   # global free


# ── candidate selection ─────────────────────────────────────────────────────

def test_provider_virtual_candidates_filtered(server):
    _seed_route_cache(server, {
        "visible__think": ("visible", "think"),
        "standard__think": ("standard", "think"),
        "visible__fast": ("visible", "fast"),
    })
    deep = server._get_provider_virtual_candidates("visible", "deep")
    assert {(pn, um) for pn, _, um in deep} == {("visible", "think")}
    # only this provider's models, never the other provider tagged 'deep'
    assert all(pn == "visible" for pn, _, _ in deep)


def test_provider_virtual_candidates_bare_aggregator(server):
    _seed_route_cache(server, {
        "visible__a": ("visible", "a"),
        "visible__b": ("visible", "b"),
        "standard__c": ("standard", "c"),
    })
    bare = server._get_provider_virtual_candidates("visible", "")
    assert {um for _, _, um in bare} == {"a", "b"}


def test_provider_virtual_candidates_free_and_caps(server):
    _seed_route_cache(server, {
        "visible__free-model": ("visible", "free-model"),
        "visible__tool-model": ("visible", "tool-model"),
        "visible__vision-model": ("visible", "vision-model"),
    })
    free = server._get_provider_virtual_candidates("visible", "free")
    assert {um for _, _, um in free} == {"free-model"}
    tools = server._get_provider_virtual_candidates("visible", "tools")
    assert {um for _, _, um in tools} == {"tool-model"}
    vision = server._get_provider_virtual_candidates("visible", "vision")
    assert {um for _, _, um in vision} == {"vision-model"}


def test_dispatch_routes_to_provider_selector(server):
    _seed_route_cache(server, {
        "visible__think": ("visible", "think"),
        "standard__think": ("standard", "think"),
    })
    cands = server._get_virtual_candidates("llmproxy__visible/deep")
    assert {(pn, um) for pn, _, um in cands} == {("visible", "think")}


# ── advertising in /v1/models ───────────────────────────────────────────────

def test_advertised_in_models_list(server, monkeypatch):
    routes = {
        "visible__free-model": ("visible", "free-model"),
        "visible__think": ("visible", "think"),
        "visible__tool-model": ("visible", "tool-model"),
        "visible__vision-model": ("visible", "vision-model"),
        "hidden__think": ("hidden", "think"),
        "localp__llama": ("localp", "llama"),
    }

    def _fake_rebuild(providers_cfg, timeout):
        _seed_route_cache(server, routes)
        return [{"id": k, "name": k} for k in routes]

    monkeypatch.setattr(server, "_rebuild_route_cache", _fake_rebuild)
    monkeypatch.setattr(server, "_sync_local_provider_models_once", lambda: None)
    server._models_list_cache = None

    resp = server.app.test_client().get("/v1/models")
    ids = {m["id"] for m in resp.get_json()["data"]}

    # eligible provider's per-provider virtuals are advertised
    assert "llmproxy__visible" in ids                # bare aggregator
    assert "llmproxy__visible/deep" in ids
    assert "llmproxy__visible/tools" in ids
    assert "llmproxy__visible/vision" in ids
    assert "llmproxy__visible/free" in ids
    # exploratory has no backing model for visible -> not advertised
    assert "llmproxy__visible/exploratory" not in ids
    # ineligible providers never get per-provider virtuals
    assert not any(i.startswith("llmproxy__hidden") for i in ids)
    assert not any(i.startswith("llmproxy__localp") for i in ids)


# ── capacity-aware /free dispatch ───────────────────────────────────────────

def test_provider_free_is_capacity_aware(server, monkeypatch):
    _seed_route_cache(server, {"visible__free-model": ("visible", "free-model")})
    captured = {}

    def _fake_cycle(endpoint, model_full, ordered, payload, timeout, on_success=None):
        captured["ordered"] = ordered
        captured["on_success"] = on_success
        return Response(b'{"ok": true}', status=200, content_type="application/json")

    monkeypatch.setattr(server, "_proxy_cycling_non_streaming", _fake_cycle)
    monkeypatch.setattr(server, "_sync_local_provider_models_once", lambda: None)

    resp = server.app.test_client().post(
        "/v1/chat/completions",
        json={"model": "llmproxy__visible/free", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    # capacity-aware path records usage on success
    assert captured["on_success"] is server._record_usage
    assert {pn for pn, _, _ in captured["ordered"]} == {"visible"}


# ── hints ───────────────────────────────────────────────────────────────────

def test_hints(server):
    h = server._virtual_model_hint
    assert "visible" in h("llmproxy__visible/deep") and "model_reasoning" in h("llmproxy__visible/deep")
    assert "model_capabilities" in h("llmproxy__visible/tools")
    assert "free" in h("llmproxy__visible/free")
    assert "route cache" in h("llmproxy__visible")
