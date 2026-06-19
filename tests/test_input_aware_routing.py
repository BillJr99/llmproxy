"""Tests for request-fit triage on all */free and */local virtuals: the
request's estimated size and type bias which reasoning tier AND model size is
tried first, layered on top of capacity/random ordering and below the hard
capability ordering. Includes tier-containment: a */free virtual only ever
serves free-list models and a */local virtual only ever serves local models.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from flask import Response


def _load_server(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def tier_config(tmp_path: Path) -> Path:
    cfg = {
        "providers": {
            "pf": {"base_url": "http://pf.example/v1", "api_key": "k", "model_filter": None},
        },
        "believed_free": ["pf/fast", "pf/mid", "pf/big"],
        "model_reasoning": {
            "pf/fast": "exploratory",
            "pf/mid": "standard",
            "pf/big": "deep",
        },
        "free_limits": {},
        "fusion": {"enabled": False},
        "sync_believed_free_on_startup": False,
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def server(monkeypatch, tier_config):
    return _load_server(monkeypatch, tier_config)


# ── pure helpers ────────────────────────────────────────────────────────────

def test_estimate_payload_tokens(server):
    p = {"messages": [{"role": "user", "content": "x" * 400}]}
    assert server._estimate_payload_tokens(p) == 100  # 400 chars / 4
    parts = {"messages": [{"role": "user", "content": [{"text": "ab"}, {"text": "cd"}]}]}
    assert server._estimate_payload_tokens(parts) == 1


def test_wants_thinking(server):
    assert server._wants_thinking({"reasoning_effort": "high"}) is True
    assert server._wants_thinking({"reasoning_effort": "low"}) is False
    assert server._wants_thinking({"reasoning": {"effort": "x"}}) is True
    assert server._wants_thinking({}) is False


def test_target_reasoning_tier_by_size_and_type(server):
    small = {"messages": [{"role": "user", "content": "hi"}]}
    assert server._target_reasoning_tier(small) == "exploratory"
    medium = {"messages": [{"role": "user", "content": "x" * (4 * 4000)}]}
    assert server._target_reasoning_tier(medium) == "standard"
    large = {"messages": [{"role": "user", "content": "x" * (4 * 20000)}]}
    assert server._target_reasoning_tier(large) == "deep"
    # explicit thinking overrides a tiny prompt
    assert server._target_reasoning_tier({"messages": [{"role": "user", "content": "hi"}],
                                          "reasoning_effort": "high"}) == "deep"


def test_order_by_request_fit_exact_tier_first_and_stable(server):
    cands = [("pf", {}, "big"), ("pf", {}, "fast"), ("pf", {}, "mid"), ("pf", {}, "untagged")]
    rmap = {"pf/fast": "exploratory", "pf/mid": "standard", "pf/big": "deep"}
    small = {"messages": [{"role": "user", "content": "hi"}]}  # exploratory target
    out = [c[2] for c in server._order_by_request_fit(cands, small, rmap)]
    assert out[0] == "fast"          # exact tier first
    assert out[-1] == "big"          # far tier (deep) last
    assert "untagged" in out[1:3]    # untagged sorts neutral (middle)


def test_order_by_request_fit_sizes_within_a_single_tier(server):
    # All candidates share the same (deep) tier, so the size axis decides which
    # right-sized deep model is tried first — this is the deep/free sub-virtual case.
    cands = [("pf", {}, "deep-8b"), ("pf", {}, "deep-70b"), ("pf", {}, "deep-1b")]
    rmap = {"pf/deep-8b": "deep", "pf/deep-70b": "deep", "pf/deep-1b": "deep"}
    light = {"messages": [{"role": "user", "content": "hi"}]}  # prefer smallest
    out_light = [c[2] for c in server._order_by_request_fit(cands, light, rmap)]
    assert out_light[0] == "deep-1b"
    assert out_light[-1] == "deep-70b"
    thinking = {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"}
    out_deep = [c[2] for c in server._order_by_request_fit(cands, thinking, rmap)]
    assert out_deep[0] == "deep-70b"  # heavy work -> biggest available
    assert out_deep[-1] == "deep-1b"


def test_free_and_local_virtual_predicates(server):
    assert server._is_free_virtual_model("llmproxy__free")
    assert server._is_free_virtual_model("llmproxy__deep/free")
    assert server._is_free_virtual_model("llmproxy/tools/free")
    assert not server._is_free_virtual_model("llmproxy__local")
    assert server._is_local_virtual_model("llmproxy__local")
    assert server._is_local_virtual_model("llmproxy/local")
    assert server._is_local_virtual_model("llmproxy__deep/local")
    assert not server._is_local_virtual_model("llmproxy__free")
    assert not server._is_local_virtual_model("llmproxy__deep/free")


# ── integration: the first model actually tried matches the tier ────────────

def _seed(server, routes):
    with server._model_route_cache_lock:
        server._model_route_cache.clear()
        server._model_route_cache.update(routes)


def _chat_ok(model):
    return Response(json.dumps({
        "id": "x", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }), status=200, content_type="application/json")


@pytest.mark.parametrize("payload_extra,expected_first", [
    ({}, "fast"),                              # tiny prompt -> exploratory
    ({"reasoning_effort": "high"}, "big"),     # thinking -> deep
])
def test_free_first_pick_follows_tier(server, monkeypatch, payload_extra, expected_first):
    _seed(server, {"pf__fast": ("pf", "fast"), "pf__mid": ("pf", "mid"), "pf__big": ("pf", "big")})
    tried: list[str] = []

    def fake_request(endpoint, pn, cfg, payload, timeout):
        tried.append(payload["model"])
        return _chat_ok(payload["model"])

    monkeypatch.setattr(server, "_proxy_request", fake_request)
    client = server.app.test_client()
    body = {"model": "llmproxy__free", "messages": [{"role": "user", "content": "hi"}]}
    body.update(payload_extra)
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert tried[0] == expected_first  # first upstream tried matches the input tier


# ── sub-virtual: right-sized model from a single-tier pool ──────────────────

@pytest.fixture
def deep_free_config(tmp_path: Path) -> Path:
    cfg = {
        "providers": {
            "pf": {"base_url": "http://pf.example/v1", "api_key": "k", "model_filter": None},
        },
        "believed_free": ["pf/deep-1b", "pf/deep-8b", "pf/deep-70b"],
        "model_reasoning": {
            "pf/deep-1b": "deep", "pf/deep-8b": "deep", "pf/deep-70b": "deep",
        },
        "free_limits": {},
        "fusion": {"enabled": False},
        "sync_believed_free_on_startup": False,
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.mark.parametrize("payload_extra,expected_first", [
    ({}, "deep-1b"),                           # light prompt -> smallest deep model
    ({"reasoning_effort": "high"}, "deep-70b"),  # heavy work -> biggest deep model
])
def test_deep_free_subvirtual_picks_right_size(monkeypatch, deep_free_config, payload_extra, expected_first):
    server = _load_server(monkeypatch, deep_free_config)
    _seed(server, {"pf__deep-1b": ("pf", "deep-1b"),
                   "pf__deep-8b": ("pf", "deep-8b"),
                   "pf__deep-70b": ("pf", "deep-70b")})
    tried: list[str] = []

    def fake_request(endpoint, pn, cfg, payload, timeout):
        tried.append(payload["model"])
        return _chat_ok(payload["model"])

    monkeypatch.setattr(server, "_proxy_request", fake_request)
    client = server.app.test_client()
    body = {"model": "llmproxy__deep/free", "messages": [{"role": "user", "content": "hi"}]}
    body.update(payload_extra)
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert tried[0] == expected_first


# ── tier containment: */free stays free, */local stays local ────────────────

@pytest.fixture
def mixed_tier_config(tmp_path: Path) -> Path:
    cfg = {
        "providers": {
            "cloud": {"base_url": "http://cloud.example/v1", "api_key": "k", "model_filter": None},
            "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "k", "model_filter": None},
        },
        # cloud/free-model is free; cloud/paid-model is paid; ollama/* is local.
        "believed_free": ["cloud/free-model"],
        "model_reasoning": {},
        "free_limits": {},
        "fusion": {"enabled": False},
        "sync_believed_free_on_startup": False,
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def test_free_virtual_never_leaves_free_tier(monkeypatch, mixed_tier_config):
    server = _load_server(monkeypatch, mixed_tier_config)
    _seed(server, {
        "cloud__free-model": ("cloud", "free-model"),
        "cloud__paid-model": ("cloud", "paid-model"),
        "ollama__local-model": ("ollama", "local-model"),
    })
    tried: list[str] = []

    def fake_request(endpoint, pn, cfg, payload, timeout):
        tried.append((pn, payload["model"]))
        return _chat_ok(payload["model"])

    monkeypatch.setattr(server, "_proxy_request", fake_request)
    client = server.app.test_client()
    resp = client.post("/v1/chat/completions", json={
        "model": "llmproxy__free", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    # Only the free-list model is ever contacted — no local, no paid leakage.
    assert tried == [("cloud", "free-model")]


def test_local_virtual_never_leaves_local_tier(monkeypatch, mixed_tier_config):
    server = _load_server(monkeypatch, mixed_tier_config)
    _seed(server, {
        "cloud__free-model": ("cloud", "free-model"),
        "cloud__paid-model": ("cloud", "paid-model"),
        "ollama__local-model": ("ollama", "local-model"),
    })
    tried: list[str] = []

    def fake_request(endpoint, pn, cfg, payload, timeout):
        tried.append((pn, payload["model"]))
        return _chat_ok(payload["model"])

    monkeypatch.setattr(server, "_proxy_request", fake_request)
    client = server.app.test_client()
    resp = client.post("/v1/chat/completions", json={
        "model": "llmproxy__local", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    # Only the localhost-backed model is ever contacted.
    assert tried == [("ollama", "local-model")]
