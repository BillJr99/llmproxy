"""Tests for input-aware first-pick on the general virtuals (llmproxy__free /
llmproxy__local): the request's estimated size and type bias which reasoning
tier is tried first, layered on top of capacity/random ordering and below the
hard capability ordering.
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


def test_order_by_reasoning_fit_exact_first_and_stable(server):
    cands = [("pf", {}, "big"), ("pf", {}, "fast"), ("pf", {}, "mid"), ("pf", {}, "untagged")]
    rmap = {"pf/fast": "exploratory", "pf/mid": "standard", "pf/big": "deep"}
    out = [c[2] for c in server._order_by_reasoning_fit(cands, "exploratory", rmap)]
    assert out[0] == "fast"          # exact tier first
    assert out[-1] == "big"          # far tier (deep) last
    assert "untagged" in out[1:3]    # untagged sorts neutral (middle)


def test_is_general_virtual(server):
    assert server._is_general_virtual("llmproxy__free")
    assert server._is_general_virtual("llmproxy__local")
    assert server._is_general_virtual("llmproxy/free")
    assert not server._is_general_virtual("llmproxy__deep/free")
    assert not server._is_general_virtual("llmproxy__tools")


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
