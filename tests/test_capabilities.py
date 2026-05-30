"""Tests for capability-aware routing & failover (tools/vision/reasoning/json)."""

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


@pytest.fixture
def cap_config(tmp_path: Path) -> Path:
    cfg = {
        "providers": {
            "capable": {"base_url": "http://capable.example/v1", "api_key": "k", "model_filter": None},
            "plain": {"base_url": "http://plain.example/v1", "api_key": "k", "model_filter": None},
        },
        "believed_free": ["capable/tool-model", "plain/free-model"],
        "model_reasoning": {},
        "model_capabilities": {
            "capable/tool-model": ["tools", "vision"],
            "capable/vision-only": ["vision"],
        },
        "free_limits": {},
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def server(monkeypatch, cap_config):
    return _load_server_with_config(monkeypatch, cap_config)


# ── pure detector helpers ──────────────────────────────────────────────────

def test_tool_use_forced(server):
    f = server._tool_use_forced
    tools = [{"type": "function", "function": {"name": "x"}}]
    assert f({"tools": tools, "tool_choice": "required"}) is True
    assert f({"tools": tools, "tool_choice": {"type": "function", "function": {"name": "x"}}}) is True
    assert f({"tools": tools, "tool_choice": "auto"}) is False
    assert f({"tools": tools, "tool_choice": "none"}) is False
    assert f({"tools": tools}) is False
    # forced choice but no tools provided -> not forced
    assert f({"tool_choice": "required"}) is False


def test_response_has_tool_call(server):
    f = server._response_has_tool_call
    with_call = json.dumps({"choices": [{"message": {"tool_calls": [{"id": "1"}]}}]}).encode()
    legacy = json.dumps({"choices": [{"message": {"function_call": {"name": "x"}}}]}).encode()
    no_call = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
    assert f(with_call) is True
    assert f(legacy) is True
    assert f(no_call) is False
    # safe defaults: malformed / unexpected shape -> True (don't fail over)
    assert f(b"not json") is True
    assert f(json.dumps({"foo": "bar"}).encode()) is True


def test_request_has_image(server):
    f = server._request_has_image
    img = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]}]}
    assert f(img) is True
    assert f({"messages": [{"role": "user", "content": "plain text"}]}) is False
    assert f({"messages": []}) is False
    assert f({}) is False


def test_request_wants_reasoning_and_json(server):
    assert server._request_wants_reasoning({"reasoning_effort": "high"}) is True
    assert server._request_wants_reasoning({"reasoning": {"effort": "high"}}) is True
    assert server._request_wants_reasoning({}) is False
    assert server._request_wants_json({"response_format": {"type": "json_object"}}) is True
    assert server._request_wants_json({"response_format": {"type": "json_schema"}}) is True
    assert server._request_wants_json({"response_format": {"type": "text"}}) is False
    assert server._request_wants_json({}) is False


def test_response_is_json(server):
    f = server._response_is_json
    good = json.dumps({"choices": [{"message": {"content": '{"a": 1}'}}]}).encode()
    bad = json.dumps({"choices": [{"message": {"content": "Sorry, here you go:"}}]}).encode()
    assert f(good) is True
    assert f(bad) is False
    # safe defaults
    assert f(b"not json") is True
    assert f(json.dumps({"choices": []}).encode()) is True


def test_needed_capabilities(server):
    assert server._needed_capabilities({"tools": [{"type": "function"}]}) == {"tools"}
    img = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}
    assert server._needed_capabilities(img) == {"vision"}
    assert server._needed_capabilities({}) == set()


def test_capability_failed_only_when_forced(server):
    tools = [{"type": "function", "function": {"name": "x"}}]
    no_call = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
    # forced tool + no tool_calls -> failure
    assert server._capability_failed({"tools": tools, "tool_choice": "required"}, no_call) is True
    # auto tool choice -> never a failure even without tool_calls
    assert server._capability_failed({"tools": tools, "tool_choice": "auto"}, no_call) is False
    # json forced + non-json -> failure
    bad_json = json.dumps({"choices": [{"message": {"content": "nope"}}]}).encode()
    assert server._capability_failed({"response_format": {"type": "json_object"}}, bad_json) is True


def test_model_capabilities_defensive(server):
    assert server._model_capabilities({}) == {}
    assert server._model_capabilities({"model_capabilities": None}) == {}
    assert server._model_capabilities({"model_capabilities": ["bad"]}) == {}  # non-dict
    parsed = server._model_capabilities({"model_capabilities": {
        "P/M": ["Tools", "bogus", "vision"],
    }})
    assert parsed == {"p/m": {"tools", "vision"}}  # lowercased, unknown dropped


def test_order_by_capability(server):
    cap_map = {"p1/m1": {"tools"}, "p2/m2": set()}
    candidates = [("p2", {}, "m2"), ("p1", {}, "m1")]
    ordered = server._order_by_capability(candidates, {"tools"}, cap_map)
    assert ordered[0][0] == "p1"  # tool-capable first
    # never drops, stable no-op when nothing needed
    assert server._order_by_capability(candidates, set(), cap_map) == candidates
    assert len(ordered) == 2


# ── reactive failover in the cycling path ───────────────────────────────────

def _resp(model: str, *, tool_call: bool) -> Response:
    if tool_call:
        body = {"choices": [{"message": {"tool_calls": [{"id": "1"}], "content": None}}]}
    else:
        body = {"choices": [{"message": {"content": "plain answer"}}]}
    return Response(json.dumps(body), status=200, content_type="application/json")


def test_failover_on_forced_tool_without_call(server, monkeypatch):
    calls: list[str] = []
    succeeded: list[tuple[str, str]] = []

    def fake_request(endpoint, pn, cfg, payload, timeout):
        calls.append(payload["model"])
        return _resp(payload["model"], tool_call=(payload["model"] == "m2"))

    monkeypatch.setattr(server, "_proxy_request", fake_request)
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    payload = {"tools": [{"type": "function", "function": {"name": "x"}}], "tool_choice": "required"}
    resp = server._proxy_cycling_non_streaming(
        "chat/completions", "test", candidates, payload, 5,
        on_success=lambda pn, um: succeeded.append((pn, um)),
    )
    assert calls == ["m1", "m2"]  # m1 failed the capability check, m2 tried
    assert succeeded == [("p2", "m2")]  # only the real success recorded
    assert b"tool_calls" in resp.get_data()


def test_no_failover_when_tool_choice_auto(server, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(server, "_proxy_request",
                        lambda e, pn, cfg, p, t: (calls.append(p["model"]) or _resp(p["model"], tool_call=False)))
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    payload = {"tools": [{"type": "function"}], "tool_choice": "auto"}
    server._proxy_cycling_non_streaming("chat/completions", "test", candidates, payload, 5)
    assert calls == ["m1"]  # first 200 returned immediately, no failover


def test_returns_last_when_all_fail_forced_tool(server, monkeypatch):
    succeeded: list = []
    monkeypatch.setattr(server, "_proxy_request",
                        lambda e, pn, cfg, p, t: _resp(p["model"], tool_call=False))
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    payload = {"tools": [{"type": "function"}], "tool_choice": "required"}
    resp = server._proxy_cycling_non_streaming(
        "chat/completions", "test", candidates, payload, 5,
        on_success=lambda pn, um: succeeded.append((pn, um)),
    )
    assert resp.status_code == 200  # last 200 body returned
    assert succeeded == []  # never recorded a success


# ── capability virtual endpoints ────────────────────────────────────────────

def _seed_route_cache(server, routes: dict[str, tuple[str, str]]):
    with server._model_route_cache_lock:
        server._model_route_cache.clear()
        server._model_route_cache.update(routes)


def test_capability_candidate_selectors(server):
    _seed_route_cache(server, {
        "capable__tool-model": ("capable", "tool-model"),
        "capable__vision-only": ("capable", "vision-only"),
        "plain__free-model": ("plain", "free-model"),
    })
    tool_cands = server._get_capability_model_candidates("tools")
    assert {(pn, um) for pn, _, um in tool_cands} == {("capable", "tool-model")}
    vision_cands = server._get_capability_model_candidates("vision")
    assert {(pn, um) for pn, _, um in vision_cands} == {("capable", "tool-model"), ("capable", "vision-only")}
    # tools/free intersects with believed_free (capable/tool-model is free)
    tool_free = server._get_capability_free_candidates("tools")
    assert {(pn, um) for pn, _, um in tool_free} == {("capable", "tool-model")}


def test_virtual_candidate_dispatch(server):
    _seed_route_cache(server, {
        "capable__tool-model": ("capable", "tool-model"),
        "capable__vision-only": ("capable", "vision-only"),
    })
    for name in ("llmproxy__tools", "llmproxy/tools", "llmproxy__vision",
                 "llmproxy__tools/free", "llmproxy__vision/free"):
        assert name in server._VIRTUAL_MODELS
        assert server._get_virtual_candidates(name) is not None
    assert {(pn, um) for pn, _, um in server._get_virtual_candidates("llmproxy__tools")} == {("capable", "tool-model")}
    assert "llmproxy__tools/free" in server._FREE_VIRTUAL_MODELS


def test_virtual_model_hint_for_capabilities(server):
    assert "model_capabilities" in server._virtual_model_hint("llmproxy__tools")


# ── expose_to_virtual_models flag ───────────────────────────────────────────

@pytest.fixture
def hidden_config(tmp_path: Path) -> Path:
    """Config with one hidden provider (expose_to_virtual_models: false) and one normal one."""
    cfg = {
        "providers": {
            "visible": {"base_url": "http://visible.example/v1", "api_key": "k", "model_filter": None},
            "hidden": {
                "base_url": "http://hidden.example/v1",
                "api_key": "k",
                "model_filter": None,
                "expose_to_virtual_models": False,
            },
            "local-hidden": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "k",
                "model_filter": None,
                "expose_to_virtual_models": False,
            },
        },
        "believed_free": ["visible/free-model", "hidden/free-model"],
        "model_reasoning": {
            "visible/fast": "exploratory",
            "hidden/fast": "exploratory",
        },
        "model_capabilities": {
            "visible/tool-model": ["tools"],
            "hidden/tool-model": ["tools"],
        },
        "free_limits": {},
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def hidden_server(monkeypatch, hidden_config):
    return _load_server_with_config(monkeypatch, hidden_config)


def test_hidden_provider_excluded_from_virtual_candidates(hidden_server):
    """A provider with expose_to_virtual_models: false must not appear in any virtual candidate list."""
    _seed_route_cache(hidden_server, {
        "visible__free-model": ("visible", "free-model"),
        "hidden__free-model": ("hidden", "free-model"),
        "visible__fast": ("visible", "fast"),
        "hidden__fast": ("hidden", "fast"),
        "visible__tool-model": ("visible", "tool-model"),
        "hidden__tool-model": ("hidden", "tool-model"),
        "local-hidden__llama": ("local-hidden", "llama"),
    })
    free_cands = hidden_server._get_free_model_candidates()
    assert all(pn != "hidden" for pn, _, _ in free_cands), "hidden provider leaked into free candidates"
    assert any(pn == "visible" for pn, _, _ in free_cands), "visible provider missing from free candidates"

    reasoning_cands = hidden_server._get_reasoning_model_candidates("exploratory")
    assert all(pn != "hidden" for pn, _, _ in reasoning_cands)
    assert any(pn == "visible" for pn, _, _ in reasoning_cands)

    cap_cands = hidden_server._get_capability_model_candidates("tools")
    assert all(pn != "hidden" for pn, _, _ in cap_cands)
    assert any(pn == "visible" for pn, _, _ in cap_cands)

    local_cands = hidden_server._get_local_model_candidates()
    assert all(pn != "local-hidden" for pn, _, _ in local_cands)


def test_hidden_provider_still_in_flat_model_list(hidden_server):
    """Models from a hidden provider still appear in the flat route cache (direct access works)."""
    _seed_route_cache(hidden_server, {
        "visible__free-model": ("visible", "free-model"),
        "hidden__free-model": ("hidden", "free-model"),
    })
    with hidden_server._model_route_cache_lock:
        snapshot = dict(hidden_server._model_route_cache)
    assert "hidden__free-model" in snapshot, "hidden provider's model should still be routable directly"
    assert "visible__free-model" in snapshot
