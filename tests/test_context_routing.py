"""Tests for context-window-aware candidate ordering.

Before this pass, per-model context limits were parsed for the ``/v1/models``
listing and then discarded. A conversation that outgrew a candidate got a 400,
which is non-transient, so it failed straight over to another candidate chosen
with no regard for context, which 400'd too: a long agentic session walked the
whole pool and ended on the last 400 or a 503.

The invariant that matters most here is the one inherited from
``_order_by_capability``: the pass REORDERS and never drops, and an *unknown*
context window is neutral rather than a demotion. Bad metadata must not be able
to turn a working request into a hard failure.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _load_server_with_config(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def server(monkeypatch, tmp_path: Path):
    cfg = {
        "providers": {"p": {"base_url": "http://p/v1", "api_key": "k", "model_filter": None}},
        "believed_free": [], "model_reasoning": {}, "model_capabilities": {}, "free_limits": {},
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return _load_server_with_config(monkeypatch, p)


CFG = {"base_url": "http://p/v1", "api_key": "k"}
SMALL = ("small", CFG, "m8k")
BIG = ("big", CFG, "m200k")
UNKNOWN = ("unk", CFG, "mU")
CANDS = [SMALL, BIG, UNKNOWN]
DISCOVERED = {"small/m8k": 8192, "big/m200k": 200000}


def _long(tokens: int) -> dict:
    return {"messages": [{"role": "user", "content": "x" * (tokens * 4)}]}


# ── coercion ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    (8192, 8192), ("8192", 8192), (8192.0, 8192), (131072, 131072),
    (None, None), (True, None), (False, None), (0, None), (-1, None),
    ("abc", None), ({}, None), ([], None),
])
def test_coerce_context_length(server, raw, expected):
    assert server._coerce_context_length(raw) == expected


# ── config override map ─────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [None, "nope", 5, [], {"m": "x"}, {"m": 0}, {"m": -1}])
def test_get_model_context_is_defensive(server, raw):
    """A hand-edited config must never raise on the routing hot path."""
    assert isinstance(server._get_model_context({"model_context": raw}), dict)


def test_get_model_context_lowercases_and_filters(server):
    out = server._get_model_context({"model_context": {"P/M8K": 4096, "_note": 1, "ok": 2048}})
    assert out == {"p/m8k": 4096, "ok": 2048}


def test_model_context_two_form_lookup(server):
    assert server._model_context_window("p", "m", {"m": 111}, {}) == 111
    assert server._model_context_window("p", "m", {"p/m": 222}, {}) == 222
    assert server._model_context_window("p", "m", {}, {"p/m": 333}) == 333
    assert server._model_context_window("p", "m", {}, {}) is None


def test_config_override_beats_discovery(server):
    """model_context exists to correct gateways that misreport their window."""
    assert server._model_context_window("big", "m200k", {"big/m200k": 4096}, DISCOVERED) == 4096


# ── estimator ───────────────────────────────────────────────────────────────

def test_estimate_context_tokens_counts_tools(server):
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"x": "y" * 4000}}}],
    }
    assert server._estimate_context_tokens(payload) > server._estimate_payload_tokens(payload)


def test_estimate_context_tokens_counts_tool_call_arguments(server):
    payload = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [
            {"id": "c", "function": {"name": "f", "arguments": "z" * 4000}}]},
    ]}
    assert server._estimate_context_tokens(payload) > server._estimate_payload_tokens(payload)


def test_estimate_context_tokens_matches_base_without_tools(server):
    payload = {"messages": [{"role": "user", "content": "hello there"}]}
    assert server._estimate_context_tokens(payload) == server._estimate_payload_tokens(payload)


def test_required_tokens_reserves_output_room(server):
    """A model that only just fits the prompt would 400 on the completion."""
    tiny = {"messages": [{"role": "user", "content": "hi"}]}
    assert server._required_context_tokens(tiny) >= server._CONTEXT_OUTPUT_RESERVE


def test_required_tokens_honors_an_explicit_budget(server):
    """An explicit max_tokens larger than the reserve is what gets reserved."""
    payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32000}
    assert server._required_context_tokens(payload) >= 32000
    smaller = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
    assert server._required_context_tokens(smaller) == server._CONTEXT_OUTPUT_RESERVE


# ── the ordering pass ───────────────────────────────────────────────────────

def test_noop_when_everything_fits(server):
    """Identity, not merely equality — this is what lets the pass run last."""
    out = server._order_by_context_fit(CANDS, {"messages": []}, {}, DISCOVERED)
    assert out is CANDS


def test_noop_when_no_context_is_known(server):
    out = server._order_by_context_fit(CANDS, _long(500_000), {}, {})
    assert out is CANDS


def test_demotes_the_known_too_small_candidate(server):
    out = server._order_by_context_fit(CANDS, _long(100_000), {}, DISCOVERED)
    assert out[-1] == SMALL
    assert out[0] == BIG


def test_unknown_context_is_neutral_not_demoted(server):
    """An untagged model sits ahead of one known to be too small."""
    out = server._order_by_context_fit(CANDS, _long(100_000), {}, DISCOVERED)
    assert out.index(UNKNOWN) < out.index(SMALL)


def test_never_drops_a_candidate(server):
    out = server._order_by_context_fit(CANDS, _long(500_000), {}, DISCOVERED)
    assert len(out) == len(CANDS)
    assert {c[0] for c in out} == {c[0] for c in CANDS}


def test_override_can_sink_a_large_model(server):
    out = server._order_by_context_fit(
        CANDS, _long(100_000), {"big/m200k": 4096}, DISCOVERED)
    assert out[0] == UNKNOWN


def test_ordering_is_stable_within_a_band(server):
    a = ("a", CFG, "ma")
    b = ("b", CFG, "mb")
    cands = [a, b, SMALL]
    out = server._order_by_context_fit(cands, _long(100_000), {}, DISCOVERED)
    assert out.index(a) < out.index(b)


def test_single_candidate_is_never_reordered(server):
    one = [SMALL]
    assert server._order_by_context_fit(one, _long(500_000), {}, DISCOVERED) is one


# ── discovery cache ─────────────────────────────────────────────────────────

def test_rebuild_populates_the_context_cache(server, monkeypatch):
    models = [
        {"id": "p__a", "context_length": 8192, "_route": ("p", "a")},
        {"id": "p__b", "context_length": None, "_route": ("p", "b")},
        {"id": "p__c", "context_length": "131072", "_route": ("p", "c")},
    ]
    monkeypatch.setattr(server, "_fetch_provider_models", lambda *a, **k: list(models))
    server._rebuild_route_cache({"p": {"base_url": "http://p/v1"}}, 5)
    snap = server._get_model_context_snapshot()
    assert snap["p/a"] == 8192
    assert snap["p/c"] == 131072
    assert "p/b" not in snap, "an unusable window must stay unknown, not become 0"


def test_context_snapshot_never_warms_the_cache(server, monkeypatch):
    called: list = []
    monkeypatch.setattr(server, "_rebuild_route_cache",
                        lambda *a, **k: called.append(1) or [])
    server._model_context_cache.clear()
    assert server._get_model_context_snapshot() == {}
    assert called == []
