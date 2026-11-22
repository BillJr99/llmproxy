"""Virtual endpoints walk the rest of their cost tier before failing.

A narrowed virtual (``flagship/free``, ``deep/free``, ``<provider>/free``, the
untiered ``flagship``/``deep``/``tools``) used to cycle only through its own
slice, so an empty or failing slice returned an error while dozens of other
free models were available. Each virtual now appends a fallback tail:

* ``*/free``  -> every other free model
* ``*/local`` -> every other local model
* untiered    -> every free model, then every local model (never paid)
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
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_sync_local_provider_models_once", lambda: None)
    return server_mod


ROUTES = {
    "cloud__a-free": ("cloud", "a-free"),
    "cloud__b-free": ("cloud", "b-free"),
    "cloud__c-free": ("cloud", "c-free"),
    "cloud__paid-model": ("cloud", "paid-model"),
    "other__x-free": ("other", "x-free"),
    "localp__llama": ("localp", "llama"),
    "localp__qwen": ("localp", "qwen"),
}

PAID = ("cloud", "paid-model")
LOCAL = {("localp", "llama"), ("localp", "qwen")}
FREE = {("cloud", "a-free"), ("cloud", "b-free"), ("cloud", "c-free"), ("other", "x-free")}


@pytest.fixture
def server(monkeypatch, tmp_path: Path):
    cfg = {
        "providers": {
            "cloud": {"base_url": "http://cloud.example/v1", "api_key": "k", "model_filter": None},
            "other": {"base_url": "http://other.example/v1", "api_key": "k", "model_filter": None},
            "localp": {"base_url": "http://localhost:11434/v1", "api_key": "k", "model_filter": None},
        },
        "believed_free": [],
        "model_reasoning": {
            "cloud/a-free": "deep",
            "cloud/paid-model": "deep",
            "localp/llama": "deep",
        },
        "model_capabilities": {},
        "free_limits": {},
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    srv = _load_server_with_config(monkeypatch, p)
    with srv._model_route_cache_lock:
        srv._model_route_cache.clear()
        srv._model_route_cache.update(ROUTES)
    return srv


def _capture(server, monkeypatch):
    captured: dict = {}

    def _fake_cycle(endpoint, model_full, ordered, payload, timeout, on_success=None, **kwargs):
        captured["ordered"] = [(pn, um) for pn, _pc, um in ordered]
        captured["route_reason"] = kwargs.get("route_reason")
        return Response(b'{"ok": true}', status=200, content_type="application/json")

    monkeypatch.setattr(server, "_proxy_cycling_non_streaming", _fake_cycle)
    return captured


def _post(server, model, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}
    return server.app.test_client().post("/v1/chat/completions", json=body)


def test_empty_flagship_free_falls_back_to_every_free_model(server, monkeypatch):
    monkeypatch.setattr(server, "_get_flagship_models", lambda *a, **k: set())
    captured = _capture(server, monkeypatch)
    resp = _post(server, "llmproxy/flagship__free")
    assert resp.status_code == 200
    assert set(captured["ordered"]) == FREE
    assert "fallback=4" in captured["route_reason"]


def test_deep_free_tries_own_pool_first_then_rest_of_free(server, monkeypatch):
    captured = _capture(server, monkeypatch)
    resp = _post(server, "llmproxy/deep__free")
    assert resp.status_code == 200
    ordered = captured["ordered"]
    assert ordered[0] == ("cloud", "a-free")
    assert set(ordered[1:]) == FREE - {("cloud", "a-free")}
    assert len(ordered) == len(set(ordered)), "no candidate is walked twice"


def test_untiered_falls_back_to_free_then_local_never_paid(server, monkeypatch):
    captured = _capture(server, monkeypatch)
    resp = _post(server, "llmproxy/deep")
    assert resp.status_code == 200
    ordered = captured["ordered"]
    primary = {("cloud", "a-free"), PAID, ("localp", "llama")}
    assert set(ordered[:3]) == primary
    tail = ordered[3:]
    assert set(tail) == (FREE | LOCAL) - primary
    # free part of the tail comes before the local part
    kinds = ["local" if c in LOCAL else "free" for c in tail]
    assert kinds == sorted(kinds, key=lambda k: k == "local")
    assert ordered.count(PAID) == 1, "paid only via the endpoint's own pool"


def test_untiered_empty_pool_never_reaches_paid(server, monkeypatch):
    monkeypatch.setattr(server, "_get_flagship_models", lambda *a, **k: set())
    captured = _capture(server, monkeypatch)
    resp = _post(server, "llmproxy/flagship")
    assert resp.status_code == 200
    assert PAID not in captured["ordered"]
    assert set(captured["ordered"]) == FREE | LOCAL


def test_local_sub_virtual_falls_back_to_every_local_model(server, monkeypatch):
    captured = _capture(server, monkeypatch)
    _post(server, "llmproxy/deep__local")
    assert captured["ordered"][0] == ("localp", "llama")
    assert set(captured["ordered"]) == LOCAL


def test_per_provider_free_falls_back_to_other_providers(server, monkeypatch):
    captured = _capture(server, monkeypatch)
    _post(server, "llmproxy__other/free")
    assert captured["ordered"][0] == ("other", "x-free")
    assert set(captured["ordered"]) == FREE


@pytest.mark.parametrize("model", ["llmproxy/free", "llmproxy/local", "llmproxy/loadbalanced"])
def test_full_pools_get_no_tail(server, monkeypatch, model):
    captured = _capture(server, monkeypatch)
    _post(server, model)
    assert "fallback=" not in (captured["route_reason"] or "")
    assert len(captured["ordered"]) == len(set(captured["ordered"]))


def test_gate_dropped_candidates_are_tried_last(server, monkeypatch):
    # b-free is known to lack tools: it is walked, but only after everything else.
    monkeypatch.setattr(
        server, "_model_capabilities",
        lambda *a, **k: {"cloud/b-free": {"vision"}},
    )
    captured = _capture(server, monkeypatch)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    _post(server, "llmproxy/free", tools=tools)
    assert captured["ordered"][-1] == ("cloud", "b-free")
    assert set(captured["ordered"]) == FREE


def test_failing_primary_pool_is_served_by_fallback(server, monkeypatch):
    """End to end through the real cycling loop: the flagship/free slice 500s,
    a free model outside it answers."""
    monkeypatch.setattr(server, "_get_flagship_models", lambda *a, **k: {"cloud/a-free"})
    calls: list[tuple[str, str]] = []

    def _fake_proxy(endpoint, pn, pc, payload, timeout):
        calls.append((pn, payload["model"]))
        if (pn, payload["model"]) == ("cloud", "a-free"):
            return Response(b'{"error": {"message": "boom"}}', status=500,
                            content_type="application/json")
        return Response(json.dumps({"choices": [{"message": {"content": "ok"}}]}),
                        status=200, content_type="application/json")

    monkeypatch.setattr(server, "_proxy_request", _fake_proxy)
    resp = _post(server, "llmproxy/flagship__free")
    assert resp.status_code == 200
    assert calls[0] == ("cloud", "a-free")
    assert len(calls) == 2 and calls[1] in FREE - {("cloud", "a-free")}
