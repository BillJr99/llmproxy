"""The route cache is dual-keyed; walking it is not the same as walking models.

``_rebuild_route_cache`` stores every model twice — once as the canonical
``provider__model`` id and once as the advertised ``provider/model`` form — so an
inbound id in either shape resolves without string surgery. Right for a lookup,
wrong for a walk: iterating ``.items()`` yielded every model twice, and every
virtual candidate pool in the proxy was silently doubled.

The bug was invisible for as long as it existed because no test ever seeded the
shape production actually has: every fixture writes one key per model. So these
tests seed BOTH, which is the only thing that would have caught it.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _make_server(monkeypatch, tmp_path: Path, models: dict[str, str]):
    """models: {upstream_id: provider}. Seeds the cache exactly as production does."""
    providers = {p: {"base_url": f"http://{p}.example/v1", "api_key": "k"}
                 for p in set(models.values())}
    cfg = {
        "providers": providers,
        "believed_free": [f"{p}/{m}" for m, p in models.items()],
        "model_reasoning": {f"{p}/{m}": "deep" for m, p in models.items()},
        "model_capabilities": {f"{p}/{m}": ["tools"] for m, p in models.items()},
        "server": {"log_level": "ERROR"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "flagship_models.json").write_text(json.dumps({
        "last_refresh_at": "2026-09-19T00:00:00+00:00",
        "members": [f"{p}/{m}" for m, p in models.items()],
        "model_scores": {m.replace("-", "").replace(".", ""): 0.9 for m in models},
    }), encoding="utf-8")

    monkeypatch.setenv("LLMPROXY_CONFIG", str(path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)

    with server_mod._model_route_cache_lock:
        server_mod._model_route_cache.clear()
        for upstream, provider in models.items():
            route = (provider, upstream)
            # BOTH keys — this is what _rebuild_route_cache really writes.
            server_mod._model_route_cache[f"{provider}__{upstream}"] = route
            server_mod._model_route_cache[f"{provider}/{upstream}"] = route
    return server_mod


@pytest.fixture
def server(monkeypatch, tmp_path: Path):
    return _make_server(monkeypatch, tmp_path, {"glm-5.3": "alpha"})


# ── the helper ──────────────────────────────────────────────────────────────

def test_the_cache_really_does_hold_two_keys_per_model(server):
    """Guard the premise: if this stops being true the rest proves nothing."""
    assert len(server._model_route_cache) == 2
    assert len(set(server._model_route_cache.values())) == 1


def test_distinct_routes_yields_each_model_once(server):
    assert server._get_distinct_routes() == [("alpha", "glm-5.3")]


def test_first_seen_order_is_preserved(monkeypatch, tmp_path):
    """Downstream passes sort on top of this order, so it must not be reshuffled."""
    s = _make_server(monkeypatch, tmp_path,
                     {"m-a": "alpha", "m-b": "beta", "m-c": "alpha"})
    assert s._get_distinct_routes() == [
        ("alpha", "m-a"), ("beta", "m-b"), ("alpha", "m-c"),
    ]


# ── every pool selector ─────────────────────────────────────────────────────

@pytest.mark.parametrize("selector", [
    "_get_free_model_candidates",
    "_get_local_model_candidates",
    "_get_loadbalanced_candidates",
    "_get_all_model_candidates",
])
def test_a_pool_selector_returns_each_model_once(server, selector):
    got = [(pn, um) for pn, _pc, um in getattr(server, selector)()]
    assert len(got) == len(set(got)), f"{selector} returned duplicates: {got}"


@pytest.mark.parametrize("level", ["deep", "flagship"])
def test_reasoning_pools_return_each_model_once(server, level):
    got = [(pn, um) for pn, _pc, um in server._get_reasoning_model_candidates(level)]
    assert got == [("alpha", "glm-5.3")]


def test_capability_pool_returns_each_model_once(server):
    got = [(pn, um) for pn, _pc, um in server._get_capability_model_candidates("tools")]
    assert got == [("alpha", "glm-5.3")]


def test_per_provider_bare_pool_returns_each_model_once(server):
    got = [(pn, um) for pn, _pc, um in
           server._get_provider_virtual_candidates("alpha", "")]
    assert got == [("alpha", "glm-5.3")]


def test_the_intersection_helpers_do_not_reintroduce_duplicates(server):
    """These filter one doubled list against a set built from another, so the
    set hid the duplication on the membership side while the result kept it."""
    got = [(pn, um) for pn, _pc, um in server._get_reasoning_free_candidates("deep")]
    assert got == [("alpha", "glm-5.3")]


# ── the consequences ────────────────────────────────────────────────────────

def test_failover_moves_to_a_different_model_not_the_same_one(monkeypatch, tmp_path):
    """The point of the ranking. Duplicates tie on every sort key, so they landed
    adjacently and a failover from the best model retried that same model before
    reaching the second-best."""
    s = _make_server(monkeypatch, tmp_path, {"strong": "alpha", "weaker": "beta"})
    monkeypatch.setattr(s, "_get_flagship_scores",
                        lambda *a, **k: {"strong": 0.99, "weaker": 0.5})
    ordered = s._flagship_ordered_candidates(
        s._get_reasoning_model_candidates("flagship"), s._get_flagship_scores(), {})
    pairs = [(pn, um) for pn, _pc, um in ordered]
    assert pairs == [("alpha", "strong"), ("beta", "weaker")]
    assert pairs[0][1] != pairs[1][1], "the second attempt must be a DIFFERENT model"


def test_provider_headroom_counts_each_model_once(monkeypatch, tmp_path):
    """The one site where the duplicate gave a wrong answer rather than a
    redundant one: it SUMS usage, so a provider hit zero capacity at half its
    real free_allowance and its models were demoted to paid early."""
    s = _make_server(monkeypatch, tmp_path, {"m1": "alpha", "m2": "alpha"})
    s._reset_usage()
    cfg = {"base_url": "http://alpha.example/v1", "api_key": "k",
           "free_allowance": {"requests_per_day": 4}}
    # Three real requests against an allowance of four: headroom remains. Counted
    # twice they come to six, which exhausts it — so this is the case where the
    # duplicate flips the answer rather than merely inflating a number.
    for _ in range(3):
        s._record_usage("alpha", "m1", config=s.load_config())
    assert s._provider_free_headroom("alpha", cfg) is True, (
        "3 of a 4-request allowance must leave headroom, not read as 6"
    )
    s._record_usage("alpha", "m2", config=s.load_config())
    assert s._provider_free_headroom("alpha", cfg) is False, "4 of 4 is spent"


def test_flagship_recompute_counts_each_candidate_once(server, monkeypatch, tmp_path):
    from llmproxy import flagship as _flagship
    monkeypatch.setattr(_flagship, "fetch_profiles", lambda sources=None: {
        _flagship.normalize_model_id("glm-5.3"): {
            "scores": {"openrouter_aa": 53.4},
            "context_length": 1_310_720, "supports_tools": True,
        },
    })
    state = server._recompute_flagship_members(
        server.load_config(), str(tmp_path / "config.json"))
    assert state["candidates_considered"] == 1, "one model in the cache, counted once"
