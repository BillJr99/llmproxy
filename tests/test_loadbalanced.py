"""Tests for the cost-tiered llmproxy__loadbalanced virtual model.

The loadbalanced virtual walks a strict cost waterfall — free → local → paid —
optimizing each tier for the request (capability, then size/reasoning fit, then
the tier's base order). Paid is a true last resort: it is never ordered ahead of
a free or local candidate, even when only a paid model carries the needed
capability tag (silent failover handles feasibility).
"""

from __future__ import annotations

import importlib
import json
import os
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


def _seed_route_cache(server, routes: dict[str, tuple[str, str]]):
    with server._model_route_cache_lock:
        server._model_route_cache.clear()
        server._model_route_cache.update(routes)


# Pricing map (provider/model lowercased -> (input, output) per token).
_PRICING = {
    "paida/expensive": (1e-5, 2e-5),
    "paidb/cheap": (1e-7, 2e-7),
}


@pytest.fixture
def lb_config(tmp_path: Path) -> Path:
    cfg = {
        "providers": {
            # Free-tier cloud provider (non-local).
            "freecloud": {"base_url": "http://free.example/v1", "api_key": "k", "model_filter": None},
            # Two paid cloud providers, different prices.
            "paida": {"base_url": "http://paida.example/v1", "api_key": "k", "model_filter": None},
            "paidb": {"base_url": "http://paidb.example/v1", "api_key": "k", "model_filter": None},
            # Local provider (loopback) -> always $0 but its own tier.
            "localp": {"base_url": "http://localhost:11434/v1", "api_key": "k", "model_filter": None},
        },
        "believed_free": ["freecloud/free-1", "freecloud/free-2"],
        "model_reasoning": {
            # Local models differ in size/speed -> tag them so the local tier is
            # optimized per prompt just like the cloud tier.
            "localp/small": "exploratory",
            "localp/big": "deep",
        },
        "model_capabilities": {
            # Only a PAID model carries vision; free/local are untagged.
            "paida/vision": ["vision"],
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
def server(monkeypatch, lb_config):
    srv = _load_server_with_config(monkeypatch, lb_config)
    monkeypatch.setattr(srv, "load_pricing_map", lambda *a, **k: dict(_PRICING))
    _seed_route_cache(srv, {
        "freecloud__free-1": ("freecloud", "free-1"),
        "freecloud__free-2": ("freecloud", "free-2"),
        "localp__small": ("localp", "small"),
        "localp__big": ("localp", "big"),
        "paida__expensive": ("paida", "expensive"),
        "paida__vision": ("paida", "vision"),
        "paidb__cheap": ("paidb", "cheap"),
    })
    return srv


def _order(server, payload):
    """Return ordered [(provider, upstream), ...] for *payload*."""
    config = server.load_config()
    cands = server._get_loadbalanced_candidates()
    ordered = server._loadbalanced_ordered_candidates(cands, payload, config)
    return [(pn, um) for pn, _pc, um in ordered]


def _tiers(server, ordered_pairs):
    config = server.load_config()
    out = []
    for pn, um in ordered_pairs:
        cfg = server.get_provider(config, pn)
        out.append(server._cost_tier(pn, um, cfg, config))
    return out


# ── recognition / pool ──────────────────────────────────────────────────────

def test_is_loadbalanced_model(server):
    assert server._is_loadbalanced_model("llmproxy__loadbalanced")
    assert server._is_loadbalanced_model("llmproxy/loadbalanced")          # legacy
    assert not server._is_loadbalanced_model("llmproxy__free")
    assert server._is_virtual_model("llmproxy__loadbalanced")


def test_get_virtual_candidates_dispatches(server):
    new = server._get_virtual_candidates("llmproxy__loadbalanced")
    legacy = server._get_virtual_candidates("llmproxy/loadbalanced")
    assert {(pn, um) for pn, _, um in new}
    assert {(pn, um) for pn, _, um in new} == {(pn, um) for pn, _, um in legacy}


def test_pool_empty_when_route_cache_empty(server):
    _seed_route_cache(server, {})
    assert server._get_loadbalanced_candidates() == []


# ── cost waterfall ──────────────────────────────────────────────────────────

def test_plain_prompt_strict_cost_order(server):
    pairs = _order(server, {"messages": [{"role": "user", "content": "hi"}]})
    tiers = _tiers(server, pairs)
    # Non-decreasing: every free before every local before every paid.
    assert tiers == sorted(tiers), tiers
    assert set(tiers) == {server._TIER_FREE, server._TIER_LOCAL, server._TIER_PAID}


def test_paid_never_before_free_or_local_even_if_only_paid_is_capable(server):
    # Vision request: only paida/vision is tagged for vision, but cost must still
    # dominate — every free and local candidate is ordered ahead of any paid one.
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    pairs = _order(server, payload)
    tiers = _tiers(server, pairs)
    assert tiers == sorted(tiers), tiers
    first_paid = tiers.index(server._TIER_PAID)
    assert all(t < server._TIER_PAID for t in tiers[:first_paid])
    # Within the paid tier, the vision-capable model is ordered first.
    paid_pairs = [p for p, t in zip(pairs, tiers, strict=True) if t == server._TIER_PAID]
    assert paid_pairs[0] == ("paida", "vision")


def test_local_tier_best_first_regardless_of_prompt(server):
    # loadbalanced prefers the most capable $0 model, even for a short prompt:
    # the deep-tagged "big" local model is ordered ahead of the exploratory
    # "small" one in both cases (quality over size-fit).
    small = _order(server, {"messages": [{"role": "user", "content": "hi"}]})
    local_small = [(pn, um) for pn, um in small if pn == "localp"]
    assert local_small[0] == ("localp", "big")

    deep = _order(server, {
        "messages": [{"role": "user", "content": "think"}],
        "reasoning_effort": "high",
    })
    local_deep = [(pn, um) for pn, um in deep if pn == "localp"]
    assert local_deep[0] == ("localp", "big")


def test_paid_tier_cheapest_first(server):
    pairs = _order(server, {"messages": [{"role": "user", "content": "hi"}]})
    paid = [(pn, um) for pn, um in pairs if pn in ("paida", "paidb")]
    # paidb/cheap is an order of magnitude cheaper than paida/expensive.
    assert paid.index(("paidb", "cheap")) < paid.index(("paida", "expensive"))


def test_free_capacity_exhausted_is_deprioritized(server, monkeypatch):
    # Give free-1 a tiny per-minute limit and burn it; free-2 stays unlimited.
    cfg_path = Path(os.environ["LLMPROXY_CONFIG"])
    cfg = json.loads(cfg_path.read_text())
    cfg["free_limits"] = {"freecloud/free-1": {
        "requests_per_minute": 2, "requests_per_day": None,
        "tokens_per_minute": None, "tokens_per_day": None,
    }}
    cfg_path.write_text(json.dumps(cfg))

    server._get_or_create_tracker("freecloud/free-1").record(requests=2)
    pairs = _order(server, {"messages": [{"role": "user", "content": "hi"}]})
    free = [(pn, um) for pn, um in pairs if pn == "freecloud"]
    assert free.index(("freecloud", "free-2")) < free.index(("freecloud", "free-1"))


# ── provider free-tier headroom ("free in the moment") ──────────────────────

def test_provider_free_allowance_headroom(server):
    config = server.load_config()
    # A non-believed_free model on a provider with provider-wide free_allowance.
    prov_cfg = {"base_url": "http://allow.example/v1", "free_allowance": {
        "requests_per_minute": 3, "requests_per_day": None,
        "tokens_per_minute": None, "tokens_per_day": None,
    }}
    config["providers"]["allowance"] = prov_cfg
    _seed_route_cache(server, {"allowance__m": ("allowance", "m")})

    # Fresh: within the free allowance -> counts as FREE in the moment.
    assert server._provider_free_headroom("allowance", prov_cfg) is True
    assert server._cost_tier("allowance", "m", prov_cfg, config) == server._TIER_FREE

    # Burn the allowance -> falls back to PAID.
    server._get_or_create_tracker("allowance/m").record(requests=3)
    assert server._provider_free_headroom("allowance", prov_cfg) is False
    assert server._cost_tier("allowance", "m", prov_cfg, config) == server._TIER_PAID


def test_no_free_allowance_means_paid(server):
    config = server.load_config()
    prov_cfg = config["providers"]["paida"]
    assert server._provider_free_headroom("paida", prov_cfg) is False
    assert server._cost_tier("paida", "expensive", prov_cfg, config) == server._TIER_PAID


def test_local_model_is_local_tier_not_free(server):
    config = server.load_config()
    cfg = config["providers"]["localp"]
    assert server._cost_tier("localp", "small", cfg, config) == server._TIER_LOCAL


# ── best-first quality ordering (free tier) ─────────────────────────────────

def test_quality_key_orders_by_reasoning_then_size(server):
    rm = {"freecloud/deep-m": "deep", "freecloud/expl-m": "exploratory"}
    deep = server._quality_key("freecloud", "deep-m", rm)
    expl = server._quality_key("freecloud", "expl-m", rm)
    assert deep > expl                       # deep tier outranks exploratory
    # Untagged: inferred from the name (param count / known reasoning models).
    big = server._quality_key("p", "llama-70b", {})
    small = server._quality_key("p", "llama-7b", {})
    assert big > small                       # 70b standard > 7b exploratory
    r1 = server._quality_key("p", "deepseek-r1", {})
    assert r1[0] == server._REASONING_LEVELS.index("deep")


def test_free_tier_prefers_most_sophisticated_with_headroom(server):
    # Tag the two free models: free-1 deep, free-2 exploratory. With headroom on
    # both, the deep model must be tried first (best-first), deterministically.
    cfg_path = Path(os.environ["LLMPROXY_CONFIG"])
    cfg = json.loads(cfg_path.read_text())
    cfg["model_reasoning"] = {
        **cfg.get("model_reasoning", {}),
        "freecloud/free-1": "deep",
        "freecloud/free-2": "exploratory",
    }
    cfg_path.write_text(json.dumps(cfg))

    for _ in range(5):  # deterministic across runs (no random sampling)
        pairs = _order(server, {"messages": [{"role": "user", "content": "hi"}]})
        free = [(pn, um) for pn, um in pairs if pn == "freecloud"]
        assert free[0] == ("freecloud", "free-1")   # deep beats exploratory


def test_free_tier_exhausted_strong_model_falls_after_viable_weak(server):
    # free-1 is deep but rate-limited (no headroom); free-2 is exploratory but has
    # capacity. The viable weaker model is tried first; the exhausted strong one is
    # still present as a last-resort failover.
    cfg_path = Path(os.environ["LLMPROXY_CONFIG"])
    cfg = json.loads(cfg_path.read_text())
    cfg["model_reasoning"] = {
        **cfg.get("model_reasoning", {}),
        "freecloud/free-1": "deep",
        "freecloud/free-2": "exploratory",
    }
    cfg["free_limits"] = {"freecloud/free-1": {
        "requests_per_minute": 2, "requests_per_day": None,
        "tokens_per_minute": None, "tokens_per_day": None,
    }}
    cfg_path.write_text(json.dumps(cfg))
    server._get_or_create_tracker("freecloud/free-1").record(requests=2)

    pairs = _order(server, {"messages": [{"role": "user", "content": "hi"}]})
    free = [(pn, um) for pn, um in pairs if pn == "freecloud"]
    assert free.index(("freecloud", "free-2")) < free.index(("freecloud", "free-1"))
    assert ("freecloud", "free-1") in free       # still reachable for failover


def test_quality_ordering_preserves_cost_waterfall(server):
    # Even with best-first quality inside tiers, no paid model precedes a free or
    # local one.
    cfg_path = Path(os.environ["LLMPROXY_CONFIG"])
    cfg = json.loads(cfg_path.read_text())
    cfg["model_reasoning"] = {
        **cfg.get("model_reasoning", {}),
        "freecloud/free-2": "deep", "freecloud/free-1": "exploratory",
    }
    cfg_path.write_text(json.dumps(cfg))
    pairs = _order(server, {"messages": [{"role": "user", "content": "hi"}]})
    tiers = _tiers(server, pairs)
    assert tiers == sorted(tiers), tiers
