"""Flagship tier: an overlay above deep, with per-provider free semantics.

Flagship differs from the three ordinary tiers in two ways that these tests
pin down. It is an *overlay*: membership lives in config['flagship_models']
rather than config['model_reasoning'], so promoting a model does not remove it
from llmproxy/deep. And it is *per-provider*: the same weights may be free on
one provider and paid on another, so every provider's instance is its own
routing target and only the free ones reach flagship/free.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _load_server(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


def _seed_routes(server, routes: dict[str, tuple[str, str]]):
    with server._model_route_cache_lock:
        server._model_route_cache.clear()
        server._model_route_cache.update(routes)


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    """Two providers serving the SAME model, free on one and paid on the other."""
    data = {
        "providers": {
            "cheapo": {"base_url": "http://cheapo.example/v1", "api_key": "k"},
            "pricey": {"base_url": "http://pricey.example/v1", "api_key": "k"},
        },
        # Both instances are flagship; only cheapo's is free.
        "flagship_models": ["cheapo/glm-5.3", "pricey/glm-5.3"],
        "believed_free": ["cheapo/glm-5.3"],
        # The same model also carries an ordinary tier tag.
        "model_reasoning": {"cheapo/glm-5.3": "deep", "pricey/glm-5.3": "deep",
                            "cheapo/tiny": "exploratory"},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def server(monkeypatch, cfg):
    s = _load_server(monkeypatch, cfg)
    _seed_routes(s, {
        "cheapo__glm-5.3": ("cheapo", "glm-5.3"),
        "pricey__glm-5.3": ("pricey", "glm-5.3"),
        "cheapo__tiny": ("cheapo", "tiny"),
    })
    return s


# ── the overlay does not cannibalise the ordinary tiers ─────────────────────

def test_flagship_member_still_appears_in_deep(server):
    """The whole point of an overlay: promoting a model must not empty out
    llmproxy/deep, which an exclusive fourth tier would have done."""
    deep = {(pn, um) for pn, _, um in server._get_reasoning_model_candidates("deep")}
    flag = {(pn, um) for pn, _, um in server._get_reasoning_model_candidates("flagship")}
    assert flag, "fixture should have flagship members"
    assert flag <= deep, "every flagship member must still be reachable via deep"


def test_flagship_reads_membership_not_model_reasoning(server):
    """model_reasoning says 'deep' for both; flagship comes from its own set."""
    flag = {(pn, um) for pn, _, um in server._get_reasoning_model_candidates("flagship")}
    assert flag == {("cheapo", "glm-5.3"), ("pricey", "glm-5.3")}


def test_model_reasoning_rejects_overlay_levels(server, caplog):
    """An overlay tier set by hand in model_reasoning is ignored, with a warning,
    rather than creating a half-state alongside the computed set."""
    parsed = server._get_model_reasoning({"model_reasoning": {"p/m": "flagship"}})
    assert parsed == {}


# ── per-provider free semantics ─────────────────────────────────────────────

def test_every_provider_instance_is_its_own_routing_target(server):
    """A model served by two providers yields two candidates, so failover has
    somewhere to go."""
    flag = server._get_reasoning_model_candidates("flagship")
    assert sorted(pn for pn, _, _ in flag) == ["cheapo", "pricey"]


def test_flagship_free_takes_only_the_free_provider(server):
    """Free is per-provider: the same weights are free on cheapo and paid on
    pricey, so only cheapo's instance reaches flagship/free."""
    free = {(pn, um) for pn, _, um in server._get_reasoning_free_candidates("flagship")}
    assert free == {("cheapo", "glm-5.3")}


# ── ranking ─────────────────────────────────────────────────────────────────

def test_flagship_member_outranks_a_plain_deep_model(server):
    """Flagship is the top tier, so a member sorts above an untagged-but-deep
    peer even though both read 'deep' in model_reasoning."""
    rmap = {"pricey/glm-5.3": "deep", "cheapo/other": "deep"}
    flagship = {"pricey/glm-5.3"}
    promoted = server._quality_key("pricey", "glm-5.3", rmap, flagship)
    plain = server._quality_key("cheapo", "other", rmap, flagship)
    assert promoted[0] > plain[0]
    assert promoted[0] == server._REASONING_LEVELS.index("flagship")


def test_unknown_tier_falls_back_to_inference_not_rank_zero(server):
    """A tier this build does not know (e.g. written by a newer one) must not
    sort as the weakest candidate."""
    rank, _ = server._quality_key("p", "some-70b-model", {"p/some-70b-model": "unheard-of"}, set())
    assert rank == server._REASONING_LEVELS.index("standard")


def test_no_flagship_set_means_no_flagship_candidates(server):
    """With membership empty the tier is simply empty — it never falls back to
    deep, because a silent fallback would defeat asking for flagship."""
    assert server._get_flagship_models({"flagship_models": []}) == set()
    assert server._get_flagship_models({}) == set()
