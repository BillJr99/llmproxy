"""Flagship tier: an overlay above deep, with per-provider free semantics.

Flagship differs from the three ordinary tiers in three ways that these tests
pin down. It is an *overlay*: membership sits beside model_reasoning rather
than in it, so promoting a model does not remove it from llmproxy/deep. It is
*per-provider*: the same weights may be free on one provider and paid on
another, so every provider's instance is its own routing target and only the
free ones reach flagship/free. And it is *never hardcoded*: membership depends
on which providers a deployment has configured, so it is computed locally into
flagship_models.json beside config.json, while the user's config holds only
the pin/exclude policy.
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
        "believed_free": ["cheapo/glm-5.3"],
        # The same model also carries an ordinary tier tag.
        "model_reasoning": {"cheapo/glm-5.3": "deep", "pricey/glm-5.3": "deep",
                            "cheapo/tiny": "exploratory"},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    # Membership is machine state in a sibling cache file, not config.
    (tmp_path / "flagship_models.json").write_text(json.dumps({
        "last_refresh_at": "2026-09-19T00:00:00+00:00",
        "bar": 41.7,
        "members": ["cheapo/glm-5.3", "pricey/glm-5.3"],
    }), encoding="utf-8")
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
    """model_reasoning says 'deep' for both; flagship comes from the cache."""
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


def test_no_cache_and_no_pins_means_an_empty_tier(server, tmp_path):
    """With no cache file and no pins the tier is simply empty — it never falls
    back to deep, because a silent fallback would defeat asking for flagship."""
    empty = tmp_path / "elsewhere" / "config.json"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text("{}", encoding="utf-8")
    assert server._get_flagship_models({}, str(empty)) == set()


# ── membership is cached state, policy is config ────────────────────────────

def test_membership_is_not_read_from_the_user_config(server, cfg):
    """A stray flagship_models key in config.json must not be honoured: the
    list is deployment-specific machine state, not something to hand-edit."""
    assert server._get_flagship_models({"flagship_models": ["cheapo/invented"]}) \
        == {"cheapo/glm-5.3", "pricey/glm-5.3"}


def test_pin_takes_effect_without_waiting_for_a_refresh(server):
    """Pins are applied at read time, so pinning a model works immediately
    rather than on the next cadence tick."""
    got = server._get_flagship_models({"flagship_tier": {"pin": ["atria-asi/Atria-Dawn-Preview"]}})
    assert "atria-asi/atria-dawn-preview" in got
    assert "cheapo/glm-5.3" in got  # cached members survive


def test_exclude_beats_a_cached_member(server):
    """Excludes are applied last, so they win over computed membership."""
    got = server._get_flagship_models({"flagship_tier": {"exclude": ["pricey/glm-5.3"]}})
    assert got == {"cheapo/glm-5.3"}


def test_exclude_beats_pin(server):
    """When a model is both pinned and excluded, exclude wins."""
    got = server._get_flagship_models({"flagship_tier": {
        "pin": ["x/y"], "exclude": ["x/y"],
    }})
    assert "x/y" not in got


def test_membership_cache_lives_beside_the_config(tmp_path):
    from llmproxy.config import get_flagship_state_path
    cfg = str(tmp_path / "config.json")
    assert get_flagship_state_path(cfg) == tmp_path / "flagship_models.json"


# ── config block defaults ───────────────────────────────────────────────────

def test_defaults_apply_to_a_config_that_predates_the_block():
    """The tier must behave identically whether or not the user has pasted the
    flagship_tier block in, so an upgrade needs no config edit."""
    from llmproxy.config import FLAGSHIP_TIER_DEFAULTS, flagship_tier_cfg
    assert flagship_tier_cfg({}) == FLAGSHIP_TIER_DEFAULTS
    assert flagship_tier_cfg({"flagship_tier": {}}) == FLAGSHIP_TIER_DEFAULTS


def test_user_values_override_defaults_key_by_key():
    from llmproxy.config import flagship_tier_cfg
    got = flagship_tier_cfg({"flagship_tier": {"min_flagship_free_models": 2}})
    assert got["min_flagship_free_models"] == 2
    assert got["refresh_frequency_days"] == 7  # untouched keys keep defaults


def test_config_example_matches_the_code_defaults():
    """CI regenerates config.example.json and diffs it, so these must agree or
    the committed example drifts from what the server actually does."""
    import json
    from pathlib import Path

    from llmproxy.config import FLAGSHIP_TIER_DEFAULTS
    root = Path(__file__).resolve().parent.parent
    example = json.loads((root / "config.example.json").read_text(encoding="utf-8"))
    assert example["flagship_tier"] == FLAGSHIP_TIER_DEFAULTS


# ── refresh cadence ─────────────────────────────────────────────────────────

def _iso(days_ago: float) -> str:
    from datetime import UTC, datetime, timedelta
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


def test_fresh_deployment_is_due(server, tmp_path):
    """No cache yet, so the tier populates on first boot rather than staying
    empty until a week has passed."""
    fresh = tmp_path / "fresh" / "config.json"
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_text("{}", encoding="utf-8")
    from llmproxy.config import FLAGSHIP_TIER_DEFAULTS
    assert server._flagship_refresh_due(FLAGSHIP_TIER_DEFAULTS, str(fresh)) is True


def test_recent_refresh_is_throttled(server, tmp_path):
    """A restart inside the window must not re-scrape every provider."""
    from llmproxy.config import FLAGSHIP_TIER_DEFAULTS, save_flagship_state
    cfg = str(tmp_path / "throttled" / "config.json")
    (tmp_path / "throttled").mkdir(parents=True, exist_ok=True)
    save_flagship_state({"last_refresh_at": _iso(1.0)}, cfg)
    assert server._flagship_refresh_due(FLAGSHIP_TIER_DEFAULTS, cfg) is False


def test_stale_refresh_is_due(server, tmp_path):
    from llmproxy.config import FLAGSHIP_TIER_DEFAULTS, save_flagship_state
    cfg = str(tmp_path / "stale" / "config.json")
    (tmp_path / "stale").mkdir(parents=True, exist_ok=True)
    save_flagship_state({"last_refresh_at": _iso(9.0)}, cfg)
    assert server._flagship_refresh_due(FLAGSHIP_TIER_DEFAULTS, cfg) is True


def test_zero_frequency_always_refreshes(server, tmp_path):
    from llmproxy.config import flagship_tier_cfg, save_flagship_state
    cfg = str(tmp_path / "always" / "config.json")
    (tmp_path / "always").mkdir(parents=True, exist_ok=True)
    save_flagship_state({"last_refresh_at": _iso(0.0)}, cfg)
    tier = flagship_tier_cfg({"flagship_tier": {"refresh_frequency_days": 0}})
    assert server._flagship_refresh_due(tier, cfg) is True


def test_disabled_tier_never_refreshes(server, tmp_path):
    """enabled: false is a master switch — no network, no recompute."""
    from llmproxy.config import flagship_tier_cfg
    cfg = str(tmp_path / "off" / "config.json")
    (tmp_path / "off").mkdir(parents=True, exist_ok=True)
    tier = flagship_tier_cfg({"flagship_tier": {"enabled": False}})
    assert server._flagship_refresh_due(tier, cfg) is False


# ── end-to-end recompute ────────────────────────────────────────────────────

def test_recompute_writes_membership_from_the_route_cache(server, cfg, monkeypatch):
    """The candidate pool is every model of every configured provider, and the
    result lands in the cache file rather than the user's config."""
    import json as _json

    from llmproxy import flagship as _flagship
    from llmproxy.config import get_flagship_state_path, load_flagship_state

    # Two providers serve the same weights; only cheapo's is free.
    monkeypatch.setattr(_flagship, "fetch_profiles", lambda sources=None: {
        _flagship.normalize_model_id("glm-5.3"): {
            "scores": {"openrouter_aa": 53.4},
            "context_length": 1_310_720, "supports_tools": True,
        },
        _flagship.normalize_model_id("tiny"): {
            "scores": {"openrouter_aa": 1.0},
            "context_length": 8192, "supports_tools": False,
        },
    })
    state = server._recompute_flagship_members(server.load_config(), str(cfg))

    assert state is not None
    assert set(state["members"]) == {"cheapo/glm-5.3", "pricey/glm-5.3"}
    assert state["candidates_considered"] == 3
    # tiny is vetoed on both context and tools despite being in the pool.
    assert not any("tiny" in m for m in state["members"])
    # Written to the sibling cache, not the config.
    assert get_flagship_state_path(str(cfg)).exists()
    assert "flagship_models" not in _json.loads(cfg.read_text(encoding="utf-8"))
    assert load_flagship_state(str(cfg))["members"] == state["members"]


def test_recompute_keeps_previous_membership_when_every_source_fails(server, cfg, monkeypatch):
    """One unreachable leaderboard must degrade the ranking, never empty the
    tier — a 503 on llmproxy/flagship is worse than a slightly stale list."""
    from llmproxy import flagship as _flagship
    from llmproxy.config import load_flagship_state

    before = load_flagship_state(str(cfg))["members"]
    monkeypatch.setattr(_flagship, "fetch_profiles", lambda sources=None: {})
    assert server._recompute_flagship_members(server.load_config(), str(cfg)) is None
    assert load_flagship_state(str(cfg))["members"] == before
