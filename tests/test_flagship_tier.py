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

It is also the only tier with a measured per-model ranking, so it is the only
one that is *ordered* rather than rotated or load-spread: the pool is walked
strongest-first and failover descends it in rank order.
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


# ── benchmark-ranked ordering ───────────────────────────────────────────────
#
# Membership says who is in the tier; the scores say how strong each member is.
# These pin the second half down: the pool must be walked strongest-first, with
# capacity and health demoting rather than reordering, so failover descends the
# ranking instead of sampling it.


@pytest.fixture
def ranked_cfg(tmp_path: Path) -> Path:
    """Four routing targets whose benchmark order contradicts every proxy for it.

    ``small-ace`` is the strongest model and the smallest; ``huge-dud`` is the
    weakest and, at 405B, the one _param_count would promote. ``twin`` is the
    same weights on two providers, so it exercises the cross-provider tie.
    """
    data = {
        "providers": {
            "alpha": {"base_url": "http://alpha.example/v1", "api_key": "k"},
            "beta": {"base_url": "http://beta.example/v1", "api_key": "k"},
        },
        "believed_free": ["alpha/small-ace", "alpha/huge-405b-dud",
                          "alpha/twin", "beta/twin"],
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "flagship_models.json").write_text(json.dumps({
        "last_refresh_at": "2026-09-19T00:00:00+00:00",
        "bar": 0.5,
        "members": ["alpha/small-ace", "alpha/huge-405b-dud",
                    "alpha/twin", "beta/twin"],
        "scores": {
            "alpha/small-ace": {"combined": 0.99, "model_key": "smallace"},
            "alpha/huge-405b-dud": {"combined": 0.51, "model_key": "huge405bdud"},
            "alpha/twin": {"combined": 0.80, "model_key": "twin"},
            "beta/twin": {"combined": 0.80, "model_key": "twin"},
        },
        "model_scores": {"smallace": 0.99, "huge405bdud": 0.51, "twin": 0.80},
    }), encoding="utf-8")
    return p


@pytest.fixture
def ranked_server(monkeypatch, ranked_cfg):
    s = _load_server(monkeypatch, ranked_cfg)
    _seed_routes(s, {
        "alpha__huge-405b-dud": ("alpha", "huge-405b-dud"),
        "beta__twin": ("beta", "twin"),
        "alpha__small-ace": ("alpha", "small-ace"),
        "alpha__twin": ("alpha", "twin"),
    })
    s._reset_usage()
    yield s
    s._reset_usage()


def _order(server, model="llmproxy__flagship"):
    cands = server._get_virtual_candidates(model)
    ordered = server._flagship_ordered_candidates(
        cands, server._get_flagship_scores(), {})
    return [(pn, um) for pn, _pc, um in ordered]


def test_the_pool_is_walked_strongest_first(ranked_server):
    scores = ranked_server._get_flagship_scores()
    ranks = [ranked_server._flagship_candidate_score(pn, um, scores)
             for pn, um in _order(ranked_server)]
    assert ranks == sorted(ranks, reverse=True)
    assert ranks[0] == 0.99


def test_score_beats_size_which_is_the_signal_it_replaces(ranked_server):
    """_param_count would put the 405B model first; the benchmark puts it last."""
    order = _order(ranked_server)
    assert order[0] == ("alpha", "small-ace")
    assert order[-1] == ("alpha", "huge-405b-dud")


def test_a_cross_provider_tie_is_broken_deterministically(ranked_server):
    """The same weights share one score, so nothing about the ranking separates
    them — the order must still be stable rather than route-cache-dependent."""
    runs = {tuple(_order(ranked_server)) for _ in range(5)}
    assert len(runs) == 1
    twins = [c for c in _order(ranked_server) if c[1] == "twin"]
    assert twins == [("alpha", "twin"), ("beta", "twin")]


def test_a_saturated_top_pick_is_demoted_but_stays_reachable(ranked_server):
    """A strict order would re-hammer a rate-limited leader on every request;
    dropping it to the tail keeps the pool self-healing without a 503."""
    ranked_server._mark_saturated(
        ranked_server._usage_key("alpha", "small-ace", None))
    order = _order(ranked_server)
    assert order[0] != ("alpha", "small-ace")
    assert ("alpha", "small-ace") in order       # still a last-resort candidate
    assert order[0] == ("alpha", "twin")          # next-strongest with headroom


def test_an_unscored_member_sorts_after_every_scored_one(ranked_server, monkeypatch):
    """An unscraped pin cannot be ranked on evidence, so it must not preempt a
    model that was actually measured."""
    _seed_routes(ranked_server, {
        "alpha__small-ace": ("alpha", "small-ace"),
        "alpha__ghost": ("alpha", "ghost"),
        "alpha__huge-405b-dud": ("alpha", "huge-405b-dud"),
    })
    monkeypatch.setattr(ranked_server, "_get_flagship_models",
                        lambda *a, **k: {"alpha/small-ace", "alpha/ghost",
                                         "alpha/huge-405b-dud"})
    assert _order(ranked_server)[-1] == ("alpha", "ghost")


def test_flagship_free_is_ranked_not_capacity_sampled(ranked_server):
    """flagship/free belongs to _FREE_VIRTUAL_MODELS too, so the flagship branch
    has to be tested first or this pool would never reach the ranking."""
    assert ranked_server._is_flagship_virtual_model("llmproxy__flagship/free")
    assert _order(ranked_server, "llmproxy__flagship/free")[0] == ("alpha", "small-ace")


def test_every_flagship_entry_point_is_recognised(ranked_server):
    for name in ("llmproxy__flagship", "llmproxy/flagship",
                 "llmproxy__flagship/free", "llmproxy/flagship/free",
                 "llmproxy__flagship/local", "llmproxy/flagship/local",
                 "llmproxy__alpha/flagship"):
        assert ranked_server._is_flagship_virtual_model(name), name
    for name in ("llmproxy__deep", "llmproxy__deep/free", "llmproxy__free",
                 "llmproxy__loadbalanced", "llmproxy__alpha/deep"):
        assert not ranked_server._is_flagship_virtual_model(name), name


def test_request_fit_never_reorders_a_ranked_pool(ranked_server):
    """The regression this change is most likely to suffer: _order_by_request_fit
    keys on _param_count once the tier term is constant, which would invert the
    ranking for flagship/free and flagship/local."""
    ordered = ranked_server._flagship_ordered_candidates(
        ranked_server._get_virtual_candidates("llmproxy__flagship/free"),
        ranked_server._get_flagship_scores(), {})
    # A large prompt targets `deep`, which is what biases request-fit to size.
    payload = {"messages": [{"role": "user", "content": "x" * 400000}]}
    refit = ranked_server._order_by_request_fit(ordered, payload, {})
    assert [c[2] for c in refit][0] == "huge-405b-dud", (
        "guard premise broken: request-fit no longer promotes the big model")
    # The dispatcher must therefore skip it; the ranked order is what ships.
    assert [c[2] for c in ordered][0] == "small-ace"


def test_a_cache_without_scores_leaves_the_pool_untouched(server):
    """The degradation path: a membership file written before scores existed
    must keep the previous behaviour rather than sort everything as unscored."""
    assert server._get_flagship_scores() == {}
    cands = server._get_virtual_candidates("llmproxy__flagship")
    assert server._flagship_ordered_candidates(cands, {}, {}) is cands


def test_recompute_persists_the_scores_it_ranked_on(server, cfg, monkeypatch):
    """Membership alone is not enough to order the tier, so the refresh has to
    write the percentiles down beside it."""
    from llmproxy import flagship as _flagship
    from llmproxy.config import load_flagship_state

    monkeypatch.setattr(_flagship, "fetch_profiles", lambda sources=None: {
        _flagship.normalize_model_id("glm-5.3"): {
            "scores": {"openrouter_aa": 53.4},
            "context_length": 1_310_720, "supports_tools": True,
        },
    })
    state = server._recompute_flagship_members(server.load_config(), str(cfg))

    assert set(state["scores"]) == {"cheapo/glm-5.3", "pricey/glm-5.3"}
    # One model on two providers: identical weights rank identically.
    assert (state["scores"]["cheapo/glm-5.3"]["combined"]
            == state["scores"]["pricey/glm-5.3"]["combined"])
    assert state["model_scores"]["glm53"] == state["scores"]["cheapo/glm-5.3"]["combined"]
    assert load_flagship_state(str(cfg))["scores"] == state["scores"]


def test_a_pinned_provider_inherits_the_score_of_the_same_weights(ranked_server):
    """Scores are per model, not per provider, so a provider absent from the
    cache still ranks when its weights were scored under another listing."""
    scores = ranked_server._get_flagship_scores()
    assert ranked_server._flagship_candidate_score("newbie", "twin", scores) == 0.80
    assert ranked_server._flagship_candidate_score("newbie", "@cf/org/twin", scores) == 0.80
    assert ranked_server._flagship_candidate_score("newbie", "nothing", scores) is None


def test_a_malformed_scores_block_is_ignored_not_fatal(tmp_path, monkeypatch):
    """config.json is hand-edited and the cache file sits beside it, so a bad
    shape must cost an ordering rather than every flagship request."""
    (tmp_path / "config.json").write_text(json.dumps({"providers": {}}), encoding="utf-8")
    (tmp_path / "flagship_models.json").write_text(json.dumps({
        "members": ["a/b"],
        "scores": "not-a-dict",
        "model_scores": {"ok": 0.5, "bad": "x", "alsobad": None, "flag": True},
    }), encoding="utf-8")
    s = _load_server(monkeypatch, tmp_path / "config.json")
    assert s._get_flagship_scores() == {"ok": 0.5}


# ── route provenance ────────────────────────────────────────────────────────
#
# A pick that cannot be explained afterwards is a pick nobody can debug, so the
# ordering that produced it has to name itself on the response.

def _dispatch(server, monkeypatch, model):
    """Drive one request through the real dispatcher, capturing what it built."""
    from flask import Response

    captured: dict = {}

    def _fake_cycle(endpoint, model_full, ordered, payload, timeout,
                    on_success=None, route_reason=None, **kwargs):
        captured["ordered"] = [(pn, um) for pn, _pc, um in ordered]
        captured["route_reason"] = route_reason
        return Response(b'{"ok": true}', status=200, content_type="application/json")

    monkeypatch.setattr(server, "_proxy_cycling_non_streaming", _fake_cycle)
    monkeypatch.setattr(server, "_sync_local_provider_models_once", lambda: None)
    resp = server.app.test_client().post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    return captured


@pytest.mark.parametrize("model", ["llmproxy__flagship", "llmproxy__flagship/free"])
def test_a_ranked_pick_names_the_ranking_that_produced_it(
        ranked_server, monkeypatch, model):
    got = _dispatch(ranked_server, monkeypatch, model)
    assert got["route_reason"].startswith("flagship_rank=4/4")
    assert got["ordered"][0] == ("alpha", "small-ace")


def test_an_unranked_pool_reports_the_ordering_it_actually_used(server, monkeypatch):
    """No scores cached: claiming a ranking we do not have would be worse than
    the stale behaviour itself."""
    got = _dispatch(server, monkeypatch, "llmproxy__flagship")
    assert "flagship_rank" not in got["route_reason"]
    assert got["route_reason"].startswith(server.ROUTE_SOURCE_CYCLING)


def test_a_partly_scored_pool_says_how_much_of_it_was_ranked(
        ranked_server, monkeypatch):
    """The tag carries ranked/total so a pinned, unscorable member is visible
    from the header instead of having to be inferred."""
    _seed_routes(ranked_server, {
        "alpha__small-ace": ("alpha", "small-ace"),
        "alpha__ghost": ("alpha", "ghost"),
    })
    monkeypatch.setattr(ranked_server, "_get_flagship_models",
                        lambda *a, **k: {"alpha/small-ace", "alpha/ghost"})
    got = _dispatch(ranked_server, monkeypatch, "llmproxy__flagship")
    assert got["route_reason"].startswith("flagship_rank=1/2")
    assert got["ordered"] == [("alpha", "small-ace"), ("alpha", "ghost")]


def test_the_models_endpoint_shows_the_real_failover_order(ranked_server):
    """The documented way to inspect a pool must agree with what it will do."""
    body = ranked_server.app.test_client().get(
        "/v1/models/llmproxy__flagship").get_json()
    assert body["_candidates"][0] == "alpha/small-ace"
    assert body["_candidates"][-1] == "alpha/huge-405b-dud"
