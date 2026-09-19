"""The flagship selection algorithm (llmproxy/flagship.py).

Covers the four rules in their precedence order: benchmarks rank via combined
percentiles, spec gates veto, the bar floats until enough DISTINCT free models
qualify, and pins/excludes win.
"""

from __future__ import annotations

import pytest

from llmproxy.flagship import (
    Candidate,
    combine_scores,
    normalize_model_id,
    passes_spec_gate,
    select_flagship,
)

BASE = {
    "min_flagship_free_models": 0,
    "start_percentile": 0.9,
    "min_context": 200000,
    "require_tools": True,
    "max_models": None,
    "pin": [],
    "exclude": [],
}


def _c(provider, model, score=None, free=False, ctx=262144, tools=True):
    return Candidate(
        provider=provider, upstream_id=model, is_free=free,
        context_length=ctx, supports_tools=tools,
        scores={} if score is None else {"openrouter_aa": score},
    )


def _cfg(**over):
    return {**BASE, **over}


# ── joining ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("@cf/zai-org/glm-5.3", "z-ai/glm-5.3"),
    ("qwen/qwen3.8-27b", "qwen/qwen3.8-27b:free"),
    ("deepseek/deepseek-v4-flash-0731:free", "@cf/deepseek-ai/deepseek-v4-flash-0731"),
    ("meta-llama/llama-3.3-70b-instruct", "llama-3.3-70b"),
])
def test_same_weights_join_across_providers(a, b):
    assert normalize_model_id(a) == normalize_model_id(b)


@pytest.mark.parametrize("a,b", [
    ("z-ai/glm-5.3", "z-ai/glm-5.3-flash"),
    ("qwen/qwen3.8-27b", "qwen/qwen3.8-max-0902"),
    ("openai/gpt-5.6-sol", "openai/gpt-5.6-luna"),
])
def test_different_models_stay_distinct(a, b):
    assert normalize_model_id(a) != normalize_model_id(b)


def test_batch_variant_collapses_to_one_model():
    """Roughly half a raw catalog listing is :batch; counting those separately
    would inflate the tier about twofold."""
    cands = [_c("or", "x/m", 50), _c("or", "x/m:batch", 50)]
    assert len(combine_scores(cands)) == 1


# ── combining incompatible scales ───────────────────────────────────────────

def test_a_wider_scale_does_not_dominate_a_narrower_one():
    """Averaging raw scores would let Epoch's scale swamp AA's 0-100 indices.
    Here the two sources agree on order, so the combined order must follow the
    agreement rather than the magnitudes."""
    cands = [
        Candidate("p", "best", scores={"aa": 60, "epoch": 1000}),
        Candidate("p", "mid", scores={"aa": 50, "epoch": 600}),
        Candidate("p", "worst", scores={"aa": 40, "epoch": 100}),
    ]
    got = combine_scores(cands)
    assert got["best"] > got["mid"] > got["worst"]


def test_sources_that_perfectly_disagree_produce_a_tie():
    """The honest answer when one source ranks a model top and another ranks it
    bottom is 'no signal', not an ordering invented from the scales."""
    cands = [
        Candidate("p", "a", scores={"aa": 90, "epoch": 100}),
        Candidate("p", "b", scores={"aa": 50, "epoch": 900}),
        Candidate("p", "c", scores={"aa": 10, "epoch": 1000}),
    ]
    got = combine_scores(cands)
    assert len(set(got.values())) == 1


def test_a_model_missing_from_one_source_is_ranked_on_the_others():
    cands = [
        Candidate("p", "a", scores={"aa": 90}),
        Candidate("p", "b", scores={"aa": 10, "epoch": 5}),
    ]
    got = combine_scores(cands)
    assert got["a"] > got["b"]


def test_unscored_models_are_absent_rather_than_zero():
    got = combine_scores([_c("p", "scored", 10), _c("p", "unscored")])
    assert "unscored" not in got


# ── spec gates veto ─────────────────────────────────────────────────────────

def test_no_tool_support_is_vetoed_however_high_the_score():
    """glm-5.2:free ranked third among free models but cannot call tools."""
    cands = [_c("p", "strong-no-tools", 99, free=True, tools=False),
             _c("p", "weaker", 10, free=True)]
    sel = select_flagship(cands, _cfg(min_flagship_free_models=1))
    assert "p/strong-no-tools" not in sel.members


def test_short_context_is_vetoed():
    cands = [_c("p", "short", 99, free=True, ctx=32768),
             _c("p", "long", 10, free=True)]
    sel = select_flagship(cands, _cfg(min_flagship_free_models=1))
    assert "p/short" not in sel.members


def test_unknown_capability_data_fails_the_gate():
    """Admitting what we cannot verify would fill the tier with whatever a
    provider happens not to document."""
    assert not passes_spec_gate(
        Candidate("p", "m", context_length=None, supports_tools=None), 200000, True)


# ── the floating bar ────────────────────────────────────────────────────────

def _pool():
    """Eight paid models, then free ones interleaved further down."""
    out = [_c("paid", f"m{i}", 100 - i) for i in range(8)]
    out += [_c("cf", "free-a", 60, free=True),
            _c("cf", "free-b", 40, free=True),
            _c("cf", "free-c", 20, free=True)]
    return out


def test_bar_floats_until_enough_distinct_free_models_qualify():
    sel = select_flagship(_pool(), _cfg(min_flagship_free_models=2))
    assert len(sel.free_models) == 2
    assert "cf/free-a" in sel.members and "cf/free-b" in sel.members
    assert "cf/free-c" not in sel.members


def test_a_higher_free_floor_widens_the_tier():
    narrow = select_flagship(_pool(), _cfg(min_flagship_free_models=1))
    wide = select_flagship(_pool(), _cfg(min_flagship_free_models=3))
    assert len(wide.distinct_models) > len(narrow.distinct_models)


def test_cross_provider_duplicates_count_once_toward_the_floor():
    """Three providers serving the same weights is one model's worth of
    capability, so the floor must not be satisfied by duplication alone."""
    cands = [_c("paid", "top", 100),
             _c("a", "shared", 50, free=True),
             _c("b", "shared", 50, free=True),
             _c("c", "shared", 50, free=True),
             _c("d", "other-free", 10, free=True)]
    sel = select_flagship(cands, _cfg(min_flagship_free_models=2))
    assert len(sel.free_models) == 2
    assert "d/other-free" in sel.members


def test_duplicates_are_still_separate_routing_targets():
    """Counting once must not collapse them in the output: each provider has
    its own quota and outage profile, which is what failover needs."""
    cands = [_c("a", "shared", 50, free=True), _c("b", "shared", 50, free=True)]
    sel = select_flagship(cands, _cfg(min_flagship_free_models=1))
    assert {"a/shared", "b/shared"} <= set(sel.members)
    assert len(sel.free_models) == 1


def test_free_floor_is_best_effort_when_the_pool_is_too_small():
    """No lower bound: if only two free models exist, asking for five yields
    two rather than erroring or looping."""
    cands = [_c("p", "x", 90), _c("f", "a", 50, free=True), _c("f", "b", 40, free=True)]
    sel = select_flagship(cands, _cfg(min_flagship_free_models=5))
    assert len(sel.free_models) == 2


def test_max_models_caps_the_tier():
    sel = select_flagship(_pool(), _cfg(min_flagship_free_models=3, max_models=4))
    assert len(sel.distinct_models) == 4


# ── pins and excludes ───────────────────────────────────────────────────────

def test_pin_bypasses_both_the_bar_and_the_veto():
    """The only way to admit a provider no benchmark covers, such as a direct
    provider absent from every leaderboard."""
    cands = [_c("p", "top", 100), _c("atria-asi", "Atria-Dawn-Preview",
                                     None, ctx=None, tools=None)]
    sel = select_flagship(cands, _cfg(pin=["atria-asi/Atria-Dawn-Preview"]))
    assert "atria-asi/atria-dawn-preview" in sel.members


def test_an_unmatched_pin_is_honoured_but_reported():
    sel = select_flagship([_c("p", "top", 100)], _cfg(pin=["ghost/model"]))
    assert "ghost/model" in sel.members
    assert sel.unverified_pins == ["ghost/model"]


def test_exclude_beats_a_qualifying_model():
    sel = select_flagship([_c("p", "top", 100)], _cfg(exclude=["p/top"]))
    assert "p/top" not in sel.members


def test_exclude_beats_a_pin():
    sel = select_flagship([_c("p", "top", 100)],
                          _cfg(pin=["p/top"], exclude=["p/top"]))
    assert "p/top" not in sel.members


# ── degenerate input ────────────────────────────────────────────────────────

def test_no_candidates_yields_an_empty_tier_not_an_error():
    sel = select_flagship([], _cfg(min_flagship_free_models=5))
    assert sel.members == [] and sel.bar is None


def test_no_free_models_still_produces_a_paid_tier():
    sel = select_flagship([_c("p", f"m{i}", 100 - i) for i in range(5)],
                          _cfg(min_flagship_free_models=5))
    assert sel.members and sel.free_models == []
