"""Aggregation rules in scripts/update_free_models.py.

Tests the contract of `aggregate(evidence, sidecar, catalog_succeeded)`:
- High-confidence positive adds a model.
- High-confidence negative blocks an add and forces a remove.
- Low-confidence alone never adds or removes.
- Models absent from a successful catalog response are flagged for removal.
- Limits are merged from the highest-confidence non-empty record.
"""

from __future__ import annotations

from scripts.sources.base import Evidence
from scripts.update_free_models import (
    CATALOG_MIN_MODELS,
    aggregate,
    catalog_source_names,
)


def _sidecar(provider: str, current_free: list[str]) -> dict:
    return {
        "providers": {
            provider: {
                "believed_free": current_free,
                "model_reasoning": {},
                "free_limits": {},
                "base_url": "u",
                "display": "X",
            }
        },
        "provider_order": [provider],
    }


def test_high_confidence_positive_adds_model():
    ev = [Evidence(provider="p", model_id="p/new", is_free=True, source="docs",
                   confidence="high", url="u")]
    out = aggregate(ev, _sidecar("p", []), catalog_succeeded=set())
    assert out["p"]["add"] == ["p/new"]
    assert out["p"]["remove"] == []


def test_high_confidence_negative_blocks_add():
    ev = [
        Evidence(provider="p", model_id="p/x", is_free=True, source="docs",
                 confidence="high", url="u"),
        Evidence(provider="p", model_id="p/x", is_free=False, source="docs2",
                 confidence="high", url="u"),
    ]
    out = aggregate(ev, _sidecar("p", []), catalog_succeeded=set())
    assert out["p"]["add"] == []


def test_low_confidence_alone_does_not_add():
    ev = [Evidence(provider="p", model_id="p/x", is_free=True, source="community",
                   confidence="low", url="u")]
    out = aggregate(ev, _sidecar("p", []), catalog_succeeded=set())
    assert out["p"]["add"] == []


def test_high_confidence_negative_removes_existing():
    ev = [Evidence(provider="p", model_id="p/old", is_free=False, source="docs",
                   confidence="high", url="u")]
    out = aggregate(ev, _sidecar("p", ["p/old"]), catalog_succeeded=set())
    assert out["p"]["remove"] == ["p/old"]


def test_absence_triggers_remove_only_when_catalog_succeeded():
    """Models absent from a successful catalog fetch should be removed, but
    only if a catalog source actually ran cleanly for that provider."""
    # No catalog evidence at all — must NOT remove
    out = aggregate([], _sidecar("p", ["p/maybe"]), catalog_succeeded=set())
    assert out["p"]["remove"] == []

    # API succeeded for p, but p/maybe wasn't in the response → remove
    ev = [Evidence(provider="p", model_id="p/other", is_free=None, source="api",
                   confidence="medium", url="u")]
    out = aggregate(ev, _sidecar("p", ["p/maybe", "p/other"]), catalog_succeeded={"p"})
    assert out["p"]["remove"] == ["p/maybe"]
    assert "p/other" not in out["p"]["remove"]


def test_limits_taken_from_high_confidence_source():
    high_lim = {"requests_per_minute": 30, "requests_per_day": 1000,
                "tokens_per_minute": None, "tokens_per_day": None}
    low_lim = {"requests_per_minute": 99, "requests_per_day": 99,
               "tokens_per_minute": None, "tokens_per_day": None}
    ev = [
        Evidence(provider="p", model_id="p/m", is_free=True, source="community",
                 confidence="low", url="u", limits=low_lim),
        Evidence(provider="p", model_id="p/m", is_free=True, source="docs",
                 confidence="high", url="u", limits=high_lim),
    ]
    out = aggregate(ev, _sidecar("p", []), catalog_succeeded=set())
    assert out["p"]["limits"]["p/m"]["requests_per_minute"] == 30


# ── pricing merge ───────────────────────────────────────────────────────────

def test_pricing_taken_from_high_confidence_source():
    low = {"input_cost_per_token": 9e-7, "output_cost_per_token": 9e-7}
    high = {"input_cost_per_token": 1e-7, "output_cost_per_token": 2e-7}
    ev = [
        Evidence(provider="p", model_id="p/paid", is_free=False, source="community",
                 confidence="low", url="u", pricing=low),
        Evidence(provider="p", model_id="p/paid", is_free=False, source="openrouter",
                 confidence="high", url="u", pricing=high),
    ]
    out = aggregate(ev, _sidecar("p", []), catalog_succeeded=set())
    assert out["p"]["pricing"]["p/paid"] == high


def test_zero_pricing_is_not_emitted():
    # A zero-cost record belongs in believed_free, never the paid pricing block.
    ev = [Evidence(provider="p", model_id="p/free", is_free=True, source="openrouter",
                   confidence="high", url="u",
                   pricing={"input_cost_per_token": 0.0, "output_cost_per_token": 0.0})]
    out = aggregate(ev, _sidecar("p", []), catalog_succeeded=set())
    assert out["p"]["pricing"] == {}


def test_pricing_skipped_for_believed_free_model():
    # Even if a source reports a price, a model that is (becoming) free carries no
    # paid price.
    ev = [Evidence(provider="p", model_id="p/m", is_free=True, source="openrouter",
                   confidence="high", url="u",
                   pricing={"input_cost_per_token": 1e-7, "output_cost_per_token": 1e-7})]
    out = aggregate(ev, _sidecar("p", ["p/m"]), catalog_succeeded=set())
    assert out["p"]["pricing"] == {}


# ── absence-based removal via catalog-enumerating sources ───────────────────

def test_openrouter_absence_removes_withdrawn_model():
    """A cloaked model that has left OpenRouter's catalog must leave
    believed_free. OpenRouter enumerates the whole catalog, so silence about a
    model is evidence of withdrawal rather than an absence of opinion."""
    ev = [
        Evidence(provider="p", model_id="p/still-here", is_free=True,
                 source="openrouter", confidence="high", url="u"),
        Evidence(provider="p", model_id="p/also-here", is_free=True,
                 source="openrouter", confidence="high", url="u"),
    ]
    sidecar = _sidecar("p", ["p/still-here", "p/also-here", "p/withdrawn"])
    out = aggregate(ev, sidecar, catalog_succeeded={"p"})
    assert out["p"]["remove"] == ["p/withdrawn"]


def test_docs_only_provider_never_removes_by_absence():
    """Docs scrapers enumerate a free-tier page, not a catalog. Their silence
    must never remove anything, even when the provider is listed as succeeded."""
    ev = [Evidence(provider="p", model_id="p/documented", is_free=True,
                   source="docs", confidence="high", url="u")]
    sidecar = _sidecar("p", ["p/documented", "p/undocumented"])
    out = aggregate(ev, sidecar, catalog_succeeded=set())
    assert out["p"]["remove"] == []


def test_partial_catalog_response_suppresses_removal():
    """A degraded fetch that lists almost nothing must not empty believed_free.
    One model out of six retained is below the floor, so absence is distrusted."""
    ev = [Evidence(provider="p", model_id="p/m1", is_free=True,
                   source="openrouter", confidence="high", url="u")]
    current = [f"p/m{i}" for i in range(1, 7)]
    out = aggregate(ev, _sidecar("p", current), catalog_succeeded={"p"})
    assert out["p"]["remove"] == []


def test_mostly_intact_catalog_response_still_removes():
    """The floor must not block the normal case: a small provider whose catalog
    still covers most of believed_free can drop the one model that vanished."""
    ev = [
        Evidence(provider="p", model_id=f"p/m{i}", is_free=True,
                 source="openrouter", confidence="high", url="u")
        for i in range(1, 6)
    ]
    current = [f"p/m{i}" for i in range(1, 7)]
    out = aggregate(ev, _sidecar("p", current), catalog_succeeded={"p"})
    assert out["p"]["remove"] == ["p/m6"]


def test_large_catalog_removes_even_when_free_tier_is_withdrawn_wholesale():
    """A provider that retires its entire free tier at once is a real event. A
    substantial catalog response is trusted on its own, without needing any of
    believed_free to survive."""
    ev = [
        Evidence(provider="p", model_id=f"p/paid{i}", is_free=False,
                 source="openrouter", confidence="high", url="u")
        for i in range(CATALOG_MIN_MODELS)
    ]
    out = aggregate(ev, _sidecar("p", ["p/was-free"]), catalog_succeeded={"p"})
    assert out["p"]["remove"] == ["p/was-free"]


def test_catalog_source_names_covers_openrouter_and_api_only():
    """The catalog set is derived from the enumerates_catalog flag. Docs and
    community sources must stay out of it, or their silence would delete."""
    names = catalog_source_names()
    assert {"openrouter", "api"} <= names
    assert "docs" not in names
    assert "community" not in names
    assert "endpoint_probe" not in names


# ── withdrawn models lose their routing metadata ────────────────────────────

def _tagged_sidecar(current_free: list[str], reasoning: dict) -> dict:
    sidecar = _sidecar("p", current_free)
    sidecar["providers"]["p"]["model_reasoning"] = dict(reasoning)
    return sidecar


def test_withdrawn_model_is_reported_separately_from_repriced():
    """A repriced model keeps its tier tag because loadbalanced still ranks it;
    a withdrawn one has nothing left to route to."""
    ev = [
        # Still listed, but now priced — a removal, not a withdrawal.
        Evidence(provider="p", model_id="p/repriced", is_free=False,
                 source="openrouter", confidence="high", url="u"),
    ] + [
        Evidence(provider="p", model_id=f"p/other{i}", is_free=False,
                 source="openrouter", confidence="high", url="u")
        for i in range(CATALOG_MIN_MODELS)
    ]
    sidecar = _tagged_sidecar(
        ["p/repriced", "p/withdrawn"],
        {"p/repriced": "deep", "p/withdrawn": "exploratory"},
    )
    out = aggregate(ev, sidecar, catalog_succeeded={"p"})
    assert out["p"]["remove"] == ["p/repriced", "p/withdrawn"]
    assert out["p"]["withdrawn"] == ["p/withdrawn"]


def test_withdrawn_includes_tags_for_models_never_in_believed_free():
    """Tags left behind by earlier sweeps are collected too, so dead routing
    metadata does not accumulate indefinitely."""
    ev = [
        Evidence(provider="p", model_id=f"p/live{i}", is_free=False,
                 source="openrouter", confidence="high", url="u")
        for i in range(CATALOG_MIN_MODELS)
    ]
    sidecar = _tagged_sidecar([], {"p/long-gone": "deep", "p/live0": "fast"})
    out = aggregate(ev, sidecar, catalog_succeeded={"p"})
    assert out["p"]["withdrawn"] == ["p/long-gone"]


def test_no_withdrawals_when_absence_is_not_trusted():
    """Without a trusted catalog, nothing is treated as withdrawn — a docs-only
    provider must never lose its tags."""
    ev = [Evidence(provider="p", model_id="p/documented", is_free=True,
                   source="docs", confidence="high", url="u")]
    sidecar = _tagged_sidecar(["p/documented"], {"p/anything": "deep"})
    out = aggregate(ev, sidecar, catalog_succeeded=set())
    assert out["p"]["withdrawn"] == []


def test_apply_updates_prunes_withdrawn_tags_only():
    from scripts.update_free_models import apply_updates

    sidecar = _tagged_sidecar(
        ["p/repriced", "p/withdrawn"],
        {"p/repriced": "deep", "p/withdrawn": "exploratory", "p/paid-but-tagged": "fast"},
    )
    updates = {"p": {
        "add": [], "remove": ["p/repriced", "p/withdrawn"],
        "withdrawn": ["p/withdrawn"], "limits": {}, "capabilities": {}, "pricing": {},
    }}
    assert apply_updates(sidecar, updates) is True
    reasoning = sidecar["providers"]["p"]["model_reasoning"]
    assert "p/withdrawn" not in reasoning          # gone upstream → tag dropped
    assert reasoning["p/repriced"] == "deep"       # still exists → tag kept
    assert reasoning["p/paid-but-tagged"] == "fast"
    assert sidecar["providers"]["p"]["believed_free"] == []
