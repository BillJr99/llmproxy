"""Aggregation rules in scripts/update_free_models.py.

Tests the contract of `aggregate(evidence, sidecar, api_succeeded)`:
- High-confidence positive adds a model.
- High-confidence negative blocks an add and forces a remove.
- Low-confidence alone never adds or removes.
- Models absent from a successful /v1/models response are flagged for removal.
- Limits are merged from the highest-confidence non-empty record.
"""

from __future__ import annotations

from scripts.sources.base import Evidence
from scripts.update_free_models import aggregate


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
    out = aggregate(ev, _sidecar("p", []), api_succeeded=set())
    assert out["p"]["add"] == ["p/new"]
    assert out["p"]["remove"] == []


def test_high_confidence_negative_blocks_add():
    ev = [
        Evidence(provider="p", model_id="p/x", is_free=True, source="docs",
                 confidence="high", url="u"),
        Evidence(provider="p", model_id="p/x", is_free=False, source="docs2",
                 confidence="high", url="u"),
    ]
    out = aggregate(ev, _sidecar("p", []), api_succeeded=set())
    assert out["p"]["add"] == []


def test_low_confidence_alone_does_not_add():
    ev = [Evidence(provider="p", model_id="p/x", is_free=True, source="community",
                   confidence="low", url="u")]
    out = aggregate(ev, _sidecar("p", []), api_succeeded=set())
    assert out["p"]["add"] == []


def test_high_confidence_negative_removes_existing():
    ev = [Evidence(provider="p", model_id="p/old", is_free=False, source="docs",
                   confidence="high", url="u")]
    out = aggregate(ev, _sidecar("p", ["p/old"]), api_succeeded=set())
    assert out["p"]["remove"] == ["p/old"]


def test_api_absence_triggers_remove_only_when_api_succeeded():
    """Models absent from a successful /v1/models fetch should be removed,
    but only if that provider's API actually ran cleanly."""
    # No api evidence, no api_succeeded — must NOT remove
    out = aggregate([], _sidecar("p", ["p/maybe"]), api_succeeded=set())
    assert out["p"]["remove"] == []

    # API succeeded for p, but p/maybe wasn't in the response → remove
    ev = [Evidence(provider="p", model_id="p/other", is_free=None, source="api",
                   confidence="medium", url="u")]
    out = aggregate(ev, _sidecar("p", ["p/maybe", "p/other"]), api_succeeded={"p"})
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
    out = aggregate(ev, _sidecar("p", []), api_succeeded=set())
    assert out["p"]["limits"]["p/m"]["requests_per_minute"] == 30
