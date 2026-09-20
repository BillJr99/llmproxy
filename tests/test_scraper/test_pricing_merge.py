"""Pricing block assembly: litellm baseline + per-source overrides.

Covers _merge_pricing in scripts/update_free_models.py — the live high-confidence
sources (OpenRouter, Together, …) override the broad litellm baseline, the result
is sorted/idempotent, and a litellm outage never wipes existing prices.
"""

from __future__ import annotations

from scripts import update_free_models as ufm
from scripts.sources.base import Evidence
from scripts.update_free_models import _merge_pricing, aggregate

_A = {"input_cost_per_token": 1e-7, "output_cost_per_token": 2e-7}   # live source
_B = {"input_cost_per_token": 5e-7, "output_cost_per_token": 9e-7}   # litellm baseline
_C = {"input_cost_per_token": 3e-6, "output_cost_per_token": 6e-6}   # baseline-only


def _sidecar(pricing=None):
    sc = {
        "providers": {
            "p": {"believed_free": [], "model_reasoning": {}, "free_limits": {},
                  "base_url": "u", "display": "X"},
        },
        "provider_order": ["p"],
    }
    if pricing is not None:
        sc["pricing"] = pricing
    return sc


def _updates_with_live_price():
    ev = [Evidence(provider="p", model_id="p/m", is_free=False, source="openrouter",
                   confidence="high", url="u", pricing=_A)]
    return aggregate(ev, _sidecar(), catalog_succeeded=set())


def test_live_source_overrides_litellm_baseline(monkeypatch):
    monkeypatch.setattr(ufm, "fetch_pricing_map", lambda *a, **k: {"p/m": _B, "p/other": _C})
    sidecar = _sidecar({})
    changed = _merge_pricing(sidecar, _updates_with_live_price(), litellm_ran=True)
    assert changed is True
    # Live price wins for p/m; the baseline-only model is retained.
    assert sidecar["pricing"]["p/m"] == _A
    assert sidecar["pricing"]["p/other"] == _C
    # Block is key-sorted for deterministic output.
    assert list(sidecar["pricing"]) == sorted(sidecar["pricing"])


def test_merge_is_idempotent(monkeypatch):
    monkeypatch.setattr(ufm, "fetch_pricing_map", lambda *a, **k: {"p/m": _B})
    sidecar = _sidecar({})
    updates = _updates_with_live_price()
    assert _merge_pricing(sidecar, updates, litellm_ran=True) is True
    # Re-running with the same inputs makes no further change.
    assert _merge_pricing(sidecar, dict(updates), litellm_ran=True) is False


def test_litellm_outage_keeps_existing_baseline(monkeypatch):
    # fetch should NOT be called when litellm_ran is False.
    def _boom(*a, **k):
        raise AssertionError("fetch_pricing_map should not run when litellm_ran=False")
    monkeypatch.setattr(ufm, "fetch_pricing_map", _boom)
    sidecar = _sidecar({"p/other": _C})
    changed = _merge_pricing(sidecar, _updates_with_live_price(), litellm_ran=False)
    assert changed is True
    assert sidecar["pricing"]["p/other"] == _C   # existing baseline preserved
    assert sidecar["pricing"]["p/m"] == _A        # live override still applied


def test_pinned_pricing_survives_a_baseline_refresh(monkeypatch):
    """GUARD: the block is rebuilt each run, so an unsourced rate needs pinning.

    Unbiased AI publishes no catalog, so no live source will ever carry
    unbiased-ai/pareto and LiteLLM has no entry for llmproxy's provider key.
    Without _PINNED_PRICING the rate would vanish on the next scrape and the
    loadbalanced virtual would have nothing to rank the model by.
    """
    monkeypatch.setattr(ufm, "fetch_pricing_map", lambda *a, **k: {"p/m": _B})
    sidecar = _sidecar({})
    _merge_pricing(sidecar, _updates_with_live_price(), litellm_ran=True)
    assert sidecar["pricing"]["unbiased-ai/pareto"] == {
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 7.5e-06,
    }


def test_a_live_source_still_beats_a_pinned_rate(monkeypatch):
    """Pinned rates are hand-transcribed, so fresher live data must win."""
    live = {"input_cost_per_token": 9e-9, "output_cost_per_token": 9e-9}
    ev = [Evidence(provider="unbiased-ai", model_id="unbiased-ai/pareto", is_free=False,
                   source="openrouter", confidence="high", url="u", pricing=live)]
    sidecar = _sidecar({})
    sidecar["providers"]["unbiased-ai"] = {
        "believed_free": [], "model_reasoning": {}, "free_limits": {},
        "base_url": "u", "display": "Unbiased AI",
    }
    sidecar["provider_order"].append("unbiased-ai")
    updates = aggregate(ev, sidecar, catalog_succeeded=set())
    monkeypatch.setattr(ufm, "fetch_pricing_map", lambda *a, **k: {})
    _merge_pricing(sidecar, updates, litellm_ran=True)
    assert sidecar["pricing"]["unbiased-ai/pareto"] == live
