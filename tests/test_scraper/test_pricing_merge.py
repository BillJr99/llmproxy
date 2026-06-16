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
    return aggregate(ev, _sidecar(), api_succeeded=set())


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
