"""Tests for the cost_observed_free_tier denylist on the updater side.

When a believed_free model serves a request reporting a non-zero cost the proxy
records its qualified id in config['cost_observed_free_tier']; the updater must
then never re-add it to believed_free and must remove it if present — in both the
sidecar aggregation and the user-config reconcile (the per-boot startup sync).
"""

from __future__ import annotations

from scripts.sources.base import Evidence
from scripts.update_free_models import (
    aggregate,
    cost_observed_denylist,
    reconcile_user_config,
)


def _free_evidence(model_id: str) -> Evidence:
    return Evidence(
        provider=model_id.split("/", 1)[0],
        model_id=model_id,
        is_free=True,
        source="openrouter",
        confidence="high",
        url="http://openrouter.example",
    )


def test_cost_observed_denylist_parsing():
    assert cost_observed_denylist({"cost_observed_free_tier": ["A/B", "c/d"]}) == {"a/b", "c/d"}
    assert cost_observed_denylist({"cost_observed_free_tier": "nope"}) == set()
    assert cost_observed_denylist({}) == set()
    assert cost_observed_denylist(None) == set()


def test_aggregate_does_not_add_denied_model():
    sidecar = {"providers": {"openrouter": {"believed_free": []}}}
    evs = [_free_evidence("openrouter/google/gemini-2.5-flash")]
    deny = {"openrouter/google/gemini-2.5-flash"}
    out = aggregate(evs, sidecar, set(), denylist=deny)
    assert out["openrouter"]["add"] == []
    # Without the denylist the same evidence would add it.
    out2 = aggregate(evs, sidecar, set())
    assert out2["openrouter"]["add"] == ["openrouter/google/gemini-2.5-flash"]


def test_aggregate_removes_denied_model_already_present():
    sidecar = {"providers": {"openrouter": {"believed_free": ["openrouter/google/gemini-2.5-flash"]}}}
    evs = [_free_evidence("openrouter/google/gemini-2.5-flash")]  # source still says free
    deny = {"openrouter/google/gemini-2.5-flash"}
    out = aggregate(evs, sidecar, set(), denylist=deny)
    assert out["openrouter"]["remove"] == ["openrouter/google/gemini-2.5-flash"]
    assert out["openrouter"]["add"] == []


def test_aggregate_denylist_is_case_insensitive():
    sidecar = {"providers": {"openrouter": {"believed_free": ["openrouter/Gemini"]}}}
    out = aggregate([], sidecar, set(), denylist={"openrouter/gemini"})
    assert out["openrouter"]["remove"] == ["openrouter/Gemini"]


def test_reconcile_removes_denied_and_blocks_resync():
    sidecar = {"providers": {"openrouter": {"believed_free": ["openrouter/gemini-2.5-flash"]}}}
    user_cfg = {
        "providers": {"openrouter": {"base_url": "x", "api_key": "k"}},
        "believed_free": ["openrouter/gemini-2.5-flash"],
        "cost_observed_free_tier": ["openrouter/gemini-2.5-flash"],
    }
    changes = reconcile_user_config(sidecar, user_cfg)
    # Removed from the live config and not re-added from the sidecar.
    assert "openrouter/gemini-2.5-flash" not in user_cfg["believed_free"]
    assert "openrouter/gemini-2.5-flash" in changes["believed_free"]["remove"]
    assert "openrouter/gemini-2.5-flash" not in changes["believed_free"]["add"]


def test_reconcile_without_denylist_keeps_model():
    sidecar = {"providers": {"openrouter": {"believed_free": ["openrouter/gemini-2.5-flash"]}}}
    user_cfg = {
        "providers": {"openrouter": {"base_url": "x", "api_key": "k"}},
        "believed_free": ["openrouter/gemini-2.5-flash"],
    }
    reconcile_user_config(sidecar, user_cfg)
    assert user_cfg["believed_free"] == ["openrouter/gemini-2.5-flash"]
