"""apply_updates: mutates the sidecar in-place and reports a change flag."""

from __future__ import annotations

import copy

from scripts.update_free_models import apply_updates


def _sidecar() -> dict:
    return {
        "providers": {
            "p": {
                "believed_free": ["p/old"],
                "model_reasoning": {"p/old": "standard"},
                "free_limits": {"p/old": {"requests_per_minute": 10, "requests_per_day": None,
                                          "tokens_per_minute": None, "tokens_per_day": None}},
                "base_url": "u",
                "display": "X",
            }
        },
        "provider_order": ["p"],
    }


def test_apply_adds_and_assigns_reasoning():
    s = _sidecar()
    updates = {"p": {"add": ["p/new-7b"], "remove": [], "limits": {}}}
    changed = apply_updates(s, updates)
    assert changed is True
    assert "p/new-7b" in s["providers"]["p"]["believed_free"]
    # 7b → exploratory under the size heuristic
    assert s["providers"]["p"]["model_reasoning"]["p/new-7b"] == "exploratory"


def test_apply_removes_and_drops_limits():
    s = _sidecar()
    updates = {"p": {"add": [], "remove": ["p/old"], "limits": {}}}
    changed = apply_updates(s, updates)
    assert changed is True
    assert "p/old" not in s["providers"]["p"]["believed_free"]
    assert "p/old" not in s["providers"]["p"]["free_limits"]


def test_no_changes_returns_false():
    s = _sidecar()
    before = copy.deepcopy(s)
    updates = {"p": {"add": [], "remove": [], "limits": {}}}
    changed = apply_updates(s, updates)
    assert changed is False
    assert s == before


def test_limits_only_apply_to_free_models():
    """If a model isn't in believed_free, its limits should be ignored."""
    s = _sidecar()
    updates = {"p": {"add": [], "remove": [],
                     "limits": {"p/not-free": {"requests_per_minute": 1,
                                               "requests_per_day": None,
                                               "tokens_per_minute": None,
                                               "tokens_per_day": None}}}}
    changed = apply_updates(s, updates)
    assert changed is False
    assert "p/not-free" not in s["providers"]["p"]["free_limits"]
