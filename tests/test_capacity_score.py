"""Token-aware capacity scoring for free-tier load balancing."""

from __future__ import annotations

from llmproxy import server


def test_no_limits_is_neutral():
    assert server._capacity_score(100, 100, {}) == 1.0


def test_request_only_limits_unchanged():
    # Regression: request limits behave exactly as before when no token limits set.
    limits = {"requests_per_minute": 10, "requests_per_day": 100}
    assert server._capacity_score(0, 0, limits) == 1.0
    assert server._capacity_score(5, 0, limits) == 0.5
    assert server._capacity_score(10, 0, limits) == 0.0  # rpm exhausted


def test_token_limits_narrow_score():
    limits = {"tokens_per_minute": 1000, "tokens_per_day": 10000}
    # Half the per-minute token budget consumed → 0.5.
    assert server._capacity_score(0, 0, limits, 500, 0) == 0.5
    # Per-minute token budget exhausted → 0.0.
    assert server._capacity_score(0, 0, limits, 1000, 0) == 0.0


def test_worst_constrained_dimension_wins():
    limits = {"requests_per_minute": 10, "tokens_per_minute": 1000}
    # rpm headroom 0.9, tpm headroom 0.1 → min == 0.1.
    score = server._capacity_score(1, 0, limits, 900, 0)
    assert score == 0.1
