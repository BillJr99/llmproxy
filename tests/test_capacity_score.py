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


# ── quota-error classifier ───────────────────────────────────────────────────

def test_quota_error_on_status():
    assert server._is_quota_error(429) is True
    assert server._is_quota_error(402) is True          # out of credits
    assert server._is_quota_error(500) is False         # transient, not quota
    assert server._is_quota_error(200) is False


def test_quota_error_on_body_codes():
    assert server._is_quota_error(200, b'{"error":{"code":"insufficient_quota"}}') is True
    assert server._is_quota_error(400, b'{"error":{"status":"RESOURCE_EXHAUSTED"}}') is True
    assert server._is_quota_error(429, b'{"error":{"type":"rate_limit_exceeded"}}') is True


def test_quota_error_generic_phrase_only_in_error_body():
    assert server._is_quota_error(200, b'{"error":{"message":"You exceeded your quota"}}') is True
    # A benign completion that happens to mention the word is not an error body.
    assert server._is_quota_error(200, b'{"choices":[{"message":{"content":"quota"}}]}') is False


def test_quota_error_ignores_empty_body():
    assert server._is_quota_error(500, b"") is False
    assert server._is_quota_error(None, None) is False


# ── saturation registry ──────────────────────────────────────────────────────

def test_mark_and_check_saturation():
    server._reset_usage()
    key = server._usage_key("groq", "m", "acct-a")
    assert server._is_saturated(key) is False
    server._mark_saturated(key, retry_after="30")
    assert server._is_saturated(key) is True


def test_saturation_expires(monkeypatch):
    server._reset_usage()
    key = server._usage_key("groq", "m")
    server._mark_saturated(key, retry_after="5")
    # Jump the monotonic clock past the cooldown.
    base = server.time.monotonic()
    monkeypatch.setattr(server.time, "monotonic", lambda: base + 6.0)
    assert server._is_saturated(key) is False


def test_candidate_saturation_is_account_scoped():
    server._reset_usage()
    server._mark_saturated(server._usage_key("groq", "m", "a"), retry_after="60")
    # Account a is cooling; account b (and the anonymous form) are not.
    assert server._is_candidate_saturated("groq", "m", "a") is True
    assert server._is_candidate_saturated("groq", "m", "b") is False
    assert server._is_candidate_saturated("groq", "m", None) is False


def test_provider_circuit_cools_whole_provider_account():
    server._reset_usage()
    server._mark_provider_circuit("groq", "a", retry_after="60")
    # Any model under provider groq / account a is now considered saturated.
    assert server._is_candidate_saturated("groq", "m1", "a") is True
    assert server._is_candidate_saturated("groq", "m2", "a") is True
    assert server._is_candidate_saturated("groq", "m1", "b") is False


def test_saturated_candidate_demoted_in_capacity_ordering():
    server._reset_usage()
    # Two free candidates, no configured limits; saturate the first one.
    cands = [
        ("groq", {"base_url": "http://groq/v1"}, "m1"),
        ("groq", {"base_url": "http://groq/v1"}, "m2"),
    ]
    server._mark_saturated(server._usage_key("groq", "m1"), retry_after="60")
    ordered = server._capacity_ordered_candidates(cands, {})
    # The cooling model is pushed to the back even with no free_limits set.
    assert ordered[-1][2] == "m1"
    assert ordered[0][2] == "m2"


def test_retry_after_http_date_parsed():
    # A far-future HTTP-date yields a positive cooldown; a past one yields ~0.
    future = "Wed, 21 Oct 2099 07:28:00 GMT"
    assert (server._parse_retry_after(future) or 0) > 0
    past = "Wed, 21 Oct 1999 07:28:00 GMT"
    assert (server._parse_retry_after(past) or 0) == 0
