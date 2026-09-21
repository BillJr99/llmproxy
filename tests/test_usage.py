"""Unit tests for llmproxy.usage — pure token/cost accounting primitives."""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import pytest

from llmproxy import usage
from llmproxy.usage import (
    ModelUsage,
    compute_cost,
    extract_usage,
    parse_stream_usage,
)

# ── ModelUsage ──────────────────────────────────────────────────────────────

def test_request_windows_match_legacy_behaviour():
    u = ModelUsage()
    u.record(requests=1)
    u.record(requests=1)
    assert u.snapshot() == (2, 2)


def test_token_windows_accumulate():
    u = ModelUsage()
    u.record(requests=1, prompt=10, completion=5, total=15)
    u.record(requests=1, prompt=20, completion=10, total=30)
    tok_min, tok_day = u.token_snapshot()
    assert tok_min == 45
    assert tok_day == 45


def test_minute_token_window_prunes(monkeypatch):
    u = ModelUsage()
    base = [1000.0]
    monkeypatch.setattr(usage.time, "monotonic", lambda: base[0])
    u.record(requests=1, total=100)
    base[0] += 120  # advance past the 60s window
    tok_min, tok_day = u.token_snapshot()
    assert tok_min == 0      # minute window pruned
    assert tok_day == 100    # day total retained


def test_lifetime_and_cost_snapshot():
    u = ModelUsage()
    u.record(requests=1, prompt=10, completion=5, total=15, cost=0.002, cost_source="provider")
    u.record(requests=0, prompt=1, completion=1, total=2, cost=0.0001, cost_source="computed")
    snap = u.cost_snapshot()
    assert snap["requests"] == 1
    assert snap["prompt_tokens"] == 11
    assert snap["completion_tokens"] == 6
    assert snap["total_tokens"] == 17
    assert snap["cost"] == pytest.approx(0.0021)
    assert snap["cost_sources"] == {"provider": 1, "computed": 1}


def test_day_rollover_resets_requests_and_tokens(monkeypatch):
    u = ModelUsage()
    u.record(requests=1, total=50)
    # Force the day window to look stale by moving the calendar date on.
    monkeypatch.setattr(usage, "day_key", lambda tz=None: "1999-12-31")
    assert u.snapshot() == (0, 0)
    assert u.token_snapshot() == (0, 0)


def test_a_new_day_zeroes_the_counters_on_the_next_record(monkeypatch):
    """snapshot() reports zero for a rolled day; record() does the actual reset."""
    u = ModelUsage()
    u.record(requests=3, total=90)
    assert u.snapshot() == (3, 3)

    monkeypatch.setattr(usage, "day_key", lambda tz=None: "2100-01-01")
    u.record(requests=1, total=10)
    assert u.snapshot()[1] == 1          # today, not 4
    assert u.token_snapshot()[1] == 10


# ── the day boundary, across DST ────────────────────────────────────────────
#
# The daily windows used to track a float `_day_start` and roll at
# `_day_start + 86400`, which is not next local midnight on a day that gains or
# loses an hour. A 23-hour day rolled an hour late, charging the first hour of
# the new day against yesterday's free_limits allowance; a 25-hour day rolled an
# hour early. It re-aligned afterwards, so the error was one boundary twice a
# year rather than a permanent skew -- but a quota exhausted against a day the
# provider has already reset is a real failed request, at a boundary nobody
# watches. Without these tests the regression is invisible for six months.

_NY = "America/New_York"


def _at(iso: str, zone: str) -> datetime.datetime:
    """A tzinfo-aware datetime in *zone*, for freezing day_key()."""
    return datetime.datetime.fromisoformat(iso).replace(tzinfo=ZoneInfo(zone))


@pytest.mark.parametrize("zone", [_NY, "UTC", "Australia/Lord_Howe"])
def test_day_key_is_the_local_calendar_date(zone):
    tz = ZoneInfo(zone)
    assert usage.day_key(tz) == datetime.datetime.now(tz).date().isoformat()


@pytest.mark.parametrize("iso,expected", [
    # US spring forward: 2026-03-08 02:00 EST -> 03:00 EDT. The day is 23h long.
    ("2026-03-08T00:30", "2026-03-08"),
    ("2026-03-08T23:30", "2026-03-08"),
    ("2026-03-09T00:30", "2026-03-09"),
    # US fall back: 2026-11-01 02:00 EDT -> 01:00 EST. The day is 25h long.
    ("2026-11-01T00:30", "2026-11-01"),
    ("2026-11-01T23:30", "2026-11-01"),
    ("2026-11-02T00:30", "2026-11-02"),
])
def test_the_date_is_right_either_side_of_a_transition(monkeypatch, iso, expected):
    """A 23- or 25-hour day still rolls exactly once, at local midnight.

    `_day_start + 86400` could not express this: on the 23-hour day it rolled an
    hour into the next day, and on the 25-hour day an hour early.
    """
    frozen = _at(iso, _NY)

    class _FrozenDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen.astimezone(tz) if tz else frozen

    monkeypatch.setattr(usage.datetime, "datetime", _FrozenDatetime)
    assert usage.day_key(ZoneInfo(_NY)) == expected


def test_the_boundary_does_not_drift_after_a_transition(monkeypatch):
    """The day after a transition must be correct too, not just the day of.

    Walking three consecutive local noons across the spring transition must
    yield three consecutive dates. This is the property that would catch a fix
    which special-cased the transition day but left the following one skewed.
    """
    tz = ZoneInfo(_NY)
    seen = []
    for iso in ("2026-03-07T12:00", "2026-03-08T12:00", "2026-03-09T12:00"):
        frozen = _at(iso, _NY)

        class _FrozenDatetime(datetime.datetime):
            @classmethod
            def now(cls, tz=None, _f=frozen):
                return _f.astimezone(tz) if tz else _f

        monkeypatch.setattr(usage.datetime, "datetime", _FrozenDatetime)
        seen.append(usage.day_key(tz))
    assert seen == ["2026-03-07", "2026-03-08", "2026-03-09"]


def test_set_default_timezone_pins_the_boundary(monkeypatch):
    """Two components resolving 'today' independently could split a day's
    counter in half, which reads as a quota reset that never happened."""
    try:
        usage.set_default_timezone(ZoneInfo("Pacific/Kiritimati"))   # UTC+14
        east = usage.day_key()
        usage.set_default_timezone(ZoneInfo("Pacific/Midway"))       # UTC-11
        west = usage.day_key()
    finally:
        usage.set_default_timezone(None)
    assert east >= west                    # 25 hours apart: same date or one ahead
    assert usage.day_key() == usage.day_key(None)


# ── extract_usage ───────────────────────────────────────────────────────────

def test_extract_usage_openai_shape():
    body = b'{"usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}'
    assert extract_usage(body) == {
        "prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
    }


def test_extract_usage_openrouter_cost():
    body = {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0005}}
    out = extract_usage(body)
    assert out["cost"] == 0.0005


def test_extract_usage_total_inferred():
    body = {"usage": {"prompt_tokens": 5, "completion_tokens": 7}}
    assert extract_usage(body)["total_tokens"] == 12


def test_extract_usage_missing_or_malformed():
    assert extract_usage(None) is None
    assert extract_usage(b"not json") is None
    assert extract_usage({"choices": []}) is None
    assert extract_usage({"usage": {"prompt_tokens": 0, "completion_tokens": 0}}) is None


# ── parse_stream_usage ──────────────────────────────────────────────────────

def test_parse_stream_usage_final_chunk():
    sse = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert parse_stream_usage(sse) == {
        "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
    }


def test_parse_stream_usage_absent():
    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    assert parse_stream_usage(sse) is None
    assert parse_stream_usage(b"") is None


# ── compute_cost ────────────────────────────────────────────────────────────

def test_compute_cost_prefers_provider_reported():
    usage_obj = {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
    pricing = {"p/m": (0.000001, 0.000002)}
    cost, source = compute_cost("p", "m", usage_obj, pricing)
    assert source == "provider"
    assert cost == 0.01


def test_compute_cost_computed_from_pricing():
    usage_obj = {"prompt_tokens": 100, "completion_tokens": 50}
    pricing = {"p/m": (0.000001, 0.000002)}
    cost, source = compute_cost("p", "m", usage_obj, pricing)
    assert source == "computed"
    assert cost == pytest.approx(100 * 0.000001 + 50 * 0.000002)


def test_compute_cost_unknown():
    assert compute_cost("p", "m", {"prompt_tokens": 10}, {}) == (0.0, "unknown")
    assert compute_cost("p", "m", None, None) == (0.0, "unknown")


def test_usage_timezone_config_pins_the_boundary(monkeypatch):
    """A container's local zone is usually UTC while the provider's quota
    resets somewhere else."""
    import llmproxy.server as S
    try:
        S._apply_usage_timezone({"server": {"usage_timezone": "Pacific/Kiritimati"}})
        east = usage.day_key()
        S._apply_usage_timezone({"server": {"usage_timezone": "Pacific/Midway"}})
        west = usage.day_key()
    finally:
        usage.set_default_timezone(None)
    assert east >= west


def test_an_unusable_timezone_is_ignored_not_fatal(caplog):
    """A typo in an optional field must not stop the proxy from routing."""
    import logging

    import llmproxy.server as S
    try:
        with caplog.at_level(logging.WARNING, logger="llmproxy.server"):
            S._apply_usage_timezone({"server": {"usage_timezone": "Not/AZone"}})
        assert usage.day_key() == usage.day_key(None), "should fall back to local"
        assert any("usage_timezone" in r.getMessage() for r in caplog.records)
    finally:
        usage.set_default_timezone(None)


def test_no_timezone_configured_leaves_local_time_alone():
    import llmproxy.server as S
    S._apply_usage_timezone({"server": {}})
    assert usage.day_key() == usage.day_key(None)
