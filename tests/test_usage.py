"""Unit tests for llmproxy.usage — pure token/cost accounting primitives."""

from __future__ import annotations

import time

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
    # Force the day window to look stale.
    u._day_start = time.time() - 86401
    assert u.snapshot() == (0, 0)
    assert u.token_snapshot() == (0, 0)


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
