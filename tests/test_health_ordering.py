"""Tests for health-aware soft demotion.

Two properties matter most and are asserted directly:

* a candidate with a *proven* good record must never rank below an untried one
  (the health multiplier is neutral, not a penalty, at normal latency); and
* a client disconnect must not count against a provider — llmproxy streams by
  default, and one cancelled stream must not cascade into a demotion.
"""

from __future__ import annotations

import pytest

from llmproxy import server
from llmproxy.usage import HEALTH_WINDOW, ModelUsage


@pytest.fixture(autouse=True)
def _clean_registry():
    server._reset_usage()
    yield
    server._reset_usage()


def _observe(provider, model, outcomes, latency_ms=200.0):
    for ok in outcomes:
        server._record_outcome(provider, model, ok, latency_ms=latency_ms)


# — the ModelUsage primitive —

def test_untouched_tracker_reads_as_healthy_but_unsampled():
    rate, latency, samples = ModelUsage().health_snapshot()
    assert (rate, latency, samples) == (1.0, 0.0, 0)


def test_health_window_is_bounded():
    usage = ModelUsage()
    for _ in range(HEALTH_WINDOW * 3):
        usage.record_outcome(True, 10.0)
    assert usage.health_snapshot()[2] == HEALTH_WINDOW


def test_recovery_evicts_old_failures():
    usage = ModelUsage()
    for _ in range(HEALTH_WINDOW):
        usage.record_outcome(False, 10.0)
    assert usage.health_snapshot()[0] == 0.0
    for _ in range(HEALTH_WINDOW):
        usage.record_outcome(True, 10.0)
    assert usage.health_snapshot()[0] == 1.0


# — the score —

def test_cold_candidate_is_neutral():
    _observe("p", "m", [False] * (server._HEALTH_MIN_SAMPLES - 1))
    assert server._health_score("p", "m") == 1.0


def test_healthy_candidate_is_not_penalised_against_an_untried_one():
    """A proven-good record must tie an untried one, never rank below it."""
    _observe("proven", "m", [True] * 10)
    assert server._health_score("proven", "m") == pytest.approx(1.0)
    assert server._health_score("proven", "m") >= server._health_score("untried", "m")


def test_score_is_monotonic_in_success_rate():
    _observe("good", "m", [True] * 10)
    _observe("ok", "m", [True] * 9 + [False])
    _observe("poor", "m", [True] * 7 + [False] * 3)
    _observe("broken", "m", [False] * 10)
    scores = [server._health_score(p, "m") for p in ("good", "ok", "poor", "broken")]
    assert scores == sorted(scores, reverse=True)


def test_broken_candidate_keeps_a_non_zero_score():
    """Demotion, never exclusion: a degraded provider still beats a 503."""
    _observe("broken", "m", [False] * 20)
    assert server._health_score("broken", "m") == pytest.approx(server._HEALTH_MIN_MULTIPLIER)


def test_slow_but_working_candidate_is_only_mildly_demoted():
    _observe("slow", "m", [True] * 10, latency_ms=20_000)
    _observe("failing", "m", [False] * 10, latency_ms=200)
    assert 0.5 < server._health_score("slow", "m") < 1.0
    assert server._health_score("slow", "m") > server._health_score("failing", "m")


def test_normal_latency_is_neutral():
    _observe("fast", "m", [True] * 10, latency_ms=50)
    _observe("normal", "m", [True] * 10, latency_ms=server._HEALTH_FAST_MS)
    assert server._health_score("fast", "m") == server._health_score("normal", "m") == 1.0


def test_accounts_are_metered_independently():
    _observe_kwargs = dict(latency_ms=200.0)
    for _ in range(10):
        server._record_outcome("p", "m", False, account_id="a", **_observe_kwargs)
        server._record_outcome("p", "m", True, account_id="b", **_observe_kwargs)
    assert server._health_score("p", "m", "a") < server._health_score("p", "m", "b")


# — the failure classifier —

@pytest.mark.parametrize("message", [
    "Client disconnected",
    "Connection reset by peer",
    "[Errno 32] Broken pipe",
    "Controller is already closed",
    "The request aborted by the client",
])
def test_client_aborts_are_not_upstream_failures(message):
    assert server._is_upstream_failure(Exception(message)) is False


def test_generator_exit_is_not_an_upstream_failure():
    assert server._is_upstream_failure(GeneratorExit()) is False


def test_genuine_transport_errors_are_upstream_failures():
    assert server._is_upstream_failure(TimeoutError("read timed out")) is True


@pytest.mark.parametrize("status,expected", [
    (500, True), (502, True), (429, True), (402, True),
    (400, False), (404, False), (200, False),
])
def test_status_classification(status, expected):
    assert server._is_upstream_failure(status=status) is expected


def test_a_cancelled_stream_does_not_demote_a_provider():
    """The whole point of the classifier, asserted end to end on the score."""
    _observe("p", "m", [True] * 10)
    healthy = server._health_score("p", "m")
    for _ in range(20):
        exc = Exception("Client disconnected")
        if server._is_upstream_failure(exc):
            server._record_outcome("p", "m", False)
    assert server._health_score("p", "m") == healthy


# — ordering integration —

def test_capacity_ordering_demotes_an_unhealthy_candidate():
    limits = {"sick/m": {"requests_per_minute": 100}, "well/m": {"requests_per_minute": 100}}
    candidates = [("sick", {}, "m"), ("well", {}, "m")]
    _observe("sick", "m", [False] * 10)
    _observe("well", "m", [True] * 10)
    # Weighted sampling is random, so assert over repeated draws rather than one.
    firsts = [server._capacity_ordered_candidates(candidates, limits)[0][0] for _ in range(200)]
    assert firsts.count("well") > firsts.count("sick") * 3
    # Never dropped, even when broken.
    assert len(server._capacity_ordered_candidates(candidates, limits)) == 2
