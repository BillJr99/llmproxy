"""The routing-state backend contract.

Everything the routing layer learns at runtime lives behind one interface in
`llmproxy.state`. Today there is one implementation; the point of the interface
is that there can be a second, shared across processes, without `server.py`
changing. These tests describe the *contract* rather than the implementation, so
a second backend can be held to exactly the same behaviour by parametrizing the
fixture below.

The subtle properties are the ones worth pinning: a watermark keeps the minimum,
a capability gap is a union that reports novelty, a cooldown is expressed as a
TTL and read back as a decision, and an unseen key reads as healthy rather than
as failing.
"""

from __future__ import annotations

import time

import pytest

from llmproxy import state
from llmproxy.state import InMemoryState, SharedState, SqliteState


@pytest.fixture(params=["memory", "sqlite"])
def backend(request, tmp_path) -> SharedState:
    """One test body, every backend.

    This is the fixture that stops the two implementations drifting: every
    behaviour below is asserted identically against both, so a shared backend
    cannot quietly decide that a watermark keeps the maximum or that an unseen
    key reads as failing.
    """
    if request.param == "memory":
        be = InMemoryState()
    else:
        be = SqliteState(tmp_path / "shared_state.db")
    yield be
    close = getattr(be, "close", None)
    if close:
        close()


def test_the_implementation_satisfies_the_protocol(backend):
    assert isinstance(backend, SharedState)
    assert backend.kind


# ── usage, tokens, health ───────────────────────────────────────────────────

def test_usage_accumulates_per_key(backend):
    backend.record_usage("p/m", requests=1, total=10)
    backend.record_usage("p/m", requests=1, total=5)
    backend.record_usage("other/m", requests=1, total=99)
    assert backend.usage_snapshot("p/m") == (2, 2)
    assert backend.token_snapshot("p/m") == (15, 15)
    assert backend.usage_snapshot("other/m") == (1, 1)


def test_an_unseen_key_is_empty_rather_than_an_error(backend):
    assert backend.usage_snapshot("never/seen") == (0, 0)
    assert backend.token_snapshot("never/seen") == (0, 0)


def test_an_unseen_key_reads_as_healthy_but_unsampled(backend):
    """1.0 over zero samples means untried, not proven good. Callers gate on
    the sample count, so the rate alone must never demote a cold candidate."""
    rate, latency, samples = backend.health_snapshot("never/tried")
    assert (rate, latency, samples) == (1.0, 0.0, 0)


def test_outcomes_drive_the_health_rate(backend):
    for ok in (True, True, False, True):
        backend.record_outcome("p/m", ok, latency_ms=10.0)
    rate, latency, samples = backend.health_snapshot("p/m")
    assert samples == 4
    assert rate == pytest.approx(0.75)
    assert latency == pytest.approx(10.0)


def test_usage_rows_carry_what_the_report_needs(backend):
    backend.record_usage("p/m", requests=1, prompt=3, completion=4, total=7, cost=0.5)
    backend.record_outcome("p/m", True, latency_ms=20.0)
    rows = dict(backend.usage_rows())
    assert "p/m" in rows
    row = rows["p/m"]
    for field in ("requests", "prompt_tokens", "completion_tokens", "total_tokens",
                  "cost", "cost_sources", "tokens_last_60s", "tokens_today",
                  "success_rate", "avg_latency_ms", "health_samples"):
        assert field in row, field
    assert row["total_tokens"] == 7


def test_reset_clears_counters_and_moves_the_since_stamp(backend):
    before = backend.usage_since
    backend.record_usage("p/m", requests=1)
    backend.mark_saturated("p/m", 60)
    time.sleep(0.001)
    after = backend.reset_usage()
    assert backend.usage_snapshot("p/m") == (0, 0)
    assert backend.is_saturated("p/m") is False   # a reset clears cooldowns too
    assert after == backend.usage_since != before


# ── believed-free models observed costing money ─────────────────────────────

def test_paid_free_reports_novelty_so_the_caller_persists_once(backend):
    assert backend.flag_paid_free("p/m", 0.01, "provider") is True
    assert backend.flag_paid_free("p/m", 0.02, "provider") is False
    entry = backend.paid_free_flags()["p/m"]
    assert entry["samples"] == 2
    assert entry["observed_cost"] == pytest.approx(0.02)   # keeps the maximum


def test_paid_free_flags_are_a_copy(backend):
    """The report must not be able to mutate the registry by accident."""
    backend.flag_paid_free("p/m", 0.01, "provider")
    backend.paid_free_flags()["p/m"]["samples"] = 999
    assert backend.paid_free_flags()["p/m"]["samples"] == 1


# ── saturation ──────────────────────────────────────────────────────────────

def test_a_cooldown_is_a_ttl_in_and_a_decision_out(backend):
    """No clock crosses the interface: callers pass seconds and read a bool.

    That is what lets a cross-process backend store wall clock while this one
    stores a monotonic deadline, without either fact being visible here.
    """
    backend.mark_saturated("p/m", 60)
    assert backend.is_saturated("p/m") is True
    assert backend.is_saturated("other/m") is False


def test_a_zero_or_negative_cooldown_does_nothing(backend):
    backend.mark_saturated("p/m", 0)
    backend.mark_saturated("p/m", -5)
    assert backend.is_saturated("p/m") is False


def test_a_cooldown_expires(backend):
    backend.mark_saturated("p/m", 0.01)
    assert backend.is_saturated("p/m") is True
    time.sleep(0.02)
    assert backend.is_saturated("p/m") is False


def test_reset_saturation_clears_every_cooldown(backend):
    backend.mark_saturated("a/m", 60)
    backend.mark_saturated("b/m", 60)
    backend.reset_saturation()
    assert backend.is_saturated("a/m") is False
    assert backend.is_saturated("b/m") is False


# ── oversize watermarks ─────────────────────────────────────────────────────

def test_the_watermark_keeps_the_minimum(backend):
    """A smaller rejection tightens the limit; a larger one never loosens it.
    The bound can only be established from above by what was actually refused."""
    backend.record_oversize("p/m", 5000)
    backend.record_oversize("p/m", 9000)
    assert backend.is_oversize("p/m", 5000) is True
    backend.record_oversize("p/m", 3000)
    assert backend.is_oversize("p/m", 3000) is True
    assert backend.is_oversize("p/m", 2999) is False


def test_a_nonpositive_size_is_not_evidence(backend):
    backend.record_oversize("p/m", 0)
    backend.record_oversize("p/m", -1)
    assert backend.has_oversize() is False
    assert backend.is_oversize("p/m", 0) is False


def test_has_oversize_is_the_fast_path(backend):
    """Lets the demotion pass skip serializing a payload when nothing has 413'd."""
    assert backend.has_oversize() is False
    backend.record_oversize("p/m", 100)
    assert backend.has_oversize() is True


def test_a_success_at_the_watermark_disproves_it(backend):
    backend.record_oversize("p/m", 5000)
    assert backend.clear_oversize_at("p/m", 5000) is True
    assert backend.is_oversize("p/m", 5000) is False
    assert backend.clear_oversize_at("p/m", 5000) is False   # nothing left to clear


def test_a_smaller_success_says_nothing_about_the_limit(backend):
    backend.record_oversize("p/m", 5000)
    assert backend.clear_oversize_at("p/m", 4999) is False
    assert backend.is_oversize("p/m", 5000) is True


# ── learned capability gaps ─────────────────────────────────────────────────

def test_a_gap_is_a_union_that_reports_novelty(backend):
    """The return value gates a warning that must fire exactly once."""
    assert backend.record_capability_gap("p/m", "tools") is True
    assert backend.record_capability_gap("p/m", "tools") is False
    assert backend.record_capability_gap("p/m", "vision") is True
    assert backend.capability_gaps("p/m") == {"tools", "vision"}


def test_gaps_are_per_target_not_per_model(backend):
    """The whole reason this exists: one deployment of some weights cannot do
    what another can."""
    backend.record_capability_gap("a/m", "tools")
    assert backend.capability_gaps("b/m") == set()


def test_an_empty_capability_is_ignored(backend):
    assert backend.record_capability_gap("p/m", "") is False
    assert backend.capability_gaps("p/m") == set()


def test_capability_gaps_are_a_copy(backend):
    backend.record_capability_gap("p/m", "tools")
    backend.capability_gaps("p/m").add("vision")
    assert backend.capability_gaps("p/m") == {"tools"}


# ── affinity pins ───────────────────────────────────────────────────────────

def test_a_pin_round_trips(backend):
    backend.record_affinity("conv1", "p", "m")
    assert backend.affinity_target("conv1") == ("p", "m")
    assert backend.affinity_target("conv2") is None


def test_a_later_success_repins(backend):
    backend.record_affinity("conv1", "p", "m")
    backend.record_affinity("conv1", "q", "n")
    assert backend.affinity_target("conv1") == ("q", "n")


def test_pins_are_capped(backend):
    """An unbounded map keyed by conversation is a slow leak on a long process."""
    for i in range(state.AFFINITY_PIN_MAX + 50):
        backend.record_affinity(f"c{i}", "p", "m")
    assert backend.affinity_count() <= state.AFFINITY_PIN_MAX


def test_reset_affinity_clears_every_pin(backend):
    backend.record_affinity("conv1", "p", "m")
    backend.reset_affinity()
    assert backend.affinity_target("conv1") is None
    assert backend.affinity_count() == 0


# ── failure ring ────────────────────────────────────────────────────────────

def test_failures_come_back_newest_first(backend):
    now = time.time()
    for i in range(3):
        backend.record_failure({"ts": now + i, "model": f"m{i}"})
    assert [r["model"] for r in backend.failure_records()] == ["m2", "m1", "m0"]


def test_failures_can_be_filtered_by_age(backend):
    now = time.time()
    backend.record_failure({"ts": now - 100, "model": "old"})
    backend.record_failure({"ts": now, "model": "new"})
    assert [r["model"] for r in backend.failure_records(since_ts=now - 10)] == ["new"]


def test_failures_past_the_ttl_are_pruned(backend):
    now = time.time()
    backend.record_failure({"ts": now - state.FAILURE_LOG_TTL_S - 1, "model": "ancient"})
    backend.record_failure({"ts": now, "model": "recent"})
    assert [r["model"] for r in backend.failure_records()] == ["recent"]


def test_the_failure_ring_is_bounded(backend):
    now = time.time()
    for i in range(state.FAILURE_LOG_MAX + 25):
        backend.record_failure({"ts": now, "model": f"m{i}"})
    assert len(backend.failure_records()) == state.FAILURE_LOG_MAX


def test_reset_failures_empties_the_ring(backend):
    backend.record_failure({"ts": time.time(), "model": "m"})
    backend.reset_failures()
    assert backend.failure_records() == []


# ── selection ───────────────────────────────────────────────────────────────

def test_get_backend_is_a_singleton_until_reset():
    state.set_backend(None)
    first = state.get_backend()
    assert state.get_backend() is first
    state.reset_for_worker()
    assert state.get_backend() is not first


def test_set_backend_installs_an_explicit_one():
    mine = InMemoryState()
    state.set_backend(mine)
    try:
        assert state.get_backend() is mine
    finally:
        state.set_backend(None)
