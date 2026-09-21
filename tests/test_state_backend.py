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


# ── single-flight leases ────────────────────────────────────────────────────

def test_a_lease_is_held_by_one_claimant(backend):
    token = backend.acquire_lease("job", 60)
    assert token
    assert backend.acquire_lease("job", 60) is None


def test_releasing_lets_the_next_claimant_in(backend):
    token = backend.acquire_lease("job", 60)
    backend.release_lease("job", token)
    assert backend.acquire_lease("job", 60) is not None


def test_different_jobs_do_not_block_each_other(backend):
    assert backend.acquire_lease("a", 60)
    assert backend.acquire_lease("b", 60)


def test_an_expired_lease_can_be_taken_over(backend):
    """The reason this is a lease and not a flag: a worker that dies mid-job
    leaves a flag set forever, so the job never runs again."""
    backend.acquire_lease("job", 0.01)
    time.sleep(0.05)
    assert backend.acquire_lease("job", 60) is not None


def test_an_overrun_holder_cannot_release_the_new_one(backend):
    """Holder-scoped release. Otherwise a slow job, taken over after its TTL,
    would free the lease its successor is relying on as it exits."""
    stale = backend.acquire_lease("job", 0.01)
    time.sleep(0.05)
    fresh = backend.acquire_lease("job", 60)
    backend.release_lease("job", stale)          # the overrun holder exiting
    assert backend.acquire_lease("job", 60) is None, "the new holder lost its lease"
    backend.release_lease("job", fresh)
    assert backend.acquire_lease("job", 60) is not None


def test_a_lease_can_be_renewed_by_its_holder(backend):
    token = backend.acquire_lease("job", 0.05)
    assert backend.renew_lease("job", token, 60) is True
    time.sleep(0.1)
    assert backend.acquire_lease("job", 60) is None, "renewal did not extend it"


def test_renewing_someone_elses_lease_fails(backend):
    backend.acquire_lease("job", 60)
    assert backend.renew_lease("job", "not-the-holder", 60) is False


def test_leases_are_listed_for_diagnostics(backend):
    """An flock is invisible; the question being asked is 'why did this not run'."""
    backend.acquire_lease("job", 60)
    held = backend.leases()
    assert [row["job"] for row in held] == ["job"]
    assert held[0]["expires_in"] > 0


def test_an_expired_lease_is_not_listed_as_held(backend):
    backend.acquire_lease("job", 0.01)
    time.sleep(0.05)
    assert backend.leases() == []


# ── derived-cache invalidation ──────────────────────────────────────────────

def test_the_cache_epoch_advances(backend):
    """Background jobs run on one worker now, so the others need telling."""
    start = backend.cache_epoch()
    backend.bump_cache_epoch()
    assert backend.cache_epoch() == start + 1
    backend.bump_cache_epoch()
    assert backend.cache_epoch() == start + 2


# ── Responses conversation store ────────────────────────────────────────────

def test_a_transcript_round_trips(backend):
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    backend.store_response("resp_1", msgs)
    assert backend.load_response("resp_1") == msgs


def test_an_unknown_id_is_none_not_an_error(backend):
    """The caller turns this into a clear 400 rather than answering without
    the referenced history, which would produce a confidently wrong reply."""
    assert backend.load_response("resp_nope") is None


def test_storing_the_same_id_replaces_it(backend):
    backend.store_response("resp_1", [{"role": "user", "content": "first"}])
    backend.store_response("resp_1", [{"role": "user", "content": "second"}])
    assert backend.load_response("resp_1") == [{"role": "user", "content": "second"}]


def test_an_empty_id_is_ignored(backend):
    backend.store_response("", [{"role": "user", "content": "x"}])
    assert backend.load_response("") is None


def test_a_transcript_is_a_copy(backend):
    """A caller mutating what it read must not corrupt the stored conversation."""
    backend.store_response("resp_1", [{"role": "user", "content": "hi"}])
    got = backend.load_response("resp_1")
    got.append({"role": "user", "content": "injected"})
    assert len(backend.load_response("resp_1")) == 1


def test_the_response_store_is_bounded(backend):
    from llmproxy.state import MAX_STORED_RESPONSES
    for i in range(MAX_STORED_RESPONSES + 5):
        backend.store_response(f"resp_{i}", [{"role": "user", "content": str(i)}])
    assert backend.load_response("resp_0") is None
    assert backend.load_response(f"resp_{MAX_STORED_RESPONSES + 4}") is not None


def test_a_transcript_can_be_deleted(backend):
    backend.store_response("resp_1", [{"role": "user", "content": "hi"}])
    assert backend.delete_response("resp_1") is True
    assert backend.delete_response("resp_1") is False
    assert backend.load_response("resp_1") is None


def test_clearing_empties_the_store(backend):
    backend.store_response("resp_1", [{"role": "user", "content": "hi"}])
    backend.clear_responses()
    assert backend.load_response("resp_1") is None


# ── the health window, which must mean the same thing everywhere ────────────

def test_the_health_window_is_bounded_by_attempts(backend):
    """Last N *attempts*, not last N minutes.

    Time-bucketed counters would be cheaper to share, but at a low request rate
    a broken provider would never accumulate enough samples inside the window to
    be demoted at all -- which is exactly when demotion matters.
    """
    from llmproxy.usage import HEALTH_WINDOW
    for _ in range(HEALTH_WINDOW * 3):
        backend.record_outcome("p/m", True, latency_ms=10.0)
    assert backend.health_snapshot("p/m")[2] == HEALTH_WINDOW


def test_recovery_evicts_old_failures(backend):
    """A provider that has recovered must stop being punished promptly."""
    from llmproxy.usage import HEALTH_WINDOW
    for _ in range(HEALTH_WINDOW):
        backend.record_outcome("p/m", False, latency_ms=10.0)
    assert backend.health_snapshot("p/m")[0] == 0.0
    for _ in range(HEALTH_WINDOW):
        backend.record_outcome("p/m", True, latency_ms=10.0)
    assert backend.health_snapshot("p/m")[0] == 1.0


def test_the_window_matches_a_single_process_deque_exactly(backend):
    """The claim the whole shared-health design rests on.

    Writers are serialised, so the surviving rows are the same ones a
    ``deque(maxlen=N)`` would have held over the same sequence -- not an
    approximation of them.
    """
    import collections
    import random

    from llmproxy.usage import HEALTH_WINDOW

    random.seed(7)
    seq = [random.random() > 0.35 for _ in range(HEALTH_WINDOW * 3)]
    for ok in seq:
        backend.record_outcome("p/m", ok, latency_ms=10.0)

    ref = collections.deque(seq, maxlen=HEALTH_WINDOW)
    expected = sum(1 for x in ref if x) / len(ref)
    rate, _latency, samples = backend.health_snapshot("p/m")
    assert samples == len(ref)
    assert rate == pytest.approx(expected)


# ── the bulk snapshot ───────────────────────────────────────────────────────

def test_the_snapshot_agrees_with_the_per_key_accessors(backend):
    """The snapshot is an optimisation, so it must not be a second opinion."""
    backend.record_usage("p/m", requests=2, total=30)
    backend.record_outcome("p/m", False, latency_ms=50.0)
    backend.mark_saturated("q/n", 60)

    snap = backend.snapshot()
    row = snap.row("p/m")
    assert (row.req_min, row.req_day) == backend.usage_snapshot("p/m")
    assert (row.tok_min, row.tok_day) == backend.token_snapshot("p/m")
    assert (row.success_rate, row.avg_latency_ms, row.health_samples) == \
        backend.health_snapshot("p/m")
    assert snap.is_saturated("q/n") is True
    assert snap.is_saturated("p/m") is False


def test_an_unseen_key_reads_empty_from_the_snapshot(backend):
    row = backend.snapshot().row("never/seen")
    assert (row.req_min, row.req_day, row.health_samples) == (0, 0, 0)
    assert row.success_rate == 1.0
