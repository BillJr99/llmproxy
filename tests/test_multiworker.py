"""Sharing routing state between processes, proved with real processes.

`test_state_backend.py` holds both backends to one contract, but a contract test
runs in one process and therefore cannot see the thing that matters here: does a
cooldown recorded by worker A actually reach worker B?

**Always `spawn`, never `fork`.** A forked child inherits the parent's module
globals, including an already-built backend and anything it has cached, which
would mask exactly the bug under test. `spawn` gives a genuinely cold process
that has to reach the database to see anything at all.

These are the tests that would fail if the shared store silently degraded to
per-process state — which is the failure mode worth the most guarding, because
everything still *works*, just wrongly.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from llmproxy import state

# Every process below is joined with an explicit timeout and its exit code
# asserted, so a hang fails the test rather than wedging the run. No timeout
# marker: pytest-timeout is not a dependency, and --strict-markers would reject
# one that is not registered.
_CTX = mp.get_context("spawn")


# Module-level so spawn can pickle them by reference.

def _w_record_gap(db, key, cap, q):
    from llmproxy.state import SqliteState
    be = SqliteState(db)
    q.put(be.record_capability_gap(key, cap))
    be.close()


def _w_read_gaps(db, key, q):
    from llmproxy.state import SqliteState
    be = SqliteState(db)
    q.put(sorted(be.capability_gaps(key)))
    be.close()


def _w_mark_saturated(db, key, cooldown, q):
    from llmproxy.state import SqliteState
    be = SqliteState(db)
    be.mark_saturated(key, cooldown)
    q.put(True)
    be.close()


def _w_is_saturated(db, key, q):
    from llmproxy.state import SqliteState
    be = SqliteState(db)
    q.put(be.is_saturated(key))
    be.close()


def _w_record_oversize(db, key, size, q):
    from llmproxy.state import SqliteState
    be = SqliteState(db)
    be.record_oversize(key, size)
    q.put(True)
    be.close()


def _w_is_oversize(db, key, size, q):
    from llmproxy.state import SqliteState
    be = SqliteState(db)
    q.put(be.is_oversize(key, size))
    be.close()


def _w_record_affinity(db, akey, provider, model, q):
    from llmproxy.state import SqliteState
    be = SqliteState(db)
    be.record_affinity(akey, provider, model)
    q.put(True)
    be.close()


def _w_affinity_target(db, akey, q):
    from llmproxy.state import SqliteState
    be = SqliteState(db)
    q.put(be.affinity_target(akey))
    be.close()


def _w_hammer_gaps(db, key, caps, q):
    """Write many gaps from one process, to collide with the others."""
    from llmproxy.state import SqliteState
    be = SqliteState(db)
    won = 0
    for cap in caps:
        if be.record_capability_gap(key, cap):
            won += 1
    q.put(won)
    be.close()


def _run(target, *args):
    q = _CTX.Queue()
    p = _CTX.Process(target=target, args=(*args, q))
    p.start()
    p.join(timeout=60)
    assert p.exitcode == 0, f"worker exited {p.exitcode}"
    return q.get(timeout=5)


@pytest.fixture
def db(tmp_path):
    """A shared store, created by the 'master' as startup does."""
    path = str(tmp_path / "shared_state.db")
    be = state.SqliteState(path)
    be.close()
    return path


# ── the point of the whole exercise ─────────────────────────────────────────

def test_a_cooldown_recorded_in_one_process_is_seen_in_another(db):
    """The bug server.workers=1 exists to avoid: a 429 that cools a candidate in
    one worker being invisible to the others, which hit the same dead endpoint
    on the very next request."""
    assert _run(_w_is_saturated, db, "p/m") is False
    _run(_w_mark_saturated, db, "p/m", 60)
    assert _run(_w_is_saturated, db, "p/m") is True


def test_an_expired_cooldown_is_not_seen_in_another_process(db):
    """Shared must not mean sticky: a cooldown that has lapsed is not a cooldown."""
    _run(_w_mark_saturated, db, "p/m", 0.01)
    import time
    time.sleep(0.05)
    assert _run(_w_is_saturated, db, "p/m") is False


def test_a_capability_gap_learned_in_one_process_reaches_the_others(db):
    _run(_w_record_gap, db, "p/m", "tools")
    assert _run(_w_read_gaps, db, "p/m") == ["tools"]


def test_a_gap_is_reported_new_exactly_once_across_processes(db):
    """The return value gates a warning. Two workers learning the same gap must
    not both warn -- and more importantly must not both act as if it were new."""
    first = _run(_w_record_gap, db, "p/m", "tools")
    second = _run(_w_record_gap, db, "p/m", "tools")
    assert (first, second) == (True, False)


def test_an_oversize_watermark_crosses_processes(db):
    _run(_w_record_oversize, db, "p/m", 5000)
    assert _run(_w_is_oversize, db, "p/m", 5000) is True
    assert _run(_w_is_oversize, db, "p/m", 4999) is False


def test_an_affinity_pin_crosses_processes(db):
    """Without this a conversation re-lands on a different model each turn,
    which is the same tool-calling inconsistency cache affinity exists to stop."""
    _run(_w_record_affinity, db, "conv1", "p", "m")
    assert _run(_w_affinity_target, db, "conv1") == ("p", "m")


# ── concurrency, not just visibility ────────────────────────────────────────

def test_concurrent_writers_do_not_lose_updates(db):
    """Four processes writing at once, with no escaped OperationalError.

    busy_timeout plus BEGIN IMMEDIATE is what makes this hold; a deferred
    transaction that reads then writes would deadlock two writers into a
    SQLITE_BUSY the timeout cannot resolve.
    """
    caps = [[f"cap{i}-{j}" for j in range(50)] for i in range(4)]
    procs, queues = [], []
    for chunk in caps:
        q = _CTX.Queue()
        p = _CTX.Process(target=_w_hammer_gaps, args=(db, "p/m", chunk, q))
        p.start()
        procs.append(p)
        queues.append(q)
    won = [q.get(timeout=60) for q in queues]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"

    assert sum(won) == 200, "every distinct gap should be recorded exactly once"
    assert len(_run(_w_read_gaps, db, "p/m")) == 200


def test_the_same_gap_from_four_processes_is_won_once(db):
    """Contention on ONE row rather than two hundred: exactly one writer may be
    told it was new, or the caller persists the same fact N times."""
    procs, queues = [], []
    for _ in range(4):
        q = _CTX.Queue()
        p = _CTX.Process(target=_w_hammer_gaps, args=(db, "p/m", ["tools"], q))
        p.start()
        procs.append(p)
        queues.append(q)
    won = [q.get(timeout=60) for q in queues]
    for p in procs:
        p.join(timeout=60)
    assert sum(won) == 1, f"expected exactly one winner, got {won}"


# ── restart semantics ───────────────────────────────────────────────────────

def test_truncate_clears_what_a_restart_should_not_inherit(db):
    """Counters and cooldowns have never survived a restart. A file-backed store
    would silently make them -- leaving a six-hour-old cooldown on a candidate
    that recovered while the proxy was down."""
    _run(_w_mark_saturated, db, "p/m", 3600)
    _run(_w_record_oversize, db, "p/m", 5000)
    assert _run(_w_is_saturated, db, "p/m") is True

    be = state.SqliteState(db)
    be.truncate()
    be.close()

    assert _run(_w_is_saturated, db, "p/m") is False
    assert _run(_w_is_oversize, db, "p/m", 5000) is False
