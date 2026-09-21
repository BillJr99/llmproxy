"""Choosing a state backend at startup, and what happens when it cannot be opened.

`server.workers > 1` is only safe when the routing state is shared. The startup
path therefore opens the store for real before forking, and if it cannot, drops
the worker count to 1.

That trade is the point of these tests. Falling back quietly to per-worker state
would reinstate exactly the bug the store exists to prevent -- every free-tier
quota counted N times, no cooldown ever crossing a worker -- while the proxy
looked perfectly healthy. Refusing to boot would turn a recoverable
misconfiguration into an outage, and `tests/test_state_dir.py` records what an
unwritable state directory has already cost this project once. Losing CPU
parallelism is the only one of the three that costs nothing but speed.

Failures are simulated with an unopenable path rather than `chmod`, because the
suite runs as root in CI and in the container image, where a directory's
permission bits do not apply -- the same reason `test_state_dir.py` patches
`mkstemp`.
"""

from __future__ import annotations

import logging

import pytest

from llmproxy import __main__ as m
from llmproxy import state

# A directory that cannot be created, for root or anyone else.
_UNOPENABLE = "/proc/llmproxy-does-not-exist/shared_state.db"


@pytest.fixture(autouse=True)
def _clean_choice():
    state.configure(None)
    yield
    state.configure(None)
    state.set_backend(None)


# ── the probe ───────────────────────────────────────────────────────────────

def test_a_good_path_probes_clean(tmp_path):
    assert state.probe_shared_store(tmp_path / "shared_state.db") is None


def test_an_unopenable_path_reports_why(tmp_path):
    """The reason is carried, not just a bool: an unwritable directory, a
    filesystem that cannot support WAL, and a full disk all fail differently
    and the operator needs to know which."""
    reason = state.probe_shared_store(_UNOPENABLE)
    assert reason is not None
    assert "Error" in reason or "error" in reason


# ── selection ───────────────────────────────────────────────────────────────

def test_one_worker_shares_nothing_and_pays_nothing(tmp_path, monkeypatch):
    """With a single worker there is nothing to share, so the proxy must not
    open a database for a feature it is not using."""
    monkeypatch.setattr(m, "get_state_dir", lambda: tmp_path)
    assert m._prepare_shared_state(1, {}) == 1
    assert state.shared_db_path() is None
    assert state.get_backend().kind == "memory"


def test_several_workers_share_through_the_state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "get_state_dir", lambda: tmp_path)
    assert m._prepare_shared_state(4, {}) == 4
    assert state.shared_db_path() == str(tmp_path / "shared_state.db")
    assert state.get_backend().kind == "sqlite"


def test_an_unopenable_store_drops_to_one_worker(tmp_path, monkeypatch, caplog):
    """Correct on one worker beats fast and wrong on four."""
    monkeypatch.setattr(m, "get_state_dir", lambda: "/proc/llmproxy-does-not-exist")
    with caplog.at_level(logging.ERROR, logger="llmproxy"):
        assert m._prepare_shared_state(4, {}) == 1
    assert state.shared_db_path() is None
    assert state.get_backend().kind == "memory"


def test_the_fallback_is_loud_and_names_the_remedy(tmp_path, monkeypatch, caplog):
    """A silent fallback is the failure mode this whole path exists to avoid."""
    monkeypatch.setattr(m, "get_state_dir", lambda: "/proc/llmproxy-does-not-exist")
    with caplog.at_level(logging.ERROR, logger="llmproxy"):
        m._prepare_shared_state(4, {})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "ONE worker" in text
    assert "shared_state.db" in text
    assert "LLMPROXY_STATE_DIR" in text, "the message must name the remedy"


# ── fork safety ─────────────────────────────────────────────────────────────

def test_a_worker_reset_keeps_the_choice_but_drops_the_instance(tmp_path, monkeypatch):
    """Every worker must build the SAME kind of backend, and its OWN instance.

    Sharing a SQLite connection across a fork() is the classic way to corrupt
    one, so the object cannot be inherited -- but the decision must be.
    """
    monkeypatch.setattr(m, "get_state_dir", lambda: tmp_path)
    m._prepare_shared_state(2, {})
    first = state.get_backend()
    state.reset_for_worker()
    second = state.get_backend()
    assert second is not first
    assert second.kind == first.kind == "sqlite"
    assert state.shared_db_path() == str(tmp_path / "shared_state.db")


def test_the_store_is_truncated_before_workers_start(tmp_path, monkeypatch):
    """Counters and cooldowns have never survived a restart; a file-backed store
    would silently make them."""
    monkeypatch.setattr(m, "get_state_dir", lambda: tmp_path)
    m._prepare_shared_state(2, {})
    state.get_backend().mark_saturated("p/m", 3600)
    assert state.get_backend().is_saturated("p/m") is True

    state.set_backend(None)
    m._prepare_shared_state(2, {})        # a restart
    assert state.get_backend().is_saturated("p/m") is False
