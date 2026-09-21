"""The gunicorn settings llmproxy boots with, and which knob does what.

`server.threads` was hardcoded at 4 for a long time while `server.workers` — the
knob that costs correctness — was the only one exposed. For a proxy that spends
its wall time blocked on upstreams, that is backwards: a streamed request holds
its gthread thread for the whole upstream duration, so threads are the real
ceiling on concurrent requests, and they share process memory so every registry
the routing layer keeps stays correct across them.

These tests assert the settings actually reach gunicorn, rather than asserting
that the config helper parses an int (which `test_capability_states.py` already
covers). Booting a server is the only other way to check that.
"""

from __future__ import annotations

import pytest

from llmproxy import __main__ as m


def _opts(server_cfg: dict) -> dict:
    return m._gunicorn_options(server_cfg, "127.0.0.1", 8080, "INFO", None)


# ── threads ─────────────────────────────────────────────────────────────────

def test_threads_defaults_to_four():
    """Unchanged for a config that does not mention it."""
    assert _opts({})["threads"] == 4


def test_threads_is_configurable():
    """The point of the change: the ceiling on concurrent requests is settable."""
    assert _opts({"threads": 32})["threads"] == 32


@pytest.mark.parametrize("raw,expected", [
    ("16", 16), (None, 4), (True, 4), ("abc", 4), ({}, 4), (-1, 1), (0, 1),
])
def test_threads_is_defensive(raw, expected):
    """A hand-edited config must start the proxy, not refuse to.

    Zero and negative clamp to 1 rather than to the default: someone who wrote
    0 wants as few as possible, and gunicorn rejects a thread count below 1.
    """
    assert _opts({"threads": raw})["threads"] == expected


# ── workers, and the relationship between the two ───────────────────────────

def test_workers_still_defaults_to_one():
    """Exposing threads must not quietly make multi-worker the default."""
    assert _opts({})["workers"] == 1


def test_threads_and_workers_are_independent():
    got = _opts({"threads": 8, "workers": 2})
    assert (got["workers"], got["threads"]) == (2, 8)


def test_raising_threads_does_not_raise_workers():
    """The whole recommendation: serve more concurrency without the accounting
    drift that a second process brings."""
    assert _opts({"threads": 64})["workers"] == 1


# ── the settings that were already there keep working ───────────────────────

def test_worker_class_is_gthread():
    """Threads only mean anything under a threaded worker class."""
    assert _opts({})["worker_class"] == "gthread"


def test_timeout_tracks_stream_timeout_with_a_floor():
    """gunicorn's worker timeout must outlast the longest request llmproxy will
    wait out, or the worker is killed mid-stream."""
    assert _opts({})["timeout"] == 300                      # default stream_timeout
    assert _opts({"stream_timeout": 600})["timeout"] == 600
    assert _opts({"stream_timeout": 5})["timeout"] == 120   # floor


def test_bind_and_loglevel():
    got = m._gunicorn_options({}, "0.0.0.0", 9000, "DEBUG", None)
    assert got["bind"] == "0.0.0.0:9000"
    assert got["loglevel"] == "debug"


def test_post_worker_init_is_passed_through():
    """It resets per-worker state after the fork; losing it would be silent."""
    sentinel = object()
    assert m._gunicorn_options({}, "h", 1, "INFO", sentinel)["post_worker_init"] is sentinel
