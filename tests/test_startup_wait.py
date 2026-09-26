"""Waiting for a late-mounted config.json before the server starts.

On some hosts the container starts before the volume holding config.json is
mounted. Starting anyway would boot the proxy on built-in defaults with no
providers, and the auto-heal step could write a default config.json into the
empty mountpoint. So the server waits with exponential backoff, and exits
non-zero if the file never appears so the restart policy can try again.

The clock is faked through ``m._sleep`` and ``m._monotonic``, so no test really
sleeps.
"""

from __future__ import annotations

import argparse
import json
import logging

import pytest

from llmproxy import __main__ as m

_LOG = logging.getLogger("test.startup_wait")


class _FakeClock:
    """A monotonic clock that only advances when slept on.

    *on_sleep* runs after each sleep, so a test can make the config appear
    partway through the wait.
    """

    def __init__(self, on_sleep=None):
        self.now = 0.0
        self.sleeps: list[float] = []
        self.on_sleep = on_sleep

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep:
            self.on_sleep(len(self.sleeps))


@pytest.fixture
def clock(monkeypatch):
    c = _FakeClock()
    monkeypatch.setattr(m, "_sleep", c.sleep)
    monkeypatch.setattr(m, "_monotonic", c.monotonic)
    return c


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setenv("LLMPROXY_CONFIG", str(path))
    monkeypatch.delenv(m._STARTUP_WAIT_ENV, raising=False)
    return path


def _args(config=None):
    return argparse.Namespace(config=config)


# ── the wait loop ───────────────────────────────────────────────────────────

def test_present_config_returns_without_sleeping(cfg_path, clock):
    cfg_path.write_text("{}")
    assert m._wait_for_config(600, _LOG) is True
    assert clock.sleeps == []


def test_config_that_appears_later_is_picked_up_with_backoff(cfg_path, clock):
    def appear(n):
        if n == 4:
            cfg_path.write_text("{}")

    clock.on_sleep = appear
    assert m._wait_for_config(600, _LOG) is True
    assert clock.sleeps == [1, 2, 4, 8]


def test_unparseable_config_keeps_waiting_until_it_parses(cfg_path, clock):
    """A half-ready network filesystem can expose a file it cannot yet serve."""
    cfg_path.write_text("{not json")

    def fix(n):
        if n == 2:
            cfg_path.write_text(json.dumps({"providers": {}}))

    clock.on_sleep = fix
    assert m._wait_for_config(600, _LOG) is True
    assert len(clock.sleeps) == 2


def test_backoff_is_capped_and_total_wait_respects_the_timeout(cfg_path, clock):
    assert m._wait_for_config(600, _LOG) is False
    assert max(clock.sleeps) == m._STARTUP_WAIT_MAX_DELAY
    assert sum(clock.sleeps) == pytest.approx(600)


def test_zero_timeout_does_not_wait(cfg_path, clock):
    assert m._wait_for_config(0, _LOG) is False
    assert clock.sleeps == []


# ── the timeout setting ─────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    (None, 600), ("", 600), ("120", 120), ("0", 0), ("abc", 600), ("90.5", 90),
])
def test_timeout_env_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(m._STARTUP_WAIT_ENV, raising=False)
    else:
        monkeypatch.setenv(m._STARTUP_WAIT_ENV, raw)
    assert m._startup_wait_timeout() == expected


# ── startup behavior ────────────────────────────────────────────────────────

def test_timeout_exits_nonzero_instead_of_starting_on_defaults(cfg_path, clock,
                                                               monkeypatch):
    monkeypatch.setenv(m._STARTUP_WAIT_ENV, "60")
    with pytest.raises(SystemExit) as exc:
        m._await_config_or_exit(_args())
    assert exc.value.code == 1
    assert not cfg_path.exists(), "nothing may be written to the empty mountpoint"


def test_main_never_imports_server_or_saves_config_on_timeout(cfg_path, clock,
                                                              monkeypatch):
    monkeypatch.setenv(m._STARTUP_WAIT_ENV, "10")
    monkeypatch.setattr("sys.argv", ["llmproxy"])
    saved = []
    monkeypatch.setattr(m, "save_config", lambda *a, **k: saved.append(a))
    with pytest.raises(SystemExit) as exc:
        m.main()
    assert exc.value.code == 1
    assert saved == []
    assert not cfg_path.exists()


def test_wait_disabled_starts_on_defaults(cfg_path, clock, monkeypatch):
    monkeypatch.setenv(m._STARTUP_WAIT_ENV, "0")
    m._await_config_or_exit(_args())  # must not raise
    assert clock.sleeps == []


def test_no_explicit_config_path_means_no_wait(tmp_path, clock, monkeypatch):
    """A bare local run with no config keeps starting on defaults."""
    monkeypatch.delenv("LLMPROXY_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    m._await_config_or_exit(_args())
    assert clock.sleeps == []


def test_temporary_log_handler_is_removed(cfg_path, clock):
    cfg_path.write_text("{}")
    m._await_config_or_exit(_args())
    log = logging.getLogger("llmproxy.startup")
    assert log.handlers == []
    assert log.propagate is True
