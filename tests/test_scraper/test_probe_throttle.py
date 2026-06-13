"""Tests for the probe-frequency throttle (scripts/update_free_models._probe_due)
and the probe-state cache helpers (llmproxy.config)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llmproxy.config import get_probe_state_path, load_probe_state, save_probe_state
from scripts.update_free_models import _probe_due


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


# --- _probe_due -------------------------------------------------------------

def test_frequency_zero_always_due():
    due, _ = _probe_due(_iso(0.0), 0)
    assert due is True


def test_frequency_negative_always_due():
    due, _ = _probe_due(_iso(0.0), -5)
    assert due is True


def test_no_timestamp_is_due():
    due, since = _probe_due(None, 7)
    assert due is True
    assert since is None


def test_recent_probe_is_throttled():
    due, since = _probe_due(_iso(0.2), 1)
    assert due is False
    assert since is not None and since < 1


def test_old_probe_is_due():
    due, since = _probe_due(_iso(3.0), 1)
    assert due is True
    assert since is not None and since >= 1


def test_exactly_at_frequency_is_due():
    due, _ = _probe_due(_iso(1.0001), 1)
    assert due is True


def test_invalid_timestamp_is_due():
    due, since = _probe_due("not-a-timestamp", 1)
    assert due is True
    assert since is None


def test_naive_timestamp_treated_as_utc():
    naive = (datetime.now(timezone.utc) - timedelta(days=0.1)).replace(tzinfo=None).isoformat()
    due, _ = _probe_due(naive, 1)
    assert due is False


def test_non_numeric_frequency_defaults_to_always_due():
    due, _ = _probe_due(_iso(0.0), "garbage")
    assert due is True


# --- probe-state cache helpers ---------------------------------------------

def test_probe_state_roundtrip(tmp_path):
    cfg = str(tmp_path / "config.json")
    assert load_probe_state(cfg) == {}
    ts = datetime.now(timezone.utc).isoformat()
    assert save_probe_state({"last_probe_at": ts}, cfg) is True
    assert get_probe_state_path(cfg) == tmp_path / "probe_state.json"
    assert load_probe_state(cfg) == {"last_probe_at": ts}


def test_probe_state_corrupt_returns_empty(tmp_path):
    cfg = str(tmp_path / "config.json")
    get_probe_state_path(cfg).write_text("{not json", encoding="utf-8")
    assert load_probe_state(cfg) == {}
