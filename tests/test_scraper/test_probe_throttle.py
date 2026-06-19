"""Tests for the probe-frequency throttle (scripts/update_free_models._probe_due)
and the cost-probe-state cache helpers (llmproxy.config)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from llmproxy.config import (
    get_cost_probe_state_path,
    load_cost_probe_state,
    save_cost_probe_state,
)
from scripts.update_free_models import _probe_due


def _iso(days_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


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
    naive = (datetime.now(UTC) - timedelta(days=0.1)).replace(tzinfo=None).isoformat()
    due, _ = _probe_due(naive, 1)
    assert due is False


def test_non_numeric_frequency_defaults_to_always_due():
    due, _ = _probe_due(_iso(0.0), "garbage")
    assert due is True


# --- cost-probe-state cache helpers -----------------------------------------

def test_probe_state_roundtrip(tmp_path):
    cfg = str(tmp_path / "config.json")
    assert load_cost_probe_state(cfg) == {}
    ts = datetime.now(UTC).isoformat()
    assert save_cost_probe_state({"last_probe_at": ts}, cfg) is True
    assert get_cost_probe_state_path(cfg) == tmp_path / "cost_probe_state.json"
    assert load_cost_probe_state(cfg) == {"last_probe_at": ts}


def test_probe_state_corrupt_returns_empty(tmp_path):
    cfg = str(tmp_path / "config.json")
    get_cost_probe_state_path(cfg).write_text("{not json", encoding="utf-8")
    assert load_cost_probe_state(cfg) == {}


class _ReadOnlyPath:
    """Stand-in for a sidecar on a read-only image layer."""

    def write_text(self, *a, **k):
        raise PermissionError("read-only image layer")

    def __str__(self):
        return "/app/llmproxy/providers.json"

    __repr__ = __str__


def test_probe_timestamp_recorded_when_sidecar_write_fails(tmp_path, monkeypatch):
    """A read-only providers.json must not prevent the probe-throttle timestamp
    (which lives next to the user config, e.g. /config/cost_probe_state.json) from
    advancing — regression for the container bind-mount case."""
    import scripts.update_free_models as ufm

    cfg = tmp_path / "config.json"   # stands in for /config/config.json
    cfg.write_text(json.dumps({"providers": {}}))

    # Force the run to reach the sidecar write and fail there.
    monkeypatch.setattr(ufm, "apply_updates", lambda *a, **k: True)
    monkeypatch.setattr(ufm, "DATA_PATH", _ReadOnlyPath())

    rc = ufm.main(["--source", "cost_probe", "--config", str(cfg)])

    assert rc == 0  # the run completed despite the read-only sidecar
    state = load_cost_probe_state(str(cfg))
    assert "last_probe_at" in state  # throttle timestamp persisted to the bind mount


def test_sidecar_mirrored_to_config_dir_when_readonly(tmp_path, monkeypatch):
    """When the bundled providers.json is read-only, the computed providers.json
    and config.example.json are mirrored to the user-config dir so a read-only
    deployment can still review them / open a providers PR."""
    import scripts.update_free_models as ufm

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"providers": {}}))
    monkeypatch.setattr(ufm, "apply_updates", lambda *a, **k: True)
    monkeypatch.setattr(ufm, "DATA_PATH", _ReadOnlyPath())

    ufm.main(["--source", "cost_probe", "--config", str(cfg)])

    mirrored_providers = tmp_path / "providers.json"
    mirrored_example = tmp_path / "config.example.json"
    assert mirrored_providers.exists() and mirrored_example.exists()
    assert "providers" in json.loads(mirrored_providers.read_text())
    assert "providers" in json.loads(mirrored_example.read_text())
