"""Cadence gate for the full free-models refresh.

`free_tier.update_frequency_days` throttles the network scrape so a
restart-heavy deployment re-scrapes on its configured cadence rather than on
every boot, while a long-lived process still refreshes without a cron.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from llmproxy.config import (
    get_update_state_path,
    load_update_state,
    save_update_state,
)
from llmproxy.server import DEFAULT_UPDATE_FREQUENCY_DAYS, _free_update_due


def _iso(days_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


# --- update-state cache helpers ---------------------------------------------

def test_update_state_roundtrip(tmp_path):
    cfg = str(tmp_path / "config.json")
    assert load_update_state(cfg) == {}
    ts = datetime.now(UTC).isoformat()
    assert save_update_state({"last_update_at": ts}, cfg) is True
    assert get_update_state_path(cfg) == tmp_path / "update_state.json"
    assert load_update_state(cfg) == {"last_update_at": ts}


def test_update_state_corrupt_returns_empty(tmp_path):
    cfg = str(tmp_path / "config.json")
    get_update_state_path(cfg).write_text("{not json", encoding="utf-8")
    assert load_update_state(cfg) == {}


# --- _free_update_due --------------------------------------------------------

def test_first_run_is_due(tmp_path):
    """With no recorded run, the refresh fires — a fresh deployment scrapes."""
    cfg = str(tmp_path / "config.json")
    assert _free_update_due({}, cfg) is True


def test_recent_refresh_is_throttled(tmp_path):
    """The second boot of a restart-heavy deployment must not re-scrape."""
    cfg = str(tmp_path / "config.json")
    save_update_state({"last_update_at": _iso(1.0)}, cfg)
    assert _free_update_due({"update_frequency_days": 7}, cfg) is False


def test_stale_refresh_is_due(tmp_path):
    cfg = str(tmp_path / "config.json")
    save_update_state({"last_update_at": _iso(9.0)}, cfg)
    assert _free_update_due({"update_frequency_days": 7}, cfg) is True


def test_default_cadence_is_weekly(tmp_path):
    """An existing config with no update_frequency_days key gets the default."""
    cfg = str(tmp_path / "config.json")
    assert DEFAULT_UPDATE_FREQUENCY_DAYS == 7
    save_update_state({"last_update_at": _iso(6.0)}, cfg)
    assert _free_update_due({}, cfg) is False
    save_update_state({"last_update_at": _iso(8.0)}, cfg)
    assert _free_update_due({}, cfg) is True


def test_zero_frequency_always_due(tmp_path):
    """0 means refresh every time the interval is checked."""
    cfg = str(tmp_path / "config.json")
    save_update_state({"last_update_at": _iso(0.0)}, cfg)
    assert _free_update_due({"update_frequency_days": 0}, cfg) is True


def test_scrape_records_the_refresh_timestamp(tmp_path, monkeypatch):
    """A completed run must advance the cadence, so the next boot is throttled."""
    import scripts.update_free_models as ufm

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"providers": {}}))
    monkeypatch.setattr(ufm, "apply_updates", lambda *a, **k: True)
    # Stubbing apply_updates to True sends main() down its "changed" branch,
    # which writes DATA_PATH — the REPO's llmproxy/providers.json. Redirect both
    # writes at a tmp path: a test must never mutate a tracked file, and this one
    # silently did until a canonicalization change made the rewrite visible.
    monkeypatch.setattr(ufm, "DATA_PATH", tmp_path / "providers.json")
    monkeypatch.setattr(ufm, "CONFIG_EXAMPLE_PATH", tmp_path / "config.example.json")
    monkeypatch.setattr(ufm, "write_config_example", lambda *a, **k: None)

    assert ufm.main(["--source", "cost_probe", "--config", str(cfg)]) == 0
    assert (tmp_path / "providers.json").exists(), \
        "the sidecar write should have gone to the tmp path, not the repo"

    assert "last_update_at" in load_update_state(str(cfg))
    assert _free_update_due({"update_frequency_days": 7}, str(cfg)) is False
