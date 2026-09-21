"""State-directory resolution, and what happens when it cannot be written.

The regression these tests exist for: llmproxy decides whether each of its
background refreshes is due by reading a last-run timestamp out of a state file,
and ``_probe_due`` treats a missing timestamp as "run now". A state directory
that could not be written therefore meant the timestamp was never recorded,
every refresh was permanently due, and the once-a-minute interval check in
``server._maybe_fire_interval_probes`` re-fired a full provider scrape for as
long as the process ran. In a container that presented as the proxy degrading
until it was restarted.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from llmproxy import config as cfgmod
from llmproxy.config import (
    get_pr_state_path,
    get_routing_metadata_path,
    get_state_dir,
    get_update_state_path,
    load_flagship_state,
    load_update_state,
    save_flagship_state,
    save_update_state,
)
from scripts.update_free_models import _probe_due


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Each test starts with no LLMPROXY_STATE_DIR and an empty fallback."""
    monkeypatch.delenv("LLMPROXY_STATE_DIR", raising=False)
    cfgmod._state_fallback.clear()
    cfgmod._state_write_warned.clear()
    yield
    cfgmod._state_fallback.clear()
    cfgmod._state_write_warned.clear()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_state_defaults_to_the_config_directory(tmp_path):
    """With LLMPROXY_STATE_DIR unset, the historical layout is preserved."""
    cfg = str(tmp_path / "config.json")
    assert get_state_dir(cfg) == tmp_path
    assert get_routing_metadata_path(cfg) == tmp_path / "routing_metadata.json"
    assert get_update_state_path(cfg) == tmp_path / "update_state.json"
    assert get_pr_state_path(cfg) == tmp_path / "pr_state.json"


def test_state_dir_env_moves_every_state_file(tmp_path, monkeypatch):
    """One variable relocates all of them, so the config mount can be read-only."""
    cfg = str(tmp_path / "config.json")
    state = tmp_path / "state"
    monkeypatch.setenv("LLMPROXY_STATE_DIR", str(state))

    assert get_state_dir(cfg) == state
    assert get_routing_metadata_path(cfg) == state / "routing_metadata.json"
    assert get_update_state_path(cfg) == state / "update_state.json"
    assert get_pr_state_path(cfg) == state / "pr_state.json"


def test_state_dir_is_read_at_call_time(tmp_path, monkeypatch):
    """Like LLMPROXY_CONFIG, so a test or a wizard can move it mid-process."""
    cfg = str(tmp_path / "config.json")
    assert get_state_dir(cfg) == tmp_path
    monkeypatch.setenv("LLMPROXY_STATE_DIR", str(tmp_path / "elsewhere"))
    assert get_state_dir(cfg) == tmp_path / "elsewhere"


# ---------------------------------------------------------------------------
# Read-forward from the pre-state-dir layout
# ---------------------------------------------------------------------------

def test_existing_state_is_read_forward_from_beside_config(tmp_path, monkeypatch):
    """Introducing LLMPROXY_STATE_DIR must not reset every refresh cadence.

    A deployment that upgrades has its state beside config.json. If the new
    location were simply read as empty, every job would come due at once on the
    first boot after the upgrade, which is the very storm this directory exists
    to prevent.
    """
    cfg = str(tmp_path / "config.json")
    stamp = datetime.now(UTC).isoformat()
    assert save_flagship_state({"last_refresh_at": stamp}, cfg) is True
    assert (tmp_path / "flagship_models.json").exists()

    state = tmp_path / "state"
    monkeypatch.setenv("LLMPROXY_STATE_DIR", str(state))

    assert load_flagship_state(cfg) == {"last_refresh_at": stamp}
    # and it is written forward, so the legacy copy stops being consulted
    assert json.loads((state / "flagship_models.json").read_text()) == {
        "last_refresh_at": stamp
    }


def test_state_dir_wins_over_the_legacy_copy(tmp_path, monkeypatch):
    """Once migrated, the state directory is authoritative."""
    cfg = str(tmp_path / "config.json")
    old = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    save_flagship_state({"last_refresh_at": old}, cfg)

    state = tmp_path / "state"
    state.mkdir()
    fresh = datetime.now(UTC).isoformat()
    (state / "flagship_models.json").write_text(json.dumps({"last_refresh_at": fresh}))
    monkeypatch.setenv("LLMPROXY_STATE_DIR", str(state))

    assert load_flagship_state(cfg) == {"last_refresh_at": fresh}


# ---------------------------------------------------------------------------
# The unwritable case: the actual regression
# ---------------------------------------------------------------------------

def _make_unwritable(monkeypatch):
    """Fail every state write the way a read-only mount does.

    Patching mkstemp rather than chmod-ing a directory, because the suite runs
    as root in CI and in the container image, where the permission bits on a
    directory are not enforced.
    """
    def _refuse(*a, **k):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(cfgmod.tempfile, "mkstemp", _refuse)


def test_a_failed_write_is_reported_as_a_failure(tmp_path, monkeypatch):
    """The in-memory fallback must not paper over the failure to the caller."""
    cfg = str(tmp_path / "config.json")
    _make_unwritable(monkeypatch)
    assert save_update_state({"last_update_at": datetime.now(UTC).isoformat()}, cfg) is False


def test_an_unwritable_state_dir_still_throttles_the_refresh(tmp_path, monkeypatch):
    """The regression: no timestamp meant every refresh was permanently due.

    The write still fails and is still reported, but the timestamp is kept in
    memory, so the cadence degrades to "once per process lifetime" rather than
    "on every interval check for as long as the container runs".
    """
    cfg = str(tmp_path / "config.json")
    _make_unwritable(monkeypatch)

    stamp = datetime.now(UTC).isoformat()
    save_update_state({"last_update_at": stamp}, cfg)

    state = load_update_state(cfg)
    assert state == {"last_update_at": stamp}
    due, _ = _probe_due(state.get("last_update_at"), 7)
    assert due is False


def test_the_fallback_wins_over_a_stale_file_on_disk(tmp_path, monkeypatch):
    """A state file can exist while its directory has become unwritable.

    That is the ordinary bind-mount case, since replacing the file needs write
    permission on the directory rather than on the file. The stale on-disk
    timestamp would otherwise keep the refresh due forever.
    """
    cfg = str(tmp_path / "config.json")
    stale = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    save_update_state({"last_update_at": stale}, cfg)
    assert _probe_due(load_update_state(cfg).get("last_update_at"), 7)[0] is True

    _make_unwritable(monkeypatch)
    fresh = datetime.now(UTC).isoformat()
    save_update_state({"last_update_at": fresh}, cfg)

    assert load_update_state(cfg) == {"last_update_at": fresh}
    assert _probe_due(load_update_state(cfg).get("last_update_at"), 7)[0] is False


def test_the_fallback_also_wins_over_a_legacy_copy(tmp_path, monkeypatch):
    """The read-forward must not resurrect a stale timestamp over a newer one.

    With LLMPROXY_STATE_DIR pointing somewhere unwritable, the new location
    never comes into existence, so the read-forward would find the old copy
    beside config.json and hand back its timestamp. If that copy is stale, the
    refresh is due again and the storm is back, in a deployment that looked like
    it had been fixed by setting the variable.
    """
    cfg = str(tmp_path / "config.json")
    stale = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    save_update_state({"last_update_at": stale}, cfg)          # beside config.json

    monkeypatch.setenv("LLMPROXY_STATE_DIR", str(tmp_path / "state"))
    _make_unwritable(monkeypatch)

    fresh = datetime.now(UTC).isoformat()
    save_update_state({"last_update_at": fresh}, cfg)          # fails, held in memory

    assert load_update_state(cfg) == {"last_update_at": fresh}
    assert _probe_due(load_update_state(cfg).get("last_update_at"), 7)[0] is False


def test_the_write_failure_is_reported_once_per_file(tmp_path, monkeypatch, capsys):
    """An unwritable directory must not flood the log on every cadence tick."""
    cfg = str(tmp_path / "config.json")
    _make_unwritable(monkeypatch)

    for _ in range(5):
        save_update_state({"last_update_at": datetime.now(UTC).isoformat()}, cfg)

    out = capsys.readouterr().out
    assert out.count("Failed to write") == 1


def test_a_later_successful_write_clears_the_fallback(tmp_path, monkeypatch):
    """Once the directory is writable again, the file is authoritative."""
    cfg = str(tmp_path / "config.json")
    _make_unwritable(monkeypatch)
    save_update_state({"last_update_at": "in-memory-only"}, cfg)
    assert load_update_state(cfg) == {"last_update_at": "in-memory-only"}

    monkeypatch.undo()
    monkeypatch.delenv("LLMPROXY_STATE_DIR", raising=False)
    stamp = datetime.now(UTC).isoformat()
    assert save_update_state({"last_update_at": stamp}, cfg) is True
    assert get_update_state_path(cfg) not in cfgmod._state_fallback
    assert load_update_state(cfg) == {"last_update_at": stamp}
