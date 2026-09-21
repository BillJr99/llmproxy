"""_sync_user_config: the config sync is a no-op, and must stay one.

reconcile_user_config used to copy the sidecar's free-tier sections into the
user's config.json. That copy is exactly what froze them - it ran once and the
data never moved again - so it was removed along with its tests. providers.json
is now read directly as the defaults layer, routing_metadata.json sits above it,
and config.json holds only deliberate overrides.

What is left here pins the contract that matters: the entry point and its flags
still work, and nothing they touch ever writes to the user's file."""

from __future__ import annotations

import json

import scripts.update_free_models as ufm
from scripts.update_free_models import _sync_user_config, main

_LIM = {"requests_per_minute": 30, "requests_per_day": 1000,
        "tokens_per_minute": None, "tokens_per_day": None}


def _sidecar() -> dict:
    return {
        "providers": {
            "google": {
                "base_url": "u", "display": "G",
                "believed_free": ["google/keep", "google/added"],
                "model_reasoning": {"google/keep": "standard", "google/added": "deep"},
                "free_limits": {"google/keep": _LIM, "google/added": _LIM},
            },
            # Configured by the user but the sidecar no longer lists 'gone' as free.
            "github": {
                "base_url": "u", "display": "GH",
                "believed_free": ["github/stillfree"],
                "model_reasoning": {"github/stillfree": "standard", "github/gone": "deep"},
                "free_limits": {"github/stillfree": _LIM},
            },
            # In the sidecar but NOT configured by the user — must be ignored.
            "groq": {
                "base_url": "u", "display": "Q",
                "believed_free": ["groq/should-not-appear"],
                "model_reasoning": {"groq/should-not-appear": "standard"},
                "free_limits": {},
            },
        },
        "provider_order": ["google", "github", "groq"],
    }


def _user_cfg() -> dict:
    return {
        "providers": {
            "google": {"base_url": "u", "api_key": "k"},
            "github": {"base_url": "u", "api_key": "k"},
            "custom": {"base_url": "u", "api_key": "k"},  # not in sidecar
        },
        "believed_free": [
            "google/keep",
            "github/gone",        # configured provider, no longer free -> removed
            "custom/mine",        # unconfigured-by-sidecar provider -> untouched
        ],
        "model_reasoning": {
            "github/gone": "deep",   # must NOT be pruned
            "custom/mine": "standard",
        },
        "free_limits": {
            "_note": "keep me",
            "google/keep": _LIM,
            "github/gone": _LIM,     # configured + no longer free -> removed
            "custom/mine": _LIM,     # untouched
        },
    }


def test_dry_run_writes_nothing(tmp_path):
    p = tmp_path / "config.json"
    original = _user_cfg()
    p.write_text(json.dumps(original, indent=2), encoding="utf-8")
    rc = _sync_user_config(_sidecar(), str(p), dry_run=True)
    assert rc == 0
    assert json.loads(p.read_text()) == original  # untouched on disk


def test_sync_never_writes_even_when_not_dry_run(tmp_path):
    """Not a dry run, and still nothing written.

    The sync used to reconcile the sidecar's free-tier sections into the user's
    config.json. That copy is what froze them at first run, so it is gone:
    providers.json is read directly as the defaults layer instead, and the
    user's file is now purely overrides.
    """
    p = tmp_path / "config.json"
    original = json.dumps(_user_cfg(), indent=2)
    p.write_text(original, encoding="utf-8")
    rc = _sync_user_config(_sidecar(), str(p), dry_run=False)
    assert rc == 0
    assert p.read_text(encoding="utf-8") == original


def test_sync_is_a_no_op_and_cannot_fail(tmp_path):
    """Nothing to sync, so a missing config is not an error any more: the runtime
    reads providers.json directly as the defaults layer."""
    rc = _sync_user_config(_sidecar(), str(tmp_path / "nope.json"), dry_run=False)
    assert rc == 0


def test_sync_config_only_leaves_the_user_config_alone(tmp_path, monkeypatch):
    """The routing metadata no longer gets copied into config.json.

    Copying it is what froze it: the sync ran once and the data never moved
    again. providers.json is now read directly as the defaults layer, so the
    user's file must come back byte-for-byte unchanged.
    """
    p = tmp_path / "config.json"
    original = json.dumps(_user_cfg(), indent=2)
    p.write_text(original, encoding="utf-8")
    monkeypatch.setattr(ufm, "load_data", _sidecar)

    def _boom(*_a, **_k):  # pragma: no cover - only fails if wrongly called
        raise AssertionError("--sync-config-only must not write the sidecar")
    monkeypatch.setattr(ufm, "write_config_example", _boom)

    rc = main(["--sync-config-only", "--config", str(p)])
    assert rc == 0
    assert p.read_text(encoding="utf-8") == original


def test_sync_config_only_requires_config():
    assert main(["--sync-config-only"]) == 2


def test_sync_config_only_dry_run_writes_nothing(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    original = _user_cfg()
    p.write_text(json.dumps(original, indent=2), encoding="utf-8")
    monkeypatch.setattr(ufm, "load_data", _sidecar)
    rc = main(["--sync-config-only", "--dry-run", "--config", str(p)])
    assert rc == 0
    assert json.loads(p.read_text()) == original  # untouched on disk
