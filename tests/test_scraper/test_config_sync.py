"""reconcile_user_config / _sync_user_config: sync a user config's free-tier
sections from the sidecar, scoped to configured providers."""

from __future__ import annotations

import json

from scripts.update_free_models import (
    _sync_user_config,
    reconcile_user_config,
)

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


def test_believed_free_reconciled():
    cfg = _user_cfg()
    changes = reconcile_user_config(_sidecar(), cfg)
    bf = cfg["believed_free"]
    assert "google/added" in bf            # newly free -> added
    assert "google/keep" in bf             # still free -> kept
    assert "github/gone" not in bf         # no longer free -> removed
    assert "custom/mine" in bf             # unconfigured provider -> untouched
    assert "groq/should-not-appear" not in bf  # provider not in user config
    assert "github/stillfree" in bf        # configured + free but missing -> added
    # adds are appended in sorted order
    assert changes["believed_free"]["add"] == ["github/stillfree", "google/added"]
    assert changes["believed_free"]["remove"] == ["github/gone"]


def test_free_limits_reconciled_and_note_preserved():
    cfg = _user_cfg()
    reconcile_user_config(_sidecar(), cfg)
    fl = cfg["free_limits"]
    assert fl["_note"] == "keep me"        # non-model key preserved
    assert "google/added" in fl            # added with limits
    assert "github/gone" not in fl         # removed (no longer free)
    assert "custom/mine" in fl             # untouched


def test_model_reasoning_is_add_only():
    cfg = _user_cfg()
    reconcile_user_config(_sidecar(), cfg)
    mr = cfg["model_reasoning"]
    assert mr["github/gone"] == "deep"     # NOT pruned despite leaving believed_free
    assert mr["google/added"] == "deep"    # newly added
    assert mr["custom/mine"] == "standard"  # untouched
    assert "groq/should-not-appear" not in mr  # unconfigured provider ignored


def test_idempotent():
    sidecar = _sidecar()
    cfg = _user_cfg()
    reconcile_user_config(sidecar, cfg)
    second = reconcile_user_config(sidecar, cfg)
    assert second["believed_free"] == {"add": [], "remove": []}
    assert second["free_limits"] == {"set": [], "remove": []}
    assert second["model_reasoning"] == {"add": []}


def test_dry_run_writes_nothing(tmp_path):
    p = tmp_path / "config.json"
    original = _user_cfg()
    p.write_text(json.dumps(original, indent=2), encoding="utf-8")
    rc = _sync_user_config(_sidecar(), str(p), dry_run=True)
    assert rc == 0
    assert json.loads(p.read_text()) == original  # untouched on disk


def test_sync_writes_when_not_dry_run(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_user_cfg(), indent=2), encoding="utf-8")
    rc = _sync_user_config(_sidecar(), str(p), dry_run=False)
    assert rc == 0
    written = json.loads(p.read_text())
    assert "google/added" in written["believed_free"]
    assert "github/gone" not in written["believed_free"]
    assert written["model_reasoning"]["github/gone"] == "deep"


def test_missing_file_returns_error(tmp_path):
    rc = _sync_user_config(_sidecar(), str(tmp_path / "nope.json"), dry_run=False)
    assert rc == 2


def test_model_capabilities_is_add_only():
    sidecar = {
        "providers": {
            "google": {
                "base_url": "u", "display": "G",
                "believed_free": ["google/added"],
                "model_capabilities": {
                    "google/added": ["tools", "vision"],
                    "google/keep": ["tools"],
                },
            },
            "groq": {  # not configured by user -> ignored
                "base_url": "u", "display": "Q",
                "believed_free": [],
                "model_capabilities": {"groq/skip": ["tools"]},
            },
        },
        "provider_order": ["google", "groq"],
    }
    cfg = {
        "providers": {
            "google": {"base_url": "u", "api_key": "k"},
            "custom": {"base_url": "u", "api_key": "k"},
        },
        "model_capabilities": {
            "google/keep": ["reasoning"],   # user-set -> must NOT be overwritten
            "custom/mine": ["tools"],       # unconfigured provider -> untouched
        },
    }
    changes = reconcile_user_config(sidecar, cfg)
    mc = cfg["model_capabilities"]
    assert mc["google/added"] == ["tools", "vision"]   # newly added
    assert mc["google/keep"] == ["reasoning"]          # add-only: NOT overwritten
    assert mc["custom/mine"] == ["tools"]              # untouched
    assert "groq/skip" not in mc                       # unconfigured provider ignored
    assert changes["model_capabilities"]["add"] == ["google/added"]
