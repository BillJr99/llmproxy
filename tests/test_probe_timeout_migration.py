"""free_tier.endpoint_probe was flattened; old configs must keep working.

The block held two keys. `frequency_minutes` was the de-facto master cadence
before the sweep got an explicit one, and is now obsolete — the endpoint probe
spends no quota and simply runs on every sweep. `timeout_sec` was its only
remaining content, and moved to `free_tier.probe_timeout_sec`, shared with the
cost probe. Both probes wait the same amount for a slow provider; they differ
in what they spend, not in how patient to be.
"""

from __future__ import annotations

from llmproxy.config import DEFAULT_FREE_TIER_CONFIG, _normalize_config


def _ft(raw: dict) -> dict:
    return _normalize_config({"free_tier": raw})["free_tier"]


def test_timeout_sec_moves_to_the_shared_key():
    got = _ft({"endpoint_probe": {"timeout_sec": 30}})
    assert got["probe_timeout_sec"] == 30


def test_the_emptied_block_is_removed():
    """Leaving an empty endpoint_probe behind would imply it still does
    something."""
    got = _ft({"endpoint_probe": {"timeout_sec": 30, "frequency_minutes": 30}})
    assert "endpoint_probe" not in got


def test_frequency_minutes_is_dropped_not_migrated():
    """There is no new key for it to become: the probe no longer throttles."""
    got = _ft({"endpoint_probe": {"frequency_minutes": 30}})
    assert "frequency_minutes" not in str(got)
    assert "probe_timeout_sec" not in got  # nothing invented from a dropped key


def test_an_explicit_new_key_wins_over_the_legacy_one():
    got = _ft({"probe_timeout_sec": 5, "endpoint_probe": {"timeout_sec": 30}})
    assert got["probe_timeout_sec"] == 5


def test_a_block_with_unknown_keys_survives():
    """Only the two known keys are consumed; anything else is left alone rather
    than silently discarded."""
    got = _ft({"endpoint_probe": {"timeout_sec": 30, "something_else": 1}})
    assert got["endpoint_probe"] == {"something_else": 1}
    assert got["probe_timeout_sec"] == 30


def test_a_modern_config_is_untouched():
    modern = {"probe_timeout_sec": 15, "cost_probe": {"enabled": True}}
    assert _ft(dict(modern)) == modern


def test_defaults_use_the_flat_key():
    assert DEFAULT_FREE_TIER_CONFIG["probe_timeout_sec"] == 10
    assert "endpoint_probe" not in DEFAULT_FREE_TIER_CONFIG


def test_config_example_matches_the_defaults():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    ft = json.loads((root / "config.example.json").read_text(encoding="utf-8"))["free_tier"]
    assert ft["probe_timeout_sec"] == DEFAULT_FREE_TIER_CONFIG["probe_timeout_sec"]
    assert "endpoint_probe" not in ft


def test_both_probes_accept_the_shared_timeout():
    """The point of one key: it must actually reach both sources."""
    from scripts.sources.cost_probe import CostProbeSource
    from scripts.sources.endpoint_probe import EndpointProbeSource
    assert CostProbeSource(timeout=42).timeout == (5, 42)
    assert EndpointProbeSource(timeout=42).timeout == 42


# ── one source of truth for the free_tier defaults ──────────────────────────

def test_runtime_defaults_match_the_generated_example_exactly():
    """config.py defines the defaults; update_free_models.py writes them into
    config.example.json. Two literals would drift, which is how
    update_frequency_days ended up in the example but not the runtime."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    example = json.loads((root / "config.example.json").read_text(encoding="utf-8"))
    assert example["free_tier"] == DEFAULT_FREE_TIER_CONFIG


def test_server_cadence_constant_comes_from_the_shared_default():
    from llmproxy.server import DEFAULT_UPDATE_FREQUENCY_DAYS
    assert DEFAULT_UPDATE_FREQUENCY_DAYS == DEFAULT_FREE_TIER_CONFIG["update_frequency_days"]


def test_the_admin_api_exposes_the_cadence_and_the_timeout():
    """The admin UI could previously set the cost-probe throttle but not the
    cadence that actually governs the sweep."""
    from llmproxy.admin import _maintenance_view
    view = _maintenance_view({})
    assert view["update_frequency_days"] == DEFAULT_FREE_TIER_CONFIG["update_frequency_days"]
    assert view["probe_timeout_sec"] == DEFAULT_FREE_TIER_CONFIG["probe_timeout_sec"]


def test_the_admin_api_reads_user_values_for_both():
    from llmproxy.admin import _maintenance_view
    view = _maintenance_view({"free_tier": {"update_frequency_days": 3, "probe_timeout_sec": 30}})
    assert view["update_frequency_days"] == 3
    assert view["probe_timeout_sec"] == 30
