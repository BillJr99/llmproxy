"""Verify the config.example.json regenerator is deterministic and matches
the committed config.example.json byte-for-byte."""

from __future__ import annotations

import json
from pathlib import Path

from llmproxy.providers import load_data
from scripts.update_free_models import (
    CONFIG_EXAMPLE_PATH,
    regenerate_config_example,
    write_config_example,
)


def test_regen_is_idempotent(tmp_path: Path):
    out1 = tmp_path / "c1.json"
    out2 = tmp_path / "c2.json"
    write_config_example(out1)
    write_config_example(out2)
    assert out1.read_text() == out2.read_text()


def test_regen_matches_committed_config_example():
    """The committed config.example.json must equal the regenerator output —
    the CI guard depends on this."""
    expected = json.loads(CONFIG_EXAMPLE_PATH.read_text())
    actual = regenerate_config_example(load_data())
    assert actual == expected


def test_regen_omits_routing_metadata_entirely():
    """The example must not carry the five routing keys.

    config.json is drained into the sidecar's CURATED layer at startup, the
    strongest layer there is. Shipping providers.json's defaults inside the
    example would pin every one of them as though a person had set it by hand,
    at a precedence no later refresh could improve — so a fresh install would be
    frozen on whatever the repo knew on the day it was copied.

    The data still reaches routing, as the defaults layer read straight from
    providers.json, which is where the providers PR keeps it current.
    """
    actual = regenerate_config_example(load_data())
    for key in ("believed_free", "cost_observed_free_tier", "model_reasoning",
                "model_capabilities", "free_limits"):
        assert key not in actual, (
            f"config.example.json still ships {key!r}; a fresh config.json would "
            "have it drained into the curated layer and frozen there"
        )


def test_regen_still_carries_the_provider_catalogue():
    """Dropping the routing keys must not drop the providers with them."""
    actual = regenerate_config_example(load_data())
    sidecar = load_data()
    for key in sidecar["providers"]:
        assert key in actual["providers"], f"provider {key} missing from the example"


def test_regen_preserves_static_providers():
    """openai and ollama are not in the sidecar — but the regen must
    continue to include them as example providers."""
    actual = regenerate_config_example(load_data())
    assert "openai" in actual["providers"]
    assert "ollama" in actual["providers"]
