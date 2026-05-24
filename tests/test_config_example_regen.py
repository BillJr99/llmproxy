"""Verify the config.example.json regenerator is deterministic and matches
the committed config.example.json byte-for-byte."""

from __future__ import annotations

import json
from pathlib import Path

from llmproxy.free_models import load_data
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


def test_regen_includes_every_believed_free_from_sidecar():
    sidecar = load_data()
    expected_models: set[str] = set()
    for prov in sidecar["providers"].values():
        expected_models.update(prov.get("believed_free", []))

    actual = regenerate_config_example(sidecar)
    assert set(actual["believed_free"]) == expected_models


def test_regen_preserves_static_providers():
    """openai and ollama are not in the sidecar — but the regen must
    continue to include them as example providers."""
    actual = regenerate_config_example(load_data())
    assert "openai" in actual["providers"]
    assert "ollama" in actual["providers"]
