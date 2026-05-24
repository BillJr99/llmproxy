"""Fireworks source: is_free=True flag → is_free, billing=metered → not."""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.fireworks import FIREWORKS_URL, FireworksSource


def _set_key(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test")


@responses.activate
def test_free_models_emitted(fixtures_dir: Path, monkeypatch):
    _set_key(monkeypatch)
    body = (fixtures_dir / "fireworks_models.json").read_text()
    responses.add(responses.GET, FIREWORKS_URL, body=body, status=200,
                  content_type="application/json")
    evs = FireworksSource().fetch()
    by_id = {e.model_id: e for e in evs}
    assert by_id["fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct"].is_free is True
    assert by_id["fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct"].is_free is True
    # Metered with non-zero pricing.
    assert by_id["fireworks/accounts/fireworks/models/firefunction-v2"].is_free is False
    # Zero pricing alone is sufficient to mark free even without is_free flag.
    assert by_id["fireworks/accounts/fireworks/models/deepseek-r1-basic"].is_free is True


def test_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    assert FireworksSource().fetch() == []


@responses.activate
def test_all_evidence_high_confidence(fixtures_dir: Path, monkeypatch):
    _set_key(monkeypatch)
    body = (fixtures_dir / "fireworks_models.json").read_text()
    responses.add(responses.GET, FIREWORKS_URL, body=body, status=200)
    evs = FireworksSource().fetch()
    assert evs
    for e in evs:
        assert e.confidence == "high"
        assert e.source == "fireworks"
