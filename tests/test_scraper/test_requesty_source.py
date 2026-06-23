"""Requesty source: $0 models → is_free=True, paid → is_free=False."""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.requesty import REQUESTY_URL, RequestySource


def _set_key(monkeypatch):
    monkeypatch.setenv("REQUESTY_API_KEY", "req-test")


@responses.activate
def test_free_models_emitted(fixtures_dir: Path, monkeypatch):
    _set_key(monkeypatch)
    body = (fixtures_dir / "requesty_models.json").read_text()
    responses.add(responses.GET, REQUESTY_URL, body=body, status=200,
                  content_type="application/json")
    evs = RequestySource().fetch()
    by_id = {e.model_id: e for e in evs}
    assert by_id["requesty/nvidia/nemotron-3-super-120b-a12b"].is_free is True
    assert by_id["requesty/nvidia/nemotron-3-nano-30b-a3b"].is_free is True
    assert by_id["requesty/anthropic/claude-opus-4-7"].is_free is False
    assert by_id["requesty/openai/gpt-5.2"].is_free is False


@responses.activate
def test_paid_models_carry_per_token_pricing(fixtures_dir: Path, monkeypatch):
    _set_key(monkeypatch)
    body = (fixtures_dir / "requesty_models.json").read_text()
    responses.add(responses.GET, REQUESTY_URL, body=body, status=200,
                  content_type="application/json")
    by_id = {e.model_id: e for e in RequestySource().fetch()}
    assert by_id["requesty/anthropic/claude-opus-4-7"].pricing == {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
    }
    # Free models carry no pricing opinion.
    assert by_id["requesty/nvidia/nemotron-3-super-120b-a12b"].pricing is None


def test_no_api_key_returns_empty(monkeypatch):
    """Without REQUESTY_API_KEY, the source yields nothing (no HTTP call)."""
    monkeypatch.delenv("REQUESTY_API_KEY", raising=False)
    assert RequestySource().fetch() == []


@responses.activate
def test_all_evidence_high_confidence(fixtures_dir: Path, monkeypatch):
    _set_key(monkeypatch)
    body = (fixtures_dir / "requesty_models.json").read_text()
    responses.add(responses.GET, REQUESTY_URL, body=body, status=200)
    evs = RequestySource().fetch()
    assert evs
    for e in evs:
        assert e.confidence == "high"
        assert e.source == "requesty"


@responses.activate
def test_auth_header_present(fixtures_dir: Path, monkeypatch):
    """Verify the source sends the Bearer auth header."""
    _set_key(monkeypatch)
    body = (fixtures_dir / "requesty_models.json").read_text()
    rsp = responses.add(responses.GET, REQUESTY_URL, body=body, status=200)
    RequestySource().fetch()
    assert rsp.call_count == 1
    auth = responses.calls[0].request.headers.get("Authorization", "")
    assert auth.startswith("Bearer ")
