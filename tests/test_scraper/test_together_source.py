"""Together AI source: $0 chat models → is_free=True, paid → is_free=False."""

from __future__ import annotations

import os
from pathlib import Path

import responses

from scripts.sources.together import TOGETHER_URL, TogetherSource


def _set_key(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "tgp-test")


@responses.activate
def test_free_chat_models_emitted(fixtures_dir: Path, monkeypatch):
    _set_key(monkeypatch)
    body = (fixtures_dir / "together_models.json").read_text()
    responses.add(responses.GET, TOGETHER_URL, body=body, status=200,
                  content_type="application/json")
    evs = TogetherSource().fetch()
    by_id = {e.model_id: e for e in evs}
    assert by_id["together/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"].is_free is True
    assert by_id["together/meta-llama/Llama-Vision-Free"].is_free is True
    assert by_id["together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"].is_free is False


@responses.activate
def test_paid_models_carry_per_token_pricing(fixtures_dir: Path, monkeypatch):
    _set_key(monkeypatch)
    body = (fixtures_dir / "together_models.json").read_text()
    responses.add(responses.GET, TOGETHER_URL, body=body, status=200,
                  content_type="application/json")
    by_id = {e.model_id: e for e in TogetherSource().fetch()}
    # Together quotes prices per 1M tokens (3.5) → convert to per token (3.5e-6).
    assert by_id["together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"].pricing == {
        "input_cost_per_token": 3.5e-6,
        "output_cost_per_token": 3.5e-6,
    }
    # Free models carry no pricing opinion.
    assert by_id["together/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"].pricing is None


@responses.activate
def test_embeddings_dropped(fixtures_dir: Path, monkeypatch):
    _set_key(monkeypatch)
    body = (fixtures_dir / "together_models.json").read_text()
    responses.add(responses.GET, TOGETHER_URL, body=body, status=200)
    evs = TogetherSource().fetch()
    ids = {e.model_id for e in evs}
    # m2-bert is type=embedding — must be dropped.
    assert not any("m2-bert" in i for i in ids)


def test_no_api_key_returns_empty(monkeypatch):
    """Without TOGETHER_API_KEY, the source yields nothing (no HTTP call)."""
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    assert TogetherSource().fetch() == []


@responses.activate
def test_all_evidence_high_confidence(fixtures_dir: Path, monkeypatch):
    _set_key(monkeypatch)
    body = (fixtures_dir / "together_models.json").read_text()
    responses.add(responses.GET, TOGETHER_URL, body=body, status=200)
    evs = TogetherSource().fetch()
    assert evs
    for e in evs:
        assert e.confidence == "high"
        assert e.source == "together"


@responses.activate
def test_auth_header_present(fixtures_dir: Path, monkeypatch):
    """Verify the source sends the Bearer auth header."""
    _set_key(monkeypatch)
    body = (fixtures_dir / "together_models.json").read_text()
    rsp = responses.add(responses.GET, TOGETHER_URL, body=body, status=200)
    if hasattr(os, "environ"):  # always true; appeases ruff F401 if needed
        pass
    TogetherSource().fetch()
    assert rsp.call_count == 1
    auth = responses.calls[0].request.headers.get("Authorization", "")
    assert auth.startswith("Bearer ")
