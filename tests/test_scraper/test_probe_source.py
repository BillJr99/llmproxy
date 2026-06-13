"""Tests for the opt-in cost probe source (scripts/sources/probe.py)."""

from __future__ import annotations

import responses

from scripts.sources import OPT_IN_SOURCES
from scripts.sources.probe import ProbeSource


def _patch(monkeypatch, *, believed_free, providers, pricing=None):
    sidecar = {"providers": {}, "pricing": pricing or {}}
    # Distribute believed_free across provider blocks the way the sidecar does.
    for pid in believed_free:
        prov = pid.split("/", 1)[0]
        sidecar["providers"].setdefault(prov, {"believed_free": []})["believed_free"].append(pid)
    config = {"providers": providers}
    monkeypatch.setattr("scripts.sources.probe.load_data", lambda: sidecar)
    monkeypatch.setattr("scripts.sources.probe.load_config", lambda *a, **k: config)


def test_probe_is_opt_in():
    assert "probe" in OPT_IN_SOURCES


@responses.activate
def test_probe_flags_paid_model(monkeypatch):
    _patch(
        monkeypatch,
        believed_free=["groq/free-model"],
        providers={"groq": {"base_url": "http://groq.example/v1", "api_key": "k"}},
    )
    responses.add(
        responses.POST, "http://groq.example/v1/chat/completions",
        json={"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.003}},
        status=200,
    )
    evs = ProbeSource().fetch()
    assert len(evs) == 1
    assert evs[0].is_free is False
    assert evs[0].confidence == "high"
    assert evs[0].model_id == "groq/free-model"
    assert evs[0].source == "probe"


@responses.activate
def test_probe_silent_on_zero_cost(monkeypatch):
    _patch(
        monkeypatch,
        believed_free=["groq/free-model"],
        providers={"groq": {"base_url": "http://groq.example/v1", "api_key": "k"}},
    )
    responses.add(
        responses.POST, "http://groq.example/v1/chat/completions",
        json={"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        status=200,
    )
    assert ProbeSource().fetch() == []


def test_probe_skips_models_without_api_key(monkeypatch):
    _patch(
        monkeypatch,
        believed_free=["groq/free-model"],
        providers={"groq": {"base_url": "http://groq.example/v1", "api_key": ""}},
    )
    # No HTTP call should be made (responses not activated → would raise if called).
    assert ProbeSource().fetch() == []


@responses.activate
def test_probe_fail_soft_on_error(monkeypatch):
    _patch(
        monkeypatch,
        believed_free=["groq/free-model"],
        providers={"groq": {"base_url": "http://groq.example/v1", "api_key": "k"}},
    )
    responses.add(
        responses.POST, "http://groq.example/v1/chat/completions",
        status=500,
    )
    assert ProbeSource().fetch() == []


@responses.activate
def test_probe_computes_cost_from_pricing(monkeypatch):
    _patch(
        monkeypatch,
        believed_free=["groq/free-model"],
        providers={"groq": {"base_url": "http://groq.example/v1", "api_key": "k"}},
        pricing={"groq/free-model": {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000002}},
    )
    responses.add(
        responses.POST, "http://groq.example/v1/chat/completions",
        json={"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
        status=200,
    )
    evs = ProbeSource().fetch()
    assert len(evs) == 1
    assert "cost=" in evs[0].notes
