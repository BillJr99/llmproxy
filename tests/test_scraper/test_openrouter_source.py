"""OpenRouter source: detect $0-priced models, ignore paid ones."""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.openrouter import (
    OPENROUTER_URL,
    OpenRouterSource,
    _capabilities,
    _pricing,
)


@responses.activate
def test_free_models_detected(fixtures_dir: Path):
    body = (fixtures_dir / "openrouter_models.json").read_text()
    responses.add(responses.GET, OPENROUTER_URL, body=body, status=200,
                  content_type="application/json")
    evs = OpenRouterSource().fetch()
    by_id = {e.model_id: e for e in evs}
    assert by_id["openrouter/meta-llama/llama-3.2-3b-instruct:free"].is_free is True
    assert by_id["openrouter/deepseek/deepseek-v3.1:free"].is_free is True
    assert by_id["openrouter/qwen/qwen3-coder:free"].is_free is True
    assert by_id["openrouter/anthropic/claude-3.5-sonnet"].is_free is False
    for ev in evs:
        assert ev.confidence == "high"
        assert ev.source == "openrouter"


@responses.activate
def test_paid_models_carry_pricing_free_do_not(fixtures_dir: Path):
    body = (fixtures_dir / "openrouter_models.json").read_text()
    responses.add(responses.GET, OPENROUTER_URL, body=body, status=200,
                  content_type="application/json")
    by_id = {e.model_id: e for e in OpenRouterSource().fetch()}
    # Paid model -> per-token pricing (OpenRouter prices are already per token).
    assert by_id["openrouter/anthropic/claude-3.5-sonnet"].pricing == {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    }
    # Free models carry no pricing opinion (captured by believed_free).
    assert by_id["openrouter/qwen/qwen3-coder:free"].pricing is None


def test_pricing_helper():
    assert _pricing(0.0, 0.0) is None                    # free -> no opinion
    assert _pricing(float("inf"), float("inf")) is None  # unknown -> no opinion
    assert _pricing(1e-6, 2e-6) == {
        "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6,
    }
    # Half-known: the missing side defaults to 0.0 but the record is emitted.
    assert _pricing(1e-6, float("inf")) == {
        "input_cost_per_token": 1e-6, "output_cost_per_token": 0.0,
    }


def test_capabilities_mapping():
    model = {
        "supported_parameters": ["tools", "reasoning", "structured_outputs"],
        "architecture": {"input_modalities": ["text", "image"]},
    }
    assert _capabilities(model) == ["tools", "vision", "reasoning", "json"]
    # text-only, no special params
    assert _capabilities({"supported_parameters": [], "architecture": {"input_modalities": ["text"]}}) == []
    # response_format also implies json; defensive against missing/bad fields
    assert _capabilities({"supported_parameters": ["response_format"]}) == ["json"]
    assert _capabilities({}) == []
    assert _capabilities({"supported_parameters": None, "architecture": None}) == []


@responses.activate
def test_network_failure_raises():
    responses.add(responses.GET, OPENROUTER_URL, status=500)
    import requests
    try:
        OpenRouterSource().fetch()
    except requests.HTTPError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected HTTPError")
