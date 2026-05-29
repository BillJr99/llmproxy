"""OpenRouter source: detect $0-priced models, ignore paid ones."""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.openrouter import OPENROUTER_URL, OpenRouterSource, _capabilities


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
