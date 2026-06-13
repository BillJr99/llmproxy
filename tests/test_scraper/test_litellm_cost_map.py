"""LiteLLM cost-map source: zero-priced models → high-breadth, medium-confidence evidence."""

from __future__ import annotations

import json
from pathlib import Path

import responses

from scripts.sources.litellm_cost_map import (
    LITELLM_COST_MAP_URL,
    LiteLLMCostMapSource,
    build_pricing_map,
)


@responses.activate
def test_zero_priced_models_emitted(fixtures_dir: Path):
    body = (fixtures_dir / "litellm_cost_map.json").read_text()
    responses.add(responses.GET, LITELLM_COST_MAP_URL, body=body, status=200,
                  content_type="application/json")
    evs = LiteLLMCostMapSource().fetch()
    ids = {e.model_id for e in evs}
    # Free chat models (zero cost both ways) become evidence.
    assert "groq/llama-3.1-8b-instant" in ids
    assert "groq/llama-3.3-70b-versatile" in ids
    assert "cerebras/llama3.1-8b" in ids
    # together_ai / fireworks_ai map to together / fireworks provider keys.
    assert "together/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" in ids
    assert "fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct" in ids
    # openrouter zero-priced model surfaced.
    assert "openrouter/google/gemini-2.0-flash-exp:free" in ids


@responses.activate
def test_paid_models_dropped(fixtures_dir: Path):
    body = (fixtures_dir / "litellm_cost_map.json").read_text()
    responses.add(responses.GET, LITELLM_COST_MAP_URL, body=body, status=200)
    evs = LiteLLMCostMapSource().fetch()
    ids = {e.model_id for e in evs}
    # OpenAI sample entry has non-zero cost.
    assert not any(i.endswith("sample_text_completion_openai") for i in ids)
    # Gemini 1.5 flash has non-zero cost — must be excluded.
    assert "google/gemini-1.5-flash-002" not in ids


@responses.activate
def test_non_chat_modes_dropped(fixtures_dir: Path):
    """Audio / embeddings shouldn't be emitted as chat models."""
    body = (fixtures_dir / "litellm_cost_map.json").read_text()
    responses.add(responses.GET, LITELLM_COST_MAP_URL, body=body, status=200)
    evs = LiteLLMCostMapSource().fetch()
    ids = {e.model_id for e in evs}
    assert "groq/whisper-large-v3" not in ids


@responses.activate
def test_unmapped_providers_dropped(fixtures_dir: Path):
    """Unknown litellm_provider values are silently dropped."""
    body = (fixtures_dir / "litellm_cost_map.json").read_text()
    responses.add(responses.GET, LITELLM_COST_MAP_URL, body=body, status=200)
    evs = LiteLLMCostMapSource().fetch()
    ids = {e.model_id for e in evs}
    assert not any("foo-model" in i for i in ids)


@responses.activate
def test_all_evidence_medium_confidence(fixtures_dir: Path):
    body = (fixtures_dir / "litellm_cost_map.json").read_text()
    responses.add(responses.GET, LITELLM_COST_MAP_URL, body=body, status=200)
    evs = LiteLLMCostMapSource().fetch()
    assert evs
    for e in evs:
        assert e.is_free is True
        assert e.confidence == "medium"
        assert e.source == "litellm_cost_map"


def test_build_pricing_map_keeps_paid_chat_models(fixtures_dir: Path):
    data = json.loads((fixtures_dir / "litellm_cost_map.json").read_text())
    pricing = build_pricing_map(data)
    # Paid, mapped, chat model is priced.
    assert "google/gemini-1.5-flash-002" in pricing
    assert pricing["google/gemini-1.5-flash-002"] == {
        "input_cost_per_token": 0.000000075,
        "output_cost_per_token": 0.0000003,
    }
    # Zero-priced models are NOT in the pricing snapshot (covered by believed_free).
    assert "groq/llama-3.1-8b-instant" not in pricing
    # Unmapped provider (openai) is dropped.
    assert not any(k.startswith("openai/") for k in pricing)
