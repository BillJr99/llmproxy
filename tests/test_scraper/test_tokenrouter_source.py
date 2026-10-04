"""TokenRouter source: new-api ratio→dollar conversion, filtering, free detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests
import responses

from scripts.sources.tokenrouter import TOKENROUTER_PRICING_URL, TokenRouterSource


def _serve(fixtures_dir: Path, mutate=None) -> None:
    body = (fixtures_dir / "tokenrouter_pricing.json").read_text()
    if mutate is not None:
        data = json.loads(body)
        mutate(data)
        body = json.dumps(data)
    responses.add(responses.GET, TOKENROUTER_PRICING_URL, body=body, status=200,
                  content_type="application/json")


@pytest.fixture(autouse=True)
def _no_believed_free(monkeypatch):
    monkeypatch.setattr(
        "scripts.sources.tokenrouter.load_data",
        lambda: {"providers": {"tokenrouter": {"believed_free": []}}},
    )


@responses.activate
def test_ratio_converted_to_per_token_dollars(fixtures_dir: Path):
    """model_ratio 1.5 is $3/1M in; completion_ratio 5 makes $15/1M out.

    The same payload's tiered_pricing quotes this model at ratio 1.5 = $3/1M,
    which is what pins the ×2 convention.
    """
    _serve(fixtures_dir)
    by_id = {e.model_id: e for e in TokenRouterSource().fetch()}
    assert by_id["tokenrouter/anthropic/claude-sonnet-4"].pricing == {
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
    }
    # model_ratio 0.15, completion_ratio 4 → $0.30 / $1.20 per 1M.
    assert by_id["tokenrouter/MiniMax-M3"].pricing == {
        "input_cost_per_token": 3e-07,
        "output_cost_per_token": 1.2e-06,
    }


@responses.activate
def test_group_ratio_scales_prices(fixtures_dir: Path):
    _serve(fixtures_dir, lambda d: d["group_ratio"].update(default=0.5))
    by_id = {e.model_id: e for e in TokenRouterSource().fetch()}
    assert by_id["tokenrouter/anthropic/claude-sonnet-4"].pricing == {
        "input_cost_per_token": 1.5e-06,
        "output_cost_per_token": 7.5e-06,
    }


@responses.activate
def test_unusable_group_ratio_yields_no_pricing_opinion(fixtures_dir: Path):
    """Better to say nothing than to publish a price off by an unknown factor."""
    _serve(fixtures_dir, lambda d: d.pop("group_ratio"))
    evs = TokenRouterSource().fetch()
    assert evs
    assert all(e.pricing is None for e in evs)


@responses.activate
def test_non_text_and_per_call_rows_skipped(fixtures_dir: Path):
    _serve(fixtures_dir)
    ids = {e.model_id for e in TokenRouterSource().fetch()}
    assert "tokenrouter/openai/text-embedding-3-small" not in ids  # embeddings
    assert "tokenrouter/wan3.0-video" not in ids                   # quota_type 1
    assert "tokenrouter/openai/gpt-audio" not in ids               # audio-chat only


@responses.activate
def test_free_requires_zero_price_on_openai_endpoint(fixtures_dir: Path):
    _serve(fixtures_dir)
    by_id = {e.model_id: e for e in TokenRouterSource().fetch()}
    free = by_id["tokenrouter/stealth/ox-alpha"]
    assert free.is_free is True
    assert free.pricing is None
    other = by_id["tokenrouter/stealth/zero-anthropic-only"]
    assert other.is_free is False
    assert "not on the OpenAI endpoint" in other.notes
    assert by_id["tokenrouter/anthropic/claude-sonnet-4"].is_free is False


@responses.activate
def test_ids_are_prefixed_with_provider_key(fixtures_dir: Path):
    _serve(fixtures_dir)
    evs = TokenRouterSource().fetch()
    assert evs
    assert all(e.model_id.startswith("tokenrouter/") for e in evs)
    assert all(e.confidence == "high" and e.source == "tokenrouter" for e in evs)


@responses.activate
def test_stale_believed_free_ids_get_negatives(fixtures_dir: Path, monkeypatch):
    _serve(fixtures_dir)
    monkeypatch.setattr(
        "scripts.sources.tokenrouter.load_data",
        lambda: {"providers": {"tokenrouter": {"believed_free": [
            "tokenrouter/stealth/gone-alpha",
            "tokenrouter/stealth/ox-alpha",   # still free — must NOT be negated
        ]}}},
    )
    evs = TokenRouterSource().fetch()
    gone = [e for e in evs if e.model_id == "tokenrouter/stealth/gone-alpha"]
    assert len(gone) == 1 and gone[0].is_free is False
    assert all(e.is_free for e in evs if e.model_id == "tokenrouter/stealth/ox-alpha")


@responses.activate
def test_needs_no_api_key(fixtures_dir: Path):
    _serve(fixtures_dir)
    assert TokenRouterSource().fetch()
    assert "Authorization" not in responses.calls[0].request.headers


@responses.activate
def test_http_error_propagates():
    responses.add(responses.GET, TOKENROUTER_PRICING_URL, status=503)
    with pytest.raises(requests.HTTPError):
        TokenRouterSource().fetch()


@responses.activate
def test_unexpected_shape_raises():
    """An HTML page or a reshaped payload is a failure, not an empty catalog."""
    responses.add(responses.GET, TOKENROUTER_PRICING_URL, json={"success": False},
                  status=200)
    with pytest.raises(ValueError):
        TokenRouterSource().fetch()
