"""AIHubMix source: free only when every reported price is zero."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests
import responses

from scripts.sources.aihubmix import AIHUBMIX_URL, AihubmixSource


def _serve(fixtures_dir: Path, status: int = 200) -> None:
    body = (fixtures_dir / "aihubmix_models.json").read_text()
    responses.add(responses.GET, AIHUBMIX_URL, body=body, status=status,
                  content_type="application/json")


def _by_id(fixtures_dir: Path) -> dict:
    _serve(fixtures_dir)
    return {e.model_id: e for e in AihubmixSource().fetch()}


@responses.activate
def test_all_zero_prices_are_free(fixtures_dir: Path):
    by_id = _by_id(fixtures_dir)
    assert by_id["aihubmix/coding-glm-5.4-free"].is_free is True
    assert by_id["aihubmix/coding-glm-5.4-free"].pricing is None
    assert by_id["aihubmix/search-free"].is_free is True


@responses.activate
def test_free_named_id_with_a_price_is_paid(fixtures_dir: Path):
    """AIHubMix re-serves legacy :free ids at a real per-token price."""
    e = _by_id(fixtures_dir)["aihubmix/meta-llama/llama-3.1-8b-instruct:free"]
    assert e.is_free is False
    assert e.pricing == {"input_cost_per_token": 2e-08, "output_cost_per_token": 2e-08}


@pytest.mark.parametrize("mid", ["cache-priced-free", "tiered-free"])
@responses.activate
def test_any_price_category_above_zero_is_paid(fixtures_dir: Path, mid: str):
    """Cache prices and context tiers count, not just input/output."""
    assert _by_id(fixtures_dir)[f"aihubmix/{mid}"].is_free is False


@responses.activate
def test_paid_models_carry_per_token_pricing(fixtures_dir: Path):
    """AIHubMix quotes USD per 1M tokens; providers.json stores per token."""
    e = _by_id(fixtures_dir)["aihubmix/gpt-4o-mini"]
    assert e.is_free is False
    assert e.pricing == {"input_cost_per_token": 1.5e-07, "output_cost_per_token": 6e-07}


@responses.activate
def test_missing_pricing_gives_no_opinion(fixtures_dir: Path):
    e = _by_id(fixtures_dir)["aihubmix/unpriced-model"]
    assert e.is_free is None
    assert e.pricing is None


@responses.activate
def test_non_chat_types_excluded(fixtures_dir: Path):
    assert "aihubmix/doubao-seedream-5-5" not in _by_id(fixtures_dir)


@responses.activate
def test_capabilities_mapped(fixtures_dir: Path):
    by_id = _by_id(fixtures_dir)
    assert by_id["aihubmix/coding-glm-5.4-free"].capabilities == ["tools", "vision"]
    assert by_id["aihubmix/glm-4.7"].capabilities == ["tools"]
    assert by_id["aihubmix/unpriced-model"].capabilities is None


@responses.activate
def test_ids_are_prefixed_and_high_confidence(fixtures_dir: Path):
    evidence = _by_id(fixtures_dir).values()
    assert all(e.model_id.startswith("aihubmix/") for e in evidence)
    assert all(e.confidence == "high" for e in evidence)


@responses.activate
def test_stale_believed_free_ids_get_negatives(fixtures_dir: Path, monkeypatch):
    import scripts.sources.aihubmix as mod
    monkeypatch.setattr(mod, "load_data", lambda: {"providers": {"aihubmix": {
        "believed_free": ["aihubmix/coding-glm-5.4-free", "aihubmix/gone-free"]}}})
    by_id = _by_id(fixtures_dir)
    assert by_id["aihubmix/gone-free"].is_free is False
    assert by_id["aihubmix/coding-glm-5.4-free"].is_free is True


@responses.activate
def test_http_error_propagates(fixtures_dir: Path):
    _serve(fixtures_dir, status=503)
    with pytest.raises(requests.HTTPError):
        AihubmixSource().fetch()
