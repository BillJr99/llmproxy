"""xKiro source: free detection, per-1M price conversion, and stale removals."""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.xkiro import XKIRO_URL, XkiroSource


def _serve(fixtures_dir: Path) -> None:
    body = (fixtures_dir / "xkiro_models.json").read_text()
    responses.add(responses.GET, XKIRO_URL, body=body, status=200,
                  content_type="application/json")


@responses.activate
def test_free_models_emitted(fixtures_dir: Path):
    _serve(fixtures_dir)
    by_id = {e.model_id: e for e in XkiroSource().fetch()}
    assert by_id["xkiro/openai/gpt-5.3-codex-spark"].is_free is True
    assert by_id["xkiro/mistralai/ministral-8b"].is_free is True
    assert by_id["xkiro/sensenova/sensenova-6.8-flash-lite"].is_free is True
    assert by_id["xkiro/openai/gpt-5.6-terra"].is_free is False
    assert by_id["xkiro/xiaomi/mimo-v2.5"].is_free is False


@responses.activate
def test_ids_are_prefixed_with_provider_key(fixtures_dir: Path):
    """tests/test_providers_data.py asserts this prefix on believed_free entries."""
    _serve(fixtures_dir)
    assert all(e.model_id.startswith("xkiro/") for e in XkiroSource().fetch())


@responses.activate
def test_paid_models_carry_per_token_pricing(fixtures_dir: Path):
    """xKiro quotes per 1M tokens; providers.json stores per token."""
    _serve(fixtures_dir)
    by_id = {e.model_id: e for e in XkiroSource().fetch()}
    # 0.091 / 0.182 per 1M tokens.
    assert by_id["xkiro/xiaomi/mimo-v2.5"].pricing == {
        "input_cost_per_token": 9.1e-08,
        "output_cost_per_token": 1.82e-07,
    }
    assert by_id["xkiro/openai/gpt-5.6-terra"].pricing == {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 6e-06,
    }
    # Free models carry no pricing opinion.
    assert by_id["xkiro/mistralai/ministral-8b"].pricing is None


@responses.activate
def test_unexpected_price_unit_yields_no_pricing_opinion(fixtures_dir: Path):
    """Better to say nothing than to be wrong by a factor of a million."""
    _serve(fixtures_dir)
    by_id = {e.model_id: e for e in XkiroSource().fetch()}
    assert by_id["xkiro/vendor/unknown-price-unit"].pricing is None


@responses.activate
def test_tier_and_price_must_agree(fixtures_dir: Path):
    """access_tier "free" on a model that bills is NOT published as free."""
    _serve(fixtures_dir)
    by_id = {e.model_id: e for e in XkiroSource().fetch()}
    ev = by_id["xkiro/vendor/tier-says-free-but-bills"]
    assert ev.is_free is False
    assert "disagree" in ev.notes


@responses.activate
def test_non_chat_modalities_excluded(fixtures_dir: Path):
    """An embedding model must never reach believed_free."""
    _serve(fixtures_dir)
    ids = {e.model_id for e in XkiroSource().fetch()}
    assert "xkiro/vendor/embed-1" not in ids


@responses.activate
def test_capabilities_mapped(fixtures_dir: Path):
    _serve(fixtures_dir)
    by_id = {e.model_id: e for e in XkiroSource().fetch()}
    assert by_id["xkiro/mistralai/ministral-8b"].capabilities == ["tools", "vision"]
    assert by_id["xkiro/openai/gpt-5.3-codex-spark"].capabilities == ["tools", "reasoning"]


@responses.activate
def test_stale_believed_free_ids_get_high_confidence_negatives(fixtures_dir: Path, monkeypatch):
    """aggregate() only removes by absence for source=="api", so the source must
    say "not free" explicitly about ids the catalog has dropped."""
    _serve(fixtures_dir)
    monkeypatch.setattr(
        "scripts.sources.xkiro.load_data",
        lambda: {"providers": {"xkiro": {"believed_free": [
            "xkiro/qwen/qwen3.8-max:free",
            "xkiro/mistralai/ministral-8b",   # still listed — must NOT be negated
        ]}}},
    )
    by_id = {e.model_id: e for e in XkiroSource().fetch()}
    stale = by_id["xkiro/qwen/qwen3.8-max:free"]
    assert stale.is_free is False
    assert stale.confidence == "high"
    assert "no longer lists" in stale.notes
    assert by_id["xkiro/mistralai/ministral-8b"].is_free is True


@responses.activate
def test_needs_no_api_key(fixtures_dir: Path, monkeypatch):
    """Unlike requesty/together, an absent key must not silence this source —
    that is what lets it run in the secret-less refresh workflow."""
    monkeypatch.delenv("XKIRO_API_KEY", raising=False)
    _serve(fixtures_dir)
    evs = XkiroSource().fetch()
    assert evs
    assert "Authorization" not in responses.calls[0].request.headers


@responses.activate
def test_all_evidence_high_confidence(fixtures_dir: Path):
    _serve(fixtures_dir)
    evs = XkiroSource().fetch()
    assert evs
    for e in evs:
        assert e.confidence == "high"
        assert e.source == "xkiro"


@responses.activate
def test_http_error_propagates(fixtures_dir: Path):
    """A failure must raise so the CLI records succeeded=False ("no evidence")
    rather than an empty catalog, which would read as "everything was removed"."""
    import pytest
    import requests
    responses.add(responses.GET, XKIRO_URL, status=503)
    with pytest.raises(requests.HTTPError):
        XkiroSource().fetch()
