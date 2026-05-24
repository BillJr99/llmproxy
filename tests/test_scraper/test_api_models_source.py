"""Provider /v1/models source: produces medium-confidence existence records."""

from __future__ import annotations

import responses

from scripts.sources.api_models import ApiModelsSource


@responses.activate
def test_only_providers_with_env_keys_are_queried(monkeypatch):
    # Only set GROQ_API_KEY; google etc. should be skipped.
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    responses.add(
        responses.GET, "https://api.groq.com/openai/v1/models",
        json={"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "gemma2-9b-it"}]},
        status=200,
    )
    src = ApiModelsSource(providers={
        "groq": "https://api.groq.com/openai/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    })
    evs = src.fetch()
    providers = {e.provider for e in evs}
    assert providers == {"groq"}
    assert {e.model_id for e in evs} == {
        "groq/llama-3.3-70b-versatile",
        "groq/gemma2-9b-it",
    }
    assert all(e.is_free is None for e in evs)
    assert all(e.confidence == "medium" for e in evs)


@responses.activate
def test_per_provider_failure_isolated(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")

    responses.add(
        responses.GET, "https://api.groq.com/openai/v1/models",
        json={"data": [{"id": "llama"}]}, status=200,
    )
    responses.add(
        responses.GET, "https://generativelanguage.googleapis.com/v1beta/openai/models",
        status=500,
    )
    src = ApiModelsSource(providers={
        "groq": "https://api.groq.com/openai/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    })
    evs = src.fetch()
    # Google failure shouldn't kill the groq result
    assert any(e.model_id == "groq/llama" for e in evs)
    assert not any(e.provider == "google" for e in evs)
