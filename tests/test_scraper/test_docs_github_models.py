"""GitHub Models docs scraper: vendor/<model> IDs from the prototyping page."""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.docs.github_models import URL, GitHubModelsDocs


@responses.activate
def test_models_extracted(fixtures_dir: Path):
    html = (fixtures_dir / "github_models_docs.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200, content_type="text/html")
    evs = {e.model_id for e in GitHubModelsDocs().fetch()}
    assert "github/openai/gpt-4o" in evs
    assert "github/openai/gpt-4o-mini" in evs
    assert "github/meta/Meta-Llama-3.1-70B-Instruct" in evs
    assert "github/mistral-ai/Mistral-Nemo" in evs
    assert "github/microsoft/Phi-3.5-mini-instruct" in evs


@responses.activate
def test_high_confidence_is_free(fixtures_dir: Path):
    html = (fixtures_dir / "github_models_docs.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = GitHubModelsDocs().fetch()
    assert evs
    for e in evs:
        assert e.is_free is True
        assert e.confidence == "high"
        assert e.source == "github-models-docs"


@responses.activate
def test_unrelated_page_yields_empty():
    """A page that doesn't look like the rate-limits doc emits nothing."""
    html = "<html><body><p>This is a totally unrelated page.</p></body></html>"
    responses.add(responses.GET, URL, body=html, status=200)
    assert GitHubModelsDocs().fetch() == []
