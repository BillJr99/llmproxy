"""Community source (freellmapi): markdown table → Evidence per known provider."""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.community import COMMUNITY_README_URL, CommunitySource


@responses.activate
def test_markdown_table_parsed(fixtures_dir: Path):
    body = (fixtures_dir / "freellmapi.md").read_text()
    responses.add(responses.GET, COMMUNITY_README_URL, body=body, status=200,
                  content_type="text/markdown")
    evs = CommunitySource().fetch()
    providers = {e.provider for e in evs}
    # Known providers parsed
    assert "google" in providers
    assert "groq" in providers
    assert "cerebras" in providers
    assert "mistral" in providers
    assert "openrouter" in providers
    # Unknown provider silently dropped
    assert "unknown-provider" not in providers


@responses.activate
def test_all_evidence_is_low_confidence_positive(fixtures_dir: Path):
    body = (fixtures_dir / "freellmapi.md").read_text()
    responses.add(responses.GET, COMMUNITY_README_URL, body=body, status=200)
    evs = CommunitySource().fetch()
    assert evs, "expected at least one evidence record"
    for e in evs:
        assert e.is_free is True
        assert e.confidence == "low"
        assert e.source == "community"
