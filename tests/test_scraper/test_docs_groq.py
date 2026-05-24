"""Groq docs scraper: extract <code>-tagged model ids and rpm/rpd/tpm/tpd."""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.docs.groq import URL, GroqDocs


@responses.activate
def test_groq_models_extracted(fixtures_dir: Path):
    html = (fixtures_dir / "groq_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = {e.model_id: e for e in GroqDocs().fetch()}
    assert "groq/llama-3.3-70b-versatile" in evs
    assert "groq/llama-3.1-8b-instant" in evs
    assert "groq/gemma2-9b-it" in evs


@responses.activate
def test_groq_limits_extracted(fixtures_dir: Path):
    html = (fixtures_dir / "groq_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = {e.model_id: e for e in GroqDocs().fetch()}
    lim = evs["groq/llama-3.3-70b-versatile"].limits
    assert lim["requests_per_minute"] == 30
    assert lim["requests_per_day"] == 1000
    assert lim["tokens_per_minute"] == 6000
    assert lim["tokens_per_day"] == 500000
