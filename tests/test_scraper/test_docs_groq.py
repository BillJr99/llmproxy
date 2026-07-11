"""Groq docs scraper: extract plain-<td> model ids and rpm/rpd/tpm/tpd (K/M suffixes)."""

from __future__ import annotations

from pathlib import Path

import pytest
import responses

pytest.importorskip("bs4")  # docs scrapers parse HTML with BeautifulSoup

from scripts.sources.docs.groq import URL, GroqDocs


@responses.activate
def test_groq_models_extracted(fixtures_dir: Path):
    html = (fixtures_dir / "groq_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = {e.model_id: e for e in GroqDocs().fetch()}
    assert "groq/llama-3.3-70b-versatile" in evs
    assert "groq/llama-3.1-8b-instant" in evs
    assert "groq/gemma2-9b-it" in evs
    # Rate-limit header table must NOT be parsed as a model.
    assert "groq/retry-after" not in evs
    assert "groq/429 too many requests" not in evs


@responses.activate
def test_groq_limits_extracted(fixtures_dir: Path):
    """Limits use K/M shorthand in the fixture; scraper must expand them."""
    html = (fixtures_dir / "groq_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = {e.model_id: e for e in GroqDocs().fetch()}
    lim = evs["groq/llama-3.3-70b-versatile"].limits
    assert lim["requests_per_minute"] == 30
    assert lim["requests_per_day"] == 1000        # "1K"
    assert lim["tokens_per_minute"] == 12000      # "12K"
    assert lim["tokens_per_day"] == 100000        # "100K"


@responses.activate
def test_groq_k_suffix_parsed(fixtures_dir: Path):
    """14.4K → 14400, 500K → 500000."""
    html = (fixtures_dir / "groq_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = {e.model_id: e for e in GroqDocs().fetch()}
    lim = evs["groq/llama-3.1-8b-instant"].limits
    assert lim["requests_per_day"] == 14400       # "14.4K"
    assert lim["tokens_per_day"] == 500000        # "500K"


@responses.activate
def test_groq_all_evidence_high_confidence_is_free(fixtures_dir: Path):
    html = (fixtures_dir / "groq_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = GroqDocs().fetch()
    assert evs
    for e in evs:
        assert e.is_free is True
        assert e.confidence == "high"
        assert e.source == "groq-docs"
