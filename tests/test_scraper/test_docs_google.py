"""Google rate-limits docs scraper.

As of mid-2025 per-model RPM/RPD limits are no longer in the page HTML.
The scraper now:
  1. Confirms a "Free" row exists in the usage-tiers table.
  2. Collects gemini-* model ID strings via regex over the raw HTML.
  3. Returns no limits (None) since the page no longer publishes them.
"""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.docs.google import URL, GoogleDocs


@responses.activate
def test_free_tier_models_extracted(fixtures_dir: Path):
    """Model IDs embedded in page text are surfaced as free evidence."""
    html = (fixtures_dir / "google_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200, content_type="text/html")
    evs = GoogleDocs().fetch()
    ids = {e.model_id for e in evs}
    assert "google/gemini-2.5-pro" in ids
    assert "google/gemini-2.5-flash" in ids
    assert "google/gemini-2.5-flash-lite" in ids
    assert "google/gemini-2.0-flash" in ids


@responses.activate
def test_no_free_tier_row_yields_empty(fixtures_dir: Path):
    """If the usage-tiers table has no 'Free' row, emit nothing."""
    html = """<!doctype html><html><body>
    <table><tr><td>Tier 1</td><td>paid</td></tr></table>
    <p>gemini-2.5-pro gemini-2.5-flash</p>
    </body></html>"""
    responses.add(responses.GET, URL, body=html, status=200)
    evs = GoogleDocs().fetch()
    assert evs == []


@responses.activate
def test_all_evidence_high_confidence_positive(fixtures_dir: Path):
    """All evidence must be is_free=True, confidence=high, source=google-docs."""
    html = (fixtures_dir / "google_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = GoogleDocs().fetch()
    assert evs
    for e in evs:
        assert e.is_free is True
        assert e.confidence == "high"
        assert e.source == "google-docs"


@responses.activate
def test_limits_are_none(fixtures_dir: Path):
    """Limits field should be None — page no longer publishes per-model limits."""
    html = (fixtures_dir / "google_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = {e.model_id: e for e in GoogleDocs().fetch()}
    assert evs["google/gemini-2.5-pro"].limits is None
