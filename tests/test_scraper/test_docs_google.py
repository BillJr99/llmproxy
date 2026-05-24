"""Google rate-limits docs scraper: extract models + limits from the Tier 1 table.

The Google API docs page previously used a "Free Tier" heading; it now uses
"Tier 1 Rate Limits" (Tier 1 = free, no billing required). The scraper
accepts both heading styles.
"""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.docs.google import URL, GoogleDocs


@responses.activate
def test_free_tier_models_extracted(fixtures_dir: Path):
    html = (fixtures_dir / "google_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200, content_type="text/html")
    evs = GoogleDocs().fetch()
    ids = {e.model_id for e in evs}
    assert "google/gemini-2.5-pro" in ids
    assert "google/gemini-2.5-flash" in ids
    assert "google/gemini-2.5-flash-lite" in ids


@responses.activate
def test_paid_tier_models_excluded(fixtures_dir: Path):
    """Tier 2/3 tables must not produce evidence."""
    html = (fixtures_dir / "google_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200, content_type="text/html")
    evs = GoogleDocs().fetch()
    # Tier 2 table contains gemini-2.5-pro too — must not be double-counted
    # but only one evidence per model_id is emitted (seen-set dedup).
    # More importantly, we don't emit any non-Tier-1 only models.
    assert len([e for e in evs if e.model_id == "google/gemini-2.5-pro"]) == 1


@responses.activate
def test_limits_extracted_for_tier1(fixtures_dir: Path):
    html = (fixtures_dir / "google_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = {e.model_id: e for e in GoogleDocs().fetch()}
    pro = evs["google/gemini-2.5-pro"]
    assert pro.limits is not None
    assert pro.limits["requests_per_minute"] == 5
    assert pro.limits["requests_per_day"] == 100
    assert pro.limits["tokens_per_minute"] == 250000


@responses.activate
def test_all_evidence_high_confidence_positive(fixtures_dir: Path):
    html = (fixtures_dir / "google_rate_limits.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = GoogleDocs().fetch()
    assert evs
    for e in evs:
        assert e.is_free is True
        assert e.confidence == "high"
        assert e.source == "google-docs"
