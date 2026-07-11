"""SambaNova docs scraper: rate-limit table → models + RPM/RPD/TPM/TPD."""

from __future__ import annotations

from pathlib import Path

import pytest
import responses

pytest.importorskip("bs4")  # docs scrapers parse HTML with BeautifulSoup

from scripts.sources.docs.sambanova import URL, SambaNovaDocs


@responses.activate
def test_models_extracted(fixtures_dir: Path):
    html = (fixtures_dir / "sambanova_apis.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200, content_type="text/html")
    evs = {e.model_id: e for e in SambaNovaDocs().fetch()}
    assert "sambanova/Meta-Llama-3.3-70B-Instruct" in evs
    assert "sambanova/DeepSeek-R1" in evs
    assert "sambanova/Qwen3-32B" in evs
    assert "sambanova/Llama-3.1-Tulu-3-405B" in evs
    # The "Endpoints" table must NOT produce a chat-completions evidence row.
    assert "sambanova/chat-completions" not in evs


@responses.activate
def test_limits_extracted_with_k_suffix(fixtures_dir: Path):
    html = (fixtures_dir / "sambanova_apis.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = {e.model_id: e for e in SambaNovaDocs().fetch()}
    lim = evs["sambanova/Meta-Llama-3.3-70B-Instruct"].limits
    assert lim["requests_per_minute"] == 20
    assert lim["requests_per_day"] == 1000       # "1K"
    assert lim["tokens_per_minute"] == 30000     # "30K"
    assert lim["tokens_per_day"] == 200000       # "200K"


@responses.activate
def test_high_confidence_is_free(fixtures_dir: Path):
    html = (fixtures_dir / "sambanova_apis.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = SambaNovaDocs().fetch()
    assert evs
    for e in evs:
        assert e.is_free is True
        assert e.confidence == "high"
        assert e.source == "sambanova-docs"
