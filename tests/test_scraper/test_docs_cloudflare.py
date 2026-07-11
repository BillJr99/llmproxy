"""Cloudflare Workers AI catalog scraper: @cf/<vendor>/<model> IDs."""

from __future__ import annotations

from pathlib import Path

import pytest
import responses

pytest.importorskip("bs4")  # docs scrapers parse HTML with BeautifulSoup

from scripts.sources.docs.cloudflare import URL, CloudflareWorkersDocs


@responses.activate
def test_models_extracted(fixtures_dir: Path):
    html = (fixtures_dir / "cloudflare_workers_ai.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200, content_type="text/html")
    ids = {e.model_id for e in CloudflareWorkersDocs().fetch()}
    assert "cloudflare-workers/@cf/meta/llama-3.1-70b-instruct" in ids
    assert "cloudflare-workers/@cf/meta/llama-3.1-8b-instruct" in ids
    assert "cloudflare-workers/@cf/mistral/mistral-7b-instruct-v0.1" in ids
    assert "cloudflare-workers/@cf/qwen/qwen1.5-14b-chat-awq" in ids
    # @hf/ prefixed model also picked up.
    assert "cloudflare-workers/@hf/thebloke/deepseek-coder-6.7b-instruct-awq" in ids


@responses.activate
def test_high_confidence_is_free(fixtures_dir: Path):
    html = (fixtures_dir / "cloudflare_workers_ai.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = CloudflareWorkersDocs().fetch()
    assert evs
    for e in evs:
        assert e.is_free is True
        assert e.confidence == "high"
        assert e.source == "cloudflare-workers-docs"
