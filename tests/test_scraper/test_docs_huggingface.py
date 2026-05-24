"""Hugging Face Inference Providers docs scraper: vendor/<model> IDs."""

from __future__ import annotations

from pathlib import Path

import responses

from scripts.sources.docs.huggingface import URL, HuggingFaceDocs


@responses.activate
def test_models_extracted(fixtures_dir: Path):
    html = (fixtures_dir / "huggingface_providers.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200, content_type="text/html")
    ids = {e.model_id for e in HuggingFaceDocs().fetch()}
    assert "huggingface/meta-llama/Llama-3.3-70B-Instruct" in ids
    assert "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct" in ids
    assert "huggingface/mistralai/Mistral-Nemo-Instruct-2407" in ids
    assert "huggingface/Qwen/Qwen2.5-72B-Instruct" in ids
    assert "huggingface/deepseek-ai/DeepSeek-R1" in ids


@responses.activate
def test_medium_confidence_is_free(fixtures_dir: Path):
    """HF source is intentionally medium confidence (per-model free status
    depends on the routed backend)."""
    html = (fixtures_dir / "huggingface_providers.html").read_text()
    responses.add(responses.GET, URL, body=html, status=200)
    evs = HuggingFaceDocs().fetch()
    assert evs
    for e in evs:
        assert e.is_free is True
        assert e.confidence == "medium"
        assert e.source == "huggingface-docs"


@responses.activate
def test_unrelated_page_yields_empty():
    """A page that doesn't mention 'inference provider' yields nothing."""
    html = "<html><body><p>Some other HF docs page</p></body></html>"
    responses.add(responses.GET, URL, body=html, status=200)
    assert HuggingFaceDocs().fetch() == []
