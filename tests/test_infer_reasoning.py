"""Coverage of llmproxy.providers.infer_reasoning_level boundaries."""

from __future__ import annotations

import pytest

from llmproxy.providers import infer_reasoning_level


@pytest.mark.parametrize("name", [
    "deepseek-r1-distill-qwen-7b",
    "qwq-32b-preview",
    "magistral-medium-2509",
    "o1-mini",
    "o3-mini-2025-01-31",
    "reasoning-pro",
])
def test_deep_keyword_overrides_size(name):
    assert infer_reasoning_level(name) == "deep"


@pytest.mark.parametrize("name,expected", [
    ("llama-3.1-405b", "deep"),
    ("qwen3-235b", "deep"),
    ("llama-3.1-70b-instruct", "standard"),
    # mixtral-8x7b regex matches "7b" first, so size-heuristic → exploratory.
    # Manual model_reasoning overrides this for cases like groq/mixtral-8x7b-32768.
    ("mixtral-8x7b", "exploratory"),
    ("llama-3.1-8b-instant", "exploratory"),
    ("gemma2-9b-it", "exploratory"),
    ("phi-3.5-mini", "exploratory"),
])
def test_size_based_inference(name, expected):
    assert infer_reasoning_level(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("mistral-large-latest", "standard"),
    ("command-r-plus", "exploratory"),  # no size, no deep kw, no standard kw → exploratory
])
def test_keyword_fallback(name, expected):
    assert infer_reasoning_level(name) == expected


def test_unknown_returns_exploratory():
    assert infer_reasoning_level("totally-unknown-tiny-model") == "exploratory"
