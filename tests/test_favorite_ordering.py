"""Unit tests for _apply_favorite_free_ordering() in server.py."""

from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture(autouse=True)
def _reload_server(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"providers": {}, "believed_free": []}))
    monkeypatch.setenv("LLMPROXY_CONFIG", str(cfg))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)


def _fn():
    from llmproxy import server as server_mod
    return server_mod._apply_favorite_free_ordering


def _c(provider, model):
    """Build a minimal (provider_name, provider_cfg, upstream_model) tuple."""
    return (provider, {}, model)


def test_empty_favorites_returns_unchanged():
    candidates = [_c("openai", "gpt-4o"), _c("google", "gemini-flash")]
    result = _fn()(candidates, {"favorite_free_models": []})
    assert result == candidates


def test_no_favorites_key_returns_unchanged():
    candidates = [_c("openai", "gpt-4o"), _c("google", "gemini-flash")]
    result = _fn()(candidates, {})
    assert result == candidates


def test_favorite_promoted_to_front():
    a = _c("openai", "gpt-4o")
    b = _c("google", "gemini-flash")
    candidates = [a, b]
    result = _fn()(candidates, {"favorite_free_models": ["google/gemini-flash"]})
    assert result[0] == b
    assert result[1] == a


def test_multiple_favorites_ranked_order():
    a = _c("openai", "gpt-4o-mini")
    b = _c("google", "gemini-flash")
    c = _c("anthropic", "haiku")
    candidates = [a, b, c]
    result = _fn()(candidates, {"favorite_free_models": ["anthropic/haiku", "google/gemini-flash"]})
    assert result[0] == c
    assert result[1] == b
    assert result[2] == a


def test_favorite_not_in_pool_silently_skipped():
    a = _c("openai", "gpt-4o")
    candidates = [a]
    # "paid-model" is not in the pool
    result = _fn()(candidates, {"favorite_free_models": ["openai/paid-model", "openai/gpt-4o"]})
    assert result == [a]


def test_bare_id_match():
    a = _c("openai", "gpt-4o-mini")
    b = _c("google", "gemini-flash")
    candidates = [a, b]
    # bare id (no provider prefix) matches by upstream_model
    result = _fn()(candidates, {"favorite_free_models": ["gemini-flash"]})
    assert result[0] == b
    assert result[1] == a


def test_qualified_id_match():
    a = _c("openai", "gpt-4o-mini")
    b = _c("google", "gemini-flash")
    candidates = [a, b]
    result = _fn()(candidates, {"favorite_free_models": ["openai/gpt-4o-mini"]})
    assert result[0] == a
    assert result[1] == b


def test_case_insensitive_match():
    a = _c("Google", "Gemini-Flash")
    candidates = [a]
    result = _fn()(candidates, {"favorite_free_models": ["google/gemini-flash"]})
    assert result == [a]


def test_non_favorite_order_preserved():
    a = _c("a", "model-a")
    b = _c("b", "model-b")
    c = _c("c", "model-c")
    d = _c("d", "model-d")
    candidates = [a, b, c, d]
    # Only promote c; a, b, d should remain in their original relative order
    result = _fn()(candidates, {"favorite_free_models": ["c/model-c"]})
    assert result == [c, a, b, d]


def test_empty_candidates_returns_empty():
    result = _fn()([], {"favorite_free_models": ["openai/gpt-4o"]})
    assert result == []


def test_variant_suffix_stripped_bare():
    # "gemini-flash" matches upstream_model "gemini-flash:free"
    a = _c("google", "gemini-flash:free")
    b = _c("openai", "gpt-4o")
    candidates = [b, a]
    result = _fn()(candidates, {"favorite_free_models": ["gemini-flash"]})
    assert result[0] == a


def test_variant_suffix_stripped_qualified():
    # "google/gemini-flash" matches (provider="google", model="gemini-flash:free")
    a = _c("google", "gemini-flash:free")
    b = _c("openai", "gpt-4o")
    candidates = [b, a]
    result = _fn()(candidates, {"favorite_free_models": ["google/gemini-flash"]})
    assert result[0] == a


def test_exact_with_suffix_still_matches():
    # Specifying "google/gemini-flash:free" also works (exact match)
    a = _c("google", "gemini-flash:free")
    b = _c("openai", "gpt-4o")
    candidates = [b, a]
    result = _fn()(candidates, {"favorite_free_models": ["google/gemini-flash:free"]})
    assert result[0] == a
