"""Unit tests for runtime ${VAR} env-reference resolution in config.py.

These cover the resolver and the provider accessors that the server uses at
every upstream-request site, so secrets/endpoints can live in the environment
rather than literally in config.json.
"""

from __future__ import annotations

import pytest

from llmproxy.config import (
    provider_api_key,
    provider_base_url,
    resolve_env_refs,
    value_has_env_ref,
)


def test_resolve_substitutes_from_environ(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-secret-123")
    assert resolve_env_refs("${MY_KEY}") == "sk-secret-123"


def test_resolve_substitutes_inside_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5")
    assert resolve_env_refs("http://${OLLAMA_HOST}:11434/v1") == "http://10.0.0.5:11434/v1"


def test_resolve_unset_var_becomes_empty(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    assert resolve_env_refs("${NOPE}") == ""
    assert resolve_env_refs("a-${NOPE}-b") == "a--b"


def test_resolve_multiple_refs(monkeypatch):
    monkeypatch.setenv("A", "1")
    monkeypatch.setenv("B", "2")
    assert resolve_env_refs("${A}/${B}") == "1/2"


def test_resolve_passthrough_for_plain_and_non_str():
    assert resolve_env_refs("plain") == "plain"
    assert resolve_env_refs("") == ""
    assert resolve_env_refs(None) is None
    assert resolve_env_refs(123) == 123


def test_provider_base_url_resolves_and_strips_slash(monkeypatch):
    monkeypatch.setenv("HOST", "example.com")
    cfg = {"base_url": "https://${HOST}/v1/"}
    assert provider_base_url(cfg) == "https://example.com/v1"


def test_provider_api_key_resolves(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-xyz")
    assert provider_api_key({"api_key": "${OPENAI_API_KEY}"}) == "sk-live-xyz"


def test_provider_accessors_handle_missing_fields():
    assert provider_base_url({}) == ""
    assert provider_api_key({}) == ""


def test_provider_api_key_literal_passthrough():
    assert provider_api_key({"api_key": "literal-key"}) == "literal-key"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("${X}", True),
        ("pre-${X}-post", True),
        ("literal", False),
        ("", False),
        (None, False),
        (42, False),
    ],
)
def test_value_has_env_ref(value, expected):
    assert value_has_env_ref(value) is expected
