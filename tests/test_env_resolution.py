"""Unit tests for runtime ${VAR} env-reference resolution in config.py.

These cover the resolver and the provider accessors that the server uses at
every upstream-request site, so secrets/endpoints can live in the environment
rather than literally in config.json.
"""

from __future__ import annotations

import pytest

from llmproxy.config import (
    account_bound_cfg,
    provider_account_strategy,
    provider_accounts,
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


# ---------------------------------------------------------------------------
# Multiple accounts per provider
# ---------------------------------------------------------------------------

def test_single_api_key_is_one_anonymous_account():
    # The legacy single-key shape yields exactly one account whose id is None,
    # so downstream usage keys stay byte-identical to provider/model.
    accts = provider_accounts({"api_key": "sk-single"})
    assert len(accts) == 1
    assert accts[0].id is None
    assert accts[0].key == "sk-single"


def test_keyless_provider_is_one_empty_account():
    accts = provider_accounts({"base_url": "http://localhost:11434/v1"})
    assert len(accts) == 1
    assert accts[0].id is None
    assert accts[0].key == ""


def test_api_keys_list_becomes_indexed_accounts():
    accts = provider_accounts({"api_keys": ["sk-1", "sk-2", "sk-3"]})
    assert [a.id for a in accts] == ["k0", "k1", "k2"]
    assert [a.key for a in accts] == ["sk-1", "sk-2", "sk-3"]


def test_accounts_resolve_env_refs_per_entry(monkeypatch):
    monkeypatch.setenv("KEY_A", "resolved-a")
    accts = provider_accounts(
        {"accounts": [{"key": "${KEY_A}", "label": "a"}, {"key": "lit-b", "label": "b"}]}
    )
    assert accts[0].key == "resolved-a"
    assert accts[0].key_raw == "${KEY_A}"  # raw form preserved for masking
    assert accts[1].key == "lit-b"


def test_priority_strategy_orders_lowest_priority_first():
    cfg = {
        "account_strategy": "priority",
        "accounts": [
            {"key": "sk-low", "label": "b", "priority": 5},
            {"key": "sk-high", "label": "a", "priority": 1},
        ],
    }
    accts = provider_accounts(cfg)
    assert [a.label for a in accts] == ["a", "b"]
    assert provider_account_strategy(cfg) == "priority"


def test_round_robin_is_default_and_preserves_declared_order():
    cfg = {"accounts": [{"key": "x", "label": "first"}, {"key": "y", "label": "second"}]}
    assert provider_account_strategy(cfg) == "round_robin"
    assert [a.label for a in provider_accounts(cfg)] == ["first", "second"]


def test_duplicate_labels_get_unique_ids():
    accts = provider_accounts(
        {"accounts": [{"key": "a", "label": "dup"}, {"key": "b", "label": "dup"}]}
    )
    assert len({a.id for a in accts}) == 2


def test_account_bound_cfg_flows_through_provider_api_key(monkeypatch):
    monkeypatch.setenv("KEY_A", "env-resolved")
    cfg = {"base_url": "http://x", "accounts": [{"key": "${KEY_A}", "label": "a"},
                                                {"key": "lit", "label": "b"}]}
    accts = provider_accounts(cfg)
    bound = account_bound_cfg(cfg, accts[0])
    # The chosen account's key reaches the existing accessor with no signature
    # change, and the account id is recoverable off the runtime copy.
    assert provider_api_key(bound) == "env-resolved"
    assert bound["_account_id"] == "a"
    assert bound["base_url"] == "http://x"  # rest of the cfg carried through


def test_single_account_label_stays_anonymous_id_none():
    # Even an explicit single-entry accounts list collapses to id None.
    accts = provider_accounts({"accounts": [{"key": "solo", "label": "only"}]})
    assert len(accts) == 1
    assert accts[0].id is None
