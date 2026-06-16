"""Structural invariants for llmproxy/providers.json.

These guard against bad scraper writes — any time a check fails, either the
scraper wrote nonsense or a hand-edit drifted from the schema. The intent
is to make these properties unmissable in CI.
"""

from __future__ import annotations

import pytest

from llmproxy.providers import FREE_LIMIT_KEYS, VALID_REASONING_LEVELS, load_data

SIDE = load_data()


def test_top_level_keys():
    assert "providers" in SIDE
    assert "provider_order" in SIDE


def test_provider_order_is_permutation_of_providers():
    assert set(SIDE["provider_order"]) == set(SIDE["providers"].keys())
    assert len(SIDE["provider_order"]) == len(SIDE["providers"])


@pytest.mark.parametrize("pkey", list(SIDE["providers"].keys()))
def test_every_believed_free_id_is_prefixed_by_its_provider(pkey):
    prov = SIDE["providers"][pkey]
    for mid in prov.get("believed_free", []):
        assert mid.startswith(f"{pkey}/"), (
            f"{pkey}.believed_free entry {mid!r} should start with {pkey!r}/"
        )


# OpenRouter is special: free models are detected at runtime via the ":free"
# id suffix, so they never appear in believed_free even though free_limits and
# model_reasoning entries do exist for them.
_FREE_LIMITS_CAN_EXCEED_BELIEVED_FREE = frozenset({"openrouter"})


@pytest.mark.parametrize("pkey", list(SIDE["providers"].keys()))
def test_free_limits_keys_subset_of_believed_free(pkey):
    if pkey in _FREE_LIMITS_CAN_EXCEED_BELIEVED_FREE:
        return  # known exception
    prov = SIDE["providers"][pkey]
    bf = set(prov.get("believed_free", []))
    fl_keys = set(prov.get("free_limits", {}).keys())
    orphans = fl_keys - bf
    assert not orphans, f"{pkey} has free_limits entries not in believed_free: {orphans}"


def test_openrouter_free_limits_match_suffix_pattern():
    """For the openrouter exception, every free_limits/model_reasoning
    entry must use the ':free' suffix convention so runtime detection works."""
    prov = SIDE["providers"]["openrouter"]
    for mid in prov.get("free_limits", {}):
        assert mid.endswith(":free"), f"openrouter limits entry without :free suffix: {mid}"


@pytest.mark.parametrize("pkey", list(SIDE["providers"].keys()))
def test_free_limits_have_canonical_shape(pkey):
    prov = SIDE["providers"][pkey]
    for mid, lim in prov.get("free_limits", {}).items():
        assert set(lim.keys()) == set(FREE_LIMIT_KEYS), (
            f"{pkey}.{mid} free_limits keys {set(lim.keys())} != expected {set(FREE_LIMIT_KEYS)}"
        )
        for k, v in lim.items():
            assert v is None or isinstance(v, int), (
                f"{pkey}.{mid}.{k} must be int|null, got {type(v).__name__}={v!r}"
            )


@pytest.mark.parametrize("pkey", list(SIDE["providers"].keys()))
def test_reasoning_values_are_valid(pkey):
    prov = SIDE["providers"][pkey]
    for mid, level in prov.get("model_reasoning", {}).items():
        assert level in VALID_REASONING_LEVELS, (
            f"{pkey}.{mid} reasoning {level!r} not in {VALID_REASONING_LEVELS}"
        )


def test_pricing_block_shape():
    """The top-level pricing block: '<provider>/<model>' → two non-negative
    per-token costs. Guards the loadbalanced paid-tier ranking input."""
    pricing = SIDE.get("pricing", {})
    assert isinstance(pricing, dict)
    for key, val in pricing.items():
        assert isinstance(key, str) and "/" in key, f"bad pricing key {key!r}"
        assert key == key.lower(), f"pricing key not lowercased: {key!r}"
        assert set(val.keys()) == {"input_cost_per_token", "output_cost_per_token"}, (
            f"{key} pricing keys {set(val.keys())} unexpected"
        )
        for k, v in val.items():
            assert isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0, (
                f"{key}.{k} must be a non-negative number, got {v!r}"
            )


@pytest.mark.parametrize("pkey", list(SIDE["providers"].keys()))
def test_free_allowance_has_canonical_shape(pkey):
    """When a provider declares a provider-wide free_allowance it must use the
    same 4-key int|null shape as free_limits."""
    prov = SIDE["providers"][pkey]
    allowance = prov.get("free_allowance")
    if allowance is None:
        return
    assert set(allowance.keys()) == set(FREE_LIMIT_KEYS), (
        f"{pkey}.free_allowance keys {set(allowance.keys())} != {set(FREE_LIMIT_KEYS)}"
    )
    for k, v in allowance.items():
        assert v is None or (isinstance(v, int) and not isinstance(v, bool)), (
            f"{pkey}.free_allowance.{k} must be int|null, got {type(v).__name__}={v!r}"
        )


@pytest.mark.parametrize("pkey", list(SIDE["providers"].keys()))
def test_template_required_fields(pkey):
    prov = SIDE["providers"][pkey]
    assert "display" in prov, f"{pkey} missing 'display'"
    assert "base_url" in prov, f"{pkey} missing 'base_url'"
    # If account_id_required is True, the URL must contain the {account_id} placeholder.
    if prov.get("account_id_required"):
        assert "{account_id}" in prov["base_url"]
    if prov.get("gateway_id_required"):
        assert "{gateway_id}" in prov["base_url"]
