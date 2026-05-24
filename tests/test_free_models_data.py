"""Structural invariants for llmproxy/free_models.json.

These guard against bad scraper writes — any time a check fails, either the
scraper wrote nonsense or a hand-edit drifted from the schema. The intent
is to make these properties unmissable in CI.
"""

from __future__ import annotations

import pytest

from llmproxy.free_models import FREE_LIMIT_KEYS, VALID_REASONING_LEVELS, load_data

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
