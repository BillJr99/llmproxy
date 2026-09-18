"""Tests for three-valued capability ordering and prompt-cache passthrough.

Two independent gaps, both about metadata that was present but not honored:

* ``_model_has_capability`` returned False for a model with *no* capability
  entry exactly as it did for one whose entry omits the capability, so ordering
  buried untagged models behind weaker tagged ones. Coverage in the shipped
  sidecar is partial, and the untagged remainder includes some of the strongest
  tool-callers in the free pool, so this was not a hypothetical.
* ``cache_control`` breakpoints were deleted by the Anthropic block flattening,
  so a client could place them correctly and still be billed uncached with
  nothing in the response to say why.
"""

from __future__ import annotations

import json

import pytest

from llmproxy import server as S
from llmproxy.dialects import anthropic as A

CFG = {"base_url": "http://p/v1", "api_key": "k"}
CAPABLE = ("cap", CFG, "tagged-yes")
INCAPABLE = ("inc", CFG, "tagged-no")
UNKNOWN = ("unk", CFG, "untagged")

CAP_MAP = {
    "tagged-yes": {"tools", "json"},
    "tagged-no": {"vision"},
}


# ── three-valued capability state ───────────────────────────────────────────

def test_capability_state_distinguishes_unknown_from_incapable():
    assert S._capability_state("cap", "tagged-yes", "tools", CAP_MAP) == S._CAP_KNOWN_CAPABLE
    assert S._capability_state("inc", "tagged-no", "tools", CAP_MAP) == S._CAP_KNOWN_INCAPABLE
    assert S._capability_state("unk", "untagged", "tools", CAP_MAP) == S._CAP_UNKNOWN


def test_unknown_sorts_between_capable_and_incapable():
    """The whole point: an untagged strong model must beat a known mismatch."""
    out = S._order_by_capability([INCAPABLE, UNKNOWN, CAPABLE], {"tools"}, CAP_MAP)
    assert out == [CAPABLE, UNKNOWN, INCAPABLE]


def test_ordering_is_a_noop_without_a_needed_capability():
    cands = [INCAPABLE, UNKNOWN, CAPABLE]
    assert S._order_by_capability(cands, set(), CAP_MAP) is cands


def test_fully_untagged_pool_keeps_its_incoming_order():
    a, b, c = ("a", CFG, "x"), ("b", CFG, "y"), ("c", CFG, "z")
    assert S._order_by_capability([a, b, c], {"tools"}, {}) == [a, b, c]


def test_ordering_never_drops_a_candidate():
    cands = [INCAPABLE, UNKNOWN, CAPABLE]
    out = S._order_by_capability(cands, {"tools", "json"}, CAP_MAP)
    assert sorted(c[0] for c in out) == sorted(c[0] for c in cands)


def test_two_form_lookup_still_works():
    cap_map = {"p/m": {"tools"}}
    assert S._capability_state("p", "m", "tools", cap_map) == S._CAP_KNOWN_CAPABLE


def test_model_has_capability_stays_boolean_for_pool_membership():
    """The capability-scoped virtuals must keep meaning 'known capable'."""
    assert S._model_has_capability("cap", "tagged-yes", "tools", CAP_MAP) is True
    assert S._model_has_capability("unk", "untagged", "tools", CAP_MAP) is False


def test_tools_detector_fires_on_a_plain_tools_array():
    """An agent sending tool_choice: auto still gets capability ordering."""
    payload = {"messages": [], "tools": [{"type": "function", "function": {"name": "f"}}]}
    assert "tools" in S._needed_capabilities(payload)
    assert S._tool_use_forced(payload) is False


# ── cache_control passthrough ───────────────────────────────────────────────

MARK = {"type": "ephemeral"}


def test_uncached_request_still_flattens_to_a_plain_string():
    """The common case must be byte-identical to before."""
    out = A._anthropic_to_openai_request(
        {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hello"}]}]})
    assert out["messages"] == [{"role": "user", "content": "hello"}]


def test_cache_control_survives_on_a_user_message():
    out = A._anthropic_to_openai_request(
        {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "ctx", "cache_control": MARK},
            {"type": "text", "text": "q"}]}]})
    parts = out["messages"][0]["content"]
    assert isinstance(parts, list)
    assert parts[0]["cache_control"] == MARK
    assert "cache_control" not in parts[1]


def test_cache_control_survives_on_the_system_prompt():
    out = A._anthropic_to_openai_request(
        {"model": "m", "system": [{"type": "text", "text": "S", "cache_control": MARK}],
         "messages": []})
    assert out["messages"][0]["content"][0]["cache_control"] == MARK


def test_cache_control_survives_on_a_tool_definition():
    out = A._anthropic_to_openai_request(
        {"model": "m", "messages": [],
         "tools": [{"name": "f", "input_schema": {}, "cache_control": MARK}]})
    assert out["tools"][0]["cache_control"] == MARK


@pytest.mark.parametrize("field", ["system", "messages", "tools"])
def test_cache_control_round_trips_back_to_anthropic(field):
    original = {
        "model": "m", "max_tokens": 10,
        "system": [{"type": "text", "text": "S", "cache_control": MARK}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "ctx", "cache_control": MARK}]}],
        "tools": [{"name": "f", "description": "", "input_schema": {},
                   "cache_control": MARK}],
    }
    back = A._to_anthropic_request(A._anthropic_to_openai_request(original))
    assert "cache_control" in json.dumps(back[field])


def test_uncached_system_round_trips_as_a_plain_string():
    back = A._to_anthropic_request(A._anthropic_to_openai_request(
        {"model": "m", "system": "plain", "messages": [
            {"role": "user", "content": "hi"}]}))
    assert back["system"] == "plain"


def test_cache_usage_counters_are_surfaced():
    out = A._anthropic_response_to_openai({
        "content": [{"type": "text", "text": "x"}],
        "usage": {"input_tokens": 10, "output_tokens": 2,
                  "cache_read_input_tokens": 900, "cache_creation_input_tokens": 40}})
    assert out["usage"]["prompt_tokens_details"]["cached_tokens"] == 900
    assert out["usage"]["cache_creation_input_tokens"] == 40


def test_cache_usage_absent_when_the_provider_reports_none():
    out = A._anthropic_response_to_openai({
        "content": [], "usage": {"input_tokens": 1, "output_tokens": 1}})
    assert "prompt_tokens_details" not in out["usage"]


def test_anthropic_beta_header_is_relayed():
    """Prompt caching was gated behind this header; dropping it broke caching."""
    assert "anthropic-beta" in S._FORWARDED_REQUEST_HEADERS


def test_relayed_headers_carry_no_credentials():
    lowered = {h.lower() for h in S._FORWARDED_REQUEST_HEADERS}
    assert not lowered & {"authorization", "x-api-key", "cookie", "api-key"}


# ── free-tier cache affinity switch ─────────────────────────────────────────

def test_free_tier_affinity_is_off_by_default():
    """Spreading load is the right default for a shared proxy."""
    assert S._free_tier_cache_affinity_enabled({"server": {}}) is False


def test_free_tier_affinity_can_be_enabled():
    assert S._free_tier_cache_affinity_enabled(
        {"server": {"free_tier_cache_affinity": True}}) is True


def test_affinity_ordering_is_deterministic_for_one_key():
    """Stickiness is the point: the same conversation must pick the same model."""
    cands = [("a", CFG, "m1"), ("b", CFG, "m2"), ("c", CFG, "m3")]
    first = S._order_by_cache_affinity(cands, "conv-1")
    assert S._order_by_cache_affinity(cands, "conv-1") == first


def test_affinity_ordering_never_drops_or_needs_a_key():
    cands = [("a", CFG, "m1"), ("b", CFG, "m2")]
    assert S._order_by_cache_affinity(cands, None) is cands
    assert len(S._order_by_cache_affinity(cands, "k")) == 2


# ── worker default ──────────────────────────────────────────────────────────

def test_worker_count_defaults_to_one():
    """Quota, health and saturation state is per-process and unshared."""
    from llmproxy import __main__ as m
    assert m._config_int_from({}, "workers", 1) == 1


@pytest.mark.parametrize("raw,expected", [
    (4, 4), ("2", 2), (None, 1), (True, 1), ("abc", 1), ({}, 1),
])
def test_worker_count_is_defensive(raw, expected):
    from llmproxy import __main__ as m
    assert m._config_int_from({"workers": raw}, "workers", 1) == expected
