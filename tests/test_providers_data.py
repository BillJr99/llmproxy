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


@pytest.mark.parametrize("pkey", list(SIDE["providers"].keys()))
def test_every_cost_observed_id_is_prefixed_by_its_provider(pkey):
    """Same invariant as believed_free, and for the same reason: the key is the
    negative counterpart to it, and the defaults layer records both as
    provider-scoped so a bare id cannot leak the claim onto another provider
    that happens to serve the same upstream name."""
    prov = SIDE["providers"][pkey]
    for mid in prov.get("cost_observed_free_tier", []):
        assert mid.startswith(f"{pkey}/"), (
            f"{pkey}.cost_observed_free_tier entry {mid!r} should start with {pkey!r}/"
        )


@pytest.mark.parametrize("pkey", list(SIDE["providers"].keys()))
def test_cost_observed_is_a_list_of_strings(pkey):
    """GUARD: promotion writes a sorted list; a hand edit must not turn it into
    a dict or sneak a non-string in."""
    raw = SIDE["providers"][pkey].get("cost_observed_free_tier", [])
    assert isinstance(raw, list)
    assert all(isinstance(mid, str) for mid in raw)


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


# ---------------------------------------------------------------------------
# Review overrides
# ---------------------------------------------------------------------------
# Facts a human corrected while reviewing an automated providers.json refresh,
# and the reason each was corrected.
#
# The providers PR promotes a running deployment's routing metadata wholesale
# (server._promote_sidecar_to_providers) and drops provenance on the way in:
# the curated/observed/family/inferred grading lives only in that deployment's
# routing_metadata.json, never in this file. So once a refresh is merged there
# is nothing left in the repo saying which facts were guesses, and a later
# refresh re-proposing a rejected one looks identical to a new observation.
# These tables are that missing record. A refresh that reasserts a reviewed
# correction now fails CI here, with the reason, instead of quietly undoing it.
#
# To retire an entry, delete it and say why in the commit message. Note this
# guards the repo only: to stop a deployment from re-proposing a fact, clear it
# in that deployment's routing_metadata.json as well (admin UI -> routing
# metadata), or the next auto-PR arrives carrying it again.

# Cost observations a review found unproven, and the evidence that would settle
# each one.
#
# These are deliberately NOT permanent vetoes. A later refresh may well be
# right, and the cost of wrongly keeping a model free is one billed request --
# against the cost of wrongly marking it paid, which is irreversible downstream
# (cost_observed_free_tier is checked before every other free-ness signal,
# unions across routing layers rather than overriding, and doubles as the
# scraper's permanent denylist, so no deployment can un-observe it by hand).
# The asymmetry is why the burden of proof sits here. The test fails to force
# the check, not to forbid the fact.
#
# THE DISCRIMINATOR. usage.compute_cost returns (cost, source), and
# server._flag_paid_free records and logs that source but does not gate on it:
# it persists any cost > 0. So the two cases are already distinguishable in the
# evidence, just not in the code.
#
#   source="provider"  upstream reported usage.cost directly. A real charge
#                      against the account. CORROBORATED -- retire the entry
#                      below, note the evidence in the commit, and let the
#                      observation merge.
#   source="computed"  priced from the litellm cost map by multiplying token
#                      counts by catalog rates. That is a published rate for the
#                      model, not a bill to this account, and it is what a
#                      free-tier key returns whenever the catalog lists a price.
#                      FALSE ALARM -- keep the entry and clear the fact in the
#                      proposing deployment.
#   source="unknown"   no cost signal at all; returns 0.0 and cannot reach here.
#
# TO INVESTIGATE A RE-PROPOSAL, on the deployment that opened the PR:
#   grep "believed_free model .* reported a cost" <llmproxy log>
# The warning line ends in `source=<...>`. Check the account's billing page for
# the same period as a second signal. If both say a real charge, the entry below
# is wrong and should go.
_COST_OBSERVATIONS_NEEDING_CORROBORATION: dict[str, dict[str, str]] = {
    # PR #148 (2026-09). Proposed by a deployment whose Groq key is on the free
    # tier, where Groq serves both models at no charge. Groq does not report
    # usage.cost, and both models are in the litellm cost map, so every call
    # prices as source="computed" -- the false-alarm signature exactly. Not
    # re-checked against a billing statement, so this is unproven rather than
    # disproven: corroborate per above before accepting a re-proposal.
    "groq": {
        "groq/llama-3.3-70b-versatile": "free-tier key; cost was computed, not billed",
        "groq/openai/gpt-oss-120b": "free-tier key; cost was computed, not billed",
    },
}


@pytest.mark.parametrize("pkey", sorted(_COST_OBSERVATIONS_NEEDING_CORROBORATION))
def test_uncorroborated_cost_observations_are_not_merged(pkey):
    """A re-proposed cost observation must be investigated, not merged on repeat.

    Failing here does not mean the refresh is wrong. It means this model was
    flagged once on evidence that could not distinguish a real bill from a
    catalog price, and the same ambiguity has to be resolved before the fact
    lands in a file every deployment inherits.
    """
    observed = set(SIDE["providers"][pkey].get("cost_observed_free_tier", []))
    for mid, why in _COST_OBSERVATIONS_NEEDING_CORROBORATION[pkey].items():
        assert mid not in observed, (
            f"{mid!r} is back in {pkey}.cost_observed_free_tier. It was reviewed "
            f"once and found unproven: {why}. Before merging, check the "
            f"`source=` on the deployment's `believed_free model ... reported a "
            f"cost` warning. If source=provider, this is a real charge -- retire "
            f"the entry in _COST_OBSERVATIONS_NEEDING_CORROBORATION and say so in "
            f"the commit. If source=computed, it is the same false alarm -- drop "
            f"it from the diff and clear the fact in that deployment's "
            f"routing_metadata.json, or the next auto-PR carries it again."
        )


# Capability tags a refresh dropped that review restored, and how to check a
# repeat. Same posture as above -- a later refresh may be right, and the failure
# is there to force the check.
#
# The stakes are much lower here, which is why the evidence bar is lower too:
# model_capabilities unions across routing layers, so a tag missing from this
# file is only a weakened default and any provider listing that publishes it
# restores the behaviour at runtime. Nothing is irreversible.
#
# TO INVESTIGATE A REPEAT: fetch the provider's own catalog
# (GET <base_url>/models) and read what it publishes for the model, or send it a
# one-image request and see whether it answers or 400s. A provider that has
# genuinely dropped a modality is evidence; inference from the model id is not,
# and inference is what dropped these.
_REQUIRED_CAPABILITIES: dict[str, dict[str, set[str]]] = {
    # PR #148 (2026-09). Inference stripped `vision` from two models that are
    # vision-capable by name -- "-vl-" is Qwen's vision-language line, "-omni-"
    # its multimodal one -- while leaving the tag on their own siblings, which
    # is the tell that this was a naming heuristic misfiring rather than a
    # catalog change. Restored to the values curated before the refresh.
    "xkiro": {
        "xkiro/qwen/qwen3-vl-plus:free": {"vision"},
        "xkiro/qwen/qwen3-omni-flash:free": {"vision"},
    },
}


@pytest.mark.parametrize("pkey", sorted(_REQUIRED_CAPABILITIES))
def test_restored_capability_tags_are_not_dropped_again(pkey):
    """A re-dropped tag must be checked against the provider, not the model id.

    Failing here does not mean the refresh is wrong -- a provider really can
    withdraw a modality. It means a tag that inference got wrong once has gone
    again, and the catalog should say so before this file does.
    """
    caps = SIDE["providers"][pkey].get("model_capabilities", {})
    for mid, required in _REQUIRED_CAPABILITIES[pkey].items():
        assert mid in caps, (
            f"{pkey}.model_capabilities lost its entry for {mid!r} entirely. If "
            f"the provider has withdrawn the model, retire it from "
            f"_REQUIRED_CAPABILITIES and say so in the commit."
        )
        missing = required - set(caps[mid])
        assert not missing, (
            f"{mid!r} has lost {sorted(missing)} again (see _REQUIRED_CAPABILITIES). "
            f"Check the provider's own catalog before accepting it: if the listing "
            f"no longer publishes the modality, retire the entry and note the "
            f"evidence. If it still does, this is the naming heuristic misfiring "
            f"a second time -- restore the tag in the diff."
        )


# ---------------------------------------------------------------------------
# model_capabilities shape
# ---------------------------------------------------------------------------
# GUARD: this field carries the bulk of what an automated refresh writes and
# had no structural coverage at all until PR #148 grew it 36x in one diff.
# server._model_capabilities drops malformed entries silently at runtime, so
# without these the only symptom of a bad write is a routing decision that
# quietly stops happening.

@pytest.mark.parametrize("pkey", list(SIDE["providers"].keys()))
def test_model_capabilities_shape(pkey):
    raw = SIDE["providers"][pkey].get("model_capabilities", {})
    assert isinstance(raw, dict), f"{pkey}.model_capabilities must be a dict"
    for mid, tags in raw.items():
        assert mid.startswith(f"{pkey}/"), (
            f"{pkey}.model_capabilities key {mid!r} should start with {pkey!r}/ "
            f"-- the defaults layer records ids provider-scoped, so an unqualified "
            f"key matches no lookup and the tags never reach a routing decision."
        )
        assert isinstance(tags, list), f"{mid}: capabilities must be a list"
        assert all(isinstance(t, str) for t in tags), f"{mid}: tags must be strings"
        assert len(set(tags)) == len(tags), f"{mid}: duplicate capability tags"


def test_model_capability_tags_are_known():
    """Tags must be ones the router actually detects.

    server._CAPABILITIES is the source of truth rather than a literal restated
    here, for the same reason REASONING_LEVELS is imported rather than copied.
    Imported inside the test because llmproxy.server pulls in Flask and the rest
    of this module is a pure data check.
    """
    from llmproxy.server import _CAPABILITIES

    valid = set(_CAPABILITIES)
    for prov in SIDE["providers"].values():
        for mid, tags in (prov.get("model_capabilities") or {}).items():
            unknown = set(tags) - valid
            assert not unknown, (
                f"{mid!r} carries unknown capability tag(s) {sorted(unknown)}; "
                f"known tags are {sorted(valid)}. Unknown tags are dropped "
                f"silently at runtime, so this would be invisible in production."
            )


@pytest.mark.parametrize("pkey", list(SIDE["providers"].keys()))
def test_every_model_reasoning_id_is_prefixed_by_its_provider(pkey):
    """The prefix invariant believed_free and cost_observed_free_tier already
    have, applied to the tier map for the same reason."""
    for mid in SIDE["providers"][pkey].get("model_reasoning", {}):
        assert mid.startswith(f"{pkey}/"), (
            f"{pkey}.model_reasoning key {mid!r} should start with {pkey!r}/"
        )
