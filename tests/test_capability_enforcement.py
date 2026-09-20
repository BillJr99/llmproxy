"""A model known to lack a needed capability is never selected.

Three separate defects met here. `_lookup_capabilities` unions the qualified,
bare and normalized id forms, so a `:free` variant whose own listing omits
`tools` still read as capable via its paid sibling's joined entry. Ordering
could not have saved it anyway, because `_order_by_capability` only sorts, and
any later pass that re-sorts could put the model back in front. And nothing
learned from the upstream 404 that resulted.

The union itself is deliberate and must survive: a gateway publishing a terse
`supported_parameters` must not be able to retract what another provider
asserted about the same weights. Every test that guards it is marked as such.
"""

from __future__ import annotations

import pytest

import llmproxy.server as S
from llmproxy.flagship import normalize_model_id


@pytest.fixture(autouse=True)
def _clean_registries():
    S._capability_gap_registry.clear()
    S._reset_failures()
    yield
    S._capability_gap_registry.clear()
    S._reset_failures()


CAPS = {
    # A variant that says, on its own listing, that it cannot call tools.
    "openrouter/z-ai/glm-9.9:free": {"reasoning"},
    # The joined entry, carrying the paid sibling's capabilities.
    normalize_model_id("z-ai/glm-9.9"): {"tools", "reasoning"},
    # A non-variant model whose gateway publishes a thin listing.
    "terseprov/terse-model": {"vision"},
    normalize_model_id("terse-model"): {"tools", "vision"},
}


# ── the variant exception ───────────────────────────────────────────────────

def test_a_variants_own_listing_is_not_unioned_with_its_siblings():
    """Listing both a model and its `:free` id is discrimination, not terseness."""
    caps = S._lookup_capabilities(CAPS, "openrouter", "z-ai/glm-9.9:free")
    assert caps == {"reasoning"}
    assert S._capability_state("openrouter", "z-ai/glm-9.9:free", "tools", CAPS) == \
        S._CAP_KNOWN_INCAPABLE


def test_a_terse_listing_on_a_non_variant_id_still_unions():
    """GUARD: the regression the union was added to prevent.

    First-match-wins ranked a model BELOW an untagged one for a capability it
    demonstrably had. Only variant ids are exempted; this must stay unioned.
    """
    caps = S._lookup_capabilities(CAPS, "terseprov", "terse-model")
    assert caps == {"tools", "vision"}
    assert S._capability_state("terseprov", "terse-model", "tools", CAPS) == \
        S._CAP_KNOWN_CAPABLE


def test_a_variant_with_no_listing_of_its_own_still_inherits():
    """Nothing was said about this target specifically, so the join applies."""
    assert "tools" in S._lookup_capabilities(CAPS, "someoneelse", "z-ai/glm-9.9:free")


def test_the_incapable_variant_leaves_the_tools_pool():
    """Membership of llmproxy/tools__free is decided by the same lookup."""
    assert S._model_has_capability("openrouter", "z-ai/glm-9.9:free", "tools", CAPS) is False
    assert S._model_has_capability("terseprov", "terse-model", "tools", CAPS) is True


# ── dropping, not merely demoting ───────────────────────────────────────────

def _pool():
    return [("openrouter", {}, "z-ai/glm-9.9:free"), ("terseprov", {}, "terse-model")]


def test_a_known_incapable_candidate_is_dropped_not_demoted():
    """Ordering cannot express a hard requirement; a later pass can undo it."""
    kept, dropped = S._drop_known_incapable(_pool(), {"tools"}, CAPS)
    assert dropped == 1
    assert [um for _pn, _c, um in kept] == ["terse-model"]


def test_an_unknown_candidate_is_kept():
    """Sparse metadata must not empty a pool: unknown is not incapable."""
    pool = [("newprov", {}, "undocumented-model")]
    kept, dropped = S._drop_known_incapable(pool, {"tools"}, CAPS)
    assert dropped == 0 and kept == pool


def test_a_pool_of_only_incapable_candidates_is_left_intact():
    """A tried-and-failed-over request beats a 503 with no upstream call made."""
    pool = [("openrouter", {}, "z-ai/glm-9.9:free")]
    kept, dropped = S._drop_known_incapable(pool, {"tools"}, CAPS)
    assert dropped == 0 and kept == pool


def test_the_gate_is_a_no_op_when_the_request_needs_nothing():
    pool = _pool()
    kept, dropped = S._drop_known_incapable(pool, set(), CAPS)
    assert dropped == 0 and kept is pool


def test_the_gate_drops_then_orders():
    """_apply_capability_gate is the two passes callers always want together."""
    ordered, dropped = S._apply_capability_gate(_pool(), {"tools"}, CAPS, "t")
    assert dropped == 1
    assert [um for _pn, _c, um in ordered] == ["terse-model"]


# ── learning from a rejection ───────────────────────────────────────────────

OPENROUTER_404 = (
    b'{"error":{"message":"No endpoints found that support tool use. '
    b'Try disabling \\"browser_exec\\".","code":404,'
    b'"metadata":{"failed_routing_step":"Filter by Tool Compatibility"}}}'
)


def test_a_tool_compatibility_rejection_is_recognised():
    assert S._detect_capability_rejection(404, OPENROUTER_404) == "tools"


def test_an_ordinary_404_teaches_nothing():
    """The matcher must stay narrow: a false positive sidelines a good model."""
    assert S._detect_capability_rejection(
        404, b'{"error":{"message":"model not found"}}') is None


def test_a_rejection_mentioning_tools_in_passing_teaches_nothing():
    assert S._detect_capability_rejection(
        400, b'{"error":{"message":"your tools array is malformed"}}') is None


def test_a_learned_gap_outranks_a_listing_that_claims_the_capability():
    """The request already failed against this exact target; that settles it."""
    claims_tools = {"openrouter/wishful-model": {"tools"}}
    assert S._capability_state("openrouter", "wishful-model", "tools", claims_tools) == \
        S._CAP_KNOWN_CAPABLE
    S._note_capability_rejection("openrouter", "wishful-model", 404, OPENROUTER_404)
    assert S._capability_state("openrouter", "wishful-model", "tools", claims_tools) == \
        S._CAP_KNOWN_INCAPABLE
    assert S._model_has_capability("openrouter", "wishful-model", "tools", claims_tools) is False


def test_a_learned_gap_is_scoped_to_the_exact_target():
    """One deployment refusing says nothing about the same weights elsewhere."""
    S._note_capability_rejection("openrouter", "shared-model", 404, OPENROUTER_404)
    assert S._learned_capability_gaps("openrouter", "shared-model") == {"tools"}
    assert S._learned_capability_gaps("otherprov", "shared-model") == set()


def test_a_learned_gap_removes_the_target_from_selection():
    """The reactive path and the gate compose: learned once, never picked again."""
    pool = [("openrouter", {}, "wishful-model"), ("good", {}, "other-model")]
    caps = {"openrouter/wishful-model": {"tools"}, "good/other-model": {"tools"}}
    kept, dropped = S._drop_known_incapable(pool, {"tools"}, caps)
    assert dropped == 0
    S._note_capability_rejection("openrouter", "wishful-model", 404, OPENROUTER_404)
    kept, dropped = S._drop_known_incapable(pool, {"tools"}, caps)
    assert dropped == 1
    assert [um for _pn, _c, um in kept] == ["other-model"]
