"""`free_tier_cache_affinity` keeps a conversation on the model that worked.

The previous mechanism was rendezvous (HRW) hashing, which is stateless: it
remembers no choice, it derives one, and the winner is uncorrelated with rank.
On an unranked free pool that is invisible, because no candidate was better than
another to begin with. On `flagship__free`, whose entire premise is strict
best-first, it would hand most conversations a hash-chosen member from their
very first turn — which is why the pass used to be suppressed there, leaving the
flag doing nothing for exactly the pool a coding agent is most likely to use.

Sticky-until-failure works on both kinds of pool because the pin is SET by
whatever the ordering chose and only ever moves that target forward.
"""

from __future__ import annotations

import pytest

import llmproxy.server as S
from llmproxy import state

POOL = [("best", {}, "m-best"), ("second", {}, "m-2"), ("third", {}, "m-3")]
KEY = "conversation-A"


@pytest.fixture(autouse=True)
def _clean():
    S._reset_affinity_pins()
    S.get_backend().reset_saturation()
    yield
    S._reset_affinity_pins()
    S.get_backend().reset_saturation()


def _heads(pool, key):
    return [pn for pn, _cfg, _um in S._order_by_sticky_affinity(pool, key)]


# ── best-first survives ─────────────────────────────────────────────────────

def test_an_unpinned_conversation_keeps_the_incoming_order():
    """The first turn must still get whatever the ranking says is best."""
    assert _heads(POOL, KEY) == ["best", "second", "third"]


def test_the_pass_is_a_no_op_without_a_key():
    assert S._order_by_sticky_affinity(POOL, None) is POOL


def test_the_pass_is_a_no_op_on_a_single_candidate():
    single = [POOL[0]]
    assert S._order_by_sticky_affinity(single, KEY) is single


# ── stickiness ──────────────────────────────────────────────────────────────

def test_a_conversation_sticks_to_the_model_that_served_it():
    S._record_affinity_success(KEY, "second", "m-2")
    assert _heads(POOL, KEY)[0] == "second"


def test_sticking_reorders_without_dropping_anything():
    """Failover must be unaffected: every candidate is still reachable."""
    S._record_affinity_success(KEY, "second", "m-2")
    assert _heads(POOL, KEY) == ["second", "best", "third"]


def test_pins_do_not_leak_between_conversations():
    S._record_affinity_success(KEY, "second", "m-2")
    assert _heads(POOL, "conversation-B")[0] == "best"


def test_a_later_success_elsewhere_re_pins():
    """The pin names the last model that actually worked, so it self-corrects.

    This is why no explicit unpin is needed on failure: a turn whose pinned
    model fails and is then served by another candidate re-pins in the same
    request.
    """
    S._record_affinity_success(KEY, "second", "m-2")
    S._record_affinity_success(KEY, "third", "m-3")
    assert _heads(POOL, KEY)[0] == "third"


def test_a_pin_for_a_target_no_longer_in_the_pool_is_ignored():
    S._record_affinity_success(KEY, "gone", "m-gone")
    assert _heads(POOL, KEY) == ["best", "second", "third"]


# ── the pin never undoes a saturation demotion ──────────────────────────────

def test_a_cooling_pinned_target_is_not_promoted():
    """Hoisting a rate-limited model back to the front would waste every turn.

    The orderings demote a candidate cooling after a 402/429 to the back;
    stickiness must not fight that.
    """
    S._record_affinity_success(KEY, "second", "m-2")
    S._mark_saturated(S._usage_key("second", "m-2"), None)
    assert _heads(POOL, KEY)[0] == "best"


def test_the_pin_resumes_once_the_cooldown_clears():
    S._record_affinity_success(KEY, "second", "m-2")
    S._mark_saturated(S._usage_key("second", "m-2"), None)
    S.get_backend().reset_saturation()
    assert _heads(POOL, KEY)[0] == "second"


# ── the map is bounded ──────────────────────────────────────────────────────

def test_the_pin_map_respects_its_cap():
    """Keyed by conversation, so unbounded growth is a slow leak."""
    for i in range(state.AFFINITY_PIN_MAX + 50):
        S._record_affinity_success(f"convo-{i}", "best", "m-best")
    assert S.get_backend().affinity_count() <= state.AFFINITY_PIN_MAX


def test_an_expired_pin_is_dropped(monkeypatch):
    S._record_affinity_success(KEY, "second", "m-2")
    real = S.time.monotonic
    monkeypatch.setattr(S.time, "monotonic",
                        lambda: real() + state.AFFINITY_PIN_TTL_S + 1)
    assert S._affinity_pinned_target(KEY) is None


# ── the flag still gates it ─────────────────────────────────────────────────

def test_the_feature_is_off_by_default():
    """Free-tier ordering spreads load by default; stickiness is opt-in."""
    assert S._free_tier_cache_affinity_enabled({"server": {}}) is False
    assert S._free_tier_cache_affinity_enabled(
        {"server": {"free_tier_cache_affinity": True}}) is True


# ── the key has to survive the conversation ─────────────────────────────────
#
# The pass above is only ever reached with a key derived from the payload, and
# that derivation was the part that did not work. `_affinity_key` hashed the
# whole cacheable prefix, which an agent rewrites on every iteration, so the
# pin was written each turn under a key nothing ever looked up again. Every
# test above passed while the feature did nothing in production. These close
# that gap by driving the pass with real payloads.

def _agent_payload(turns: int) -> dict:
    messages = [
        {"role": "system", "content": "you are a coding agent. " * 40},
        {"role": "user", "content": "refactor the parser"},
    ]
    for i in range(turns):
        messages += [
            {"role": "assistant", "content": f"reading file {i}"},
            {"role": "tool", "content": f"contents {i}"},
        ]
    return {"messages": messages}


def test_a_pin_set_on_turn_one_is_honoured_forty_turns_later():
    """The end-to-end regression, through the real key derivation."""
    S._record_affinity_success(S._affinity_key(_agent_payload(0)), "third", "m-3")
    for turn in (1, 5, 20, 40):
        key = S._affinity_key(_agent_payload(turn))
        assert _heads(POOL, key)[0] == "third", f"lost the pin by turn {turn}"


def test_a_second_conversation_is_not_dragged_onto_the_first_ones_model():
    """GUARD: stickiness must not become a global pin. Two conversations under
    the same system prompt keep their own choices."""
    other = {
        "messages": [
            {"role": "system", "content": "you are a coding agent. " * 40},
            {"role": "user", "content": "write the docs"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    S._record_affinity_success(S._affinity_key(_agent_payload(0)), "third", "m-3")
    assert _heads(POOL, S._affinity_key(other))[0] == "best"


def test_the_pin_follows_a_failover_within_the_conversation():
    """The pin is written on SUCCESS, so a turn served by a different model
    re-pins to it, and the next turn starts there instead of replaying the
    failure that sent it away."""
    payload = _agent_payload(3)
    S._record_affinity_success(S._affinity_key(payload), "third", "m-3")
    assert _heads(POOL, S._affinity_key(payload))[0] == "third"
    S._record_affinity_success(S._affinity_key(_agent_payload(4)), "second", "m-2")
    assert _heads(POOL, S._affinity_key(_agent_payload(5)))[0] == "second"
