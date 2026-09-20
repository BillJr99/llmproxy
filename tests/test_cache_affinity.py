"""Tests for prompt-cache affinity via rendezvous (HRW) hashing.

The two things worth guarding are the degenerate case (a bare first turn has no
reusable prefix, so pinning it would cost load spreading and buy nothing) and
the minimal-disruption property that makes HRW the right choice over a modulo
ring.
"""

from __future__ import annotations

from collections import Counter

from llmproxy import server


def _cands(n):
    return [(f"p{i}", {}, "m") for i in range(n)]


def _winner(key, pool):
    return server._order_by_cache_affinity(pool, key)[0][0]


# — key derivation —

def test_bare_first_turn_has_no_affinity_key():
    """No reusable prefix exists yet, so affinity must stay out of the way."""
    assert server._affinity_key({"messages": [{"role": "user", "content": "hello"}]}) is None


def test_explicit_prompt_cache_key_wins():
    payload = {"prompt_cache_key": "conv-42", "messages": [{"role": "user", "content": "hi"}]}
    assert server._affinity_key(payload) == "conv-42"


def test_prompt_cache_key_is_read_from_metadata_too():
    assert server._affinity_key({"metadata": {"prompt_cache_key": "c9"}, "messages": []}) == "c9"


def test_oversized_client_key_is_truncated():
    payload = {"prompt_cache_key": "x" * 9000, "messages": []}
    assert len(server._affinity_key(payload)) == server._AFFINITY_KEY_MAX_CHARS


def test_continuation_keys_on_the_stable_prefix():
    """The trailing user turn changes every request; the root does not."""
    prefix = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    first = server._affinity_key({"messages": prefix + [{"role": "user", "content": "next"}]})
    second = server._affinity_key({"messages": prefix + [{"role": "user", "content": "different"}]})
    assert first is not None and first == second


def test_an_agent_loop_keeps_one_key_for_the_whole_conversation():
    """The regression this whole mechanism turned on.

    Keying on the full cacheable prefix made the key a function of the
    transcript LENGTH, so an agent appending an assistant turn and a tool
    result per iteration got a brand-new key on every request. The sticky pin
    was written every turn and read back never, and each turn restarted from
    the top of the ranking — paying a 429 on every model above the pinned one
    before reaching it again.
    """
    messages = [
        {"role": "system", "content": "s" * 400},
        {"role": "user", "content": "do the thing"},
    ]
    keys = {server._affinity_key({"messages": messages})}
    for i in range(20):
        messages = messages + [
            {"role": "assistant", "content": f"calling tool {i}"},
            {"role": "tool", "content": f"result {i}"},
        ]
        keys.add(server._affinity_key({"messages": messages}))
    messages = messages + [{"role": "user", "content": "and now this"}]
    keys.add(server._affinity_key({"messages": messages}))
    assert len(keys) == 1 and None not in keys


def test_a_first_turn_and_its_continuation_share_a_key():
    """Turn 1 pins the model that worked; turn 2 has to be able to read that
    pin, which means both turns must land in the same key space."""
    root = [
        {"role": "system", "content": "s" * 400},
        {"role": "user", "content": "open the file"},
    ]
    turn_two = root + [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now close it"},
    ]
    assert server._affinity_key({"messages": root}) == server._affinity_key(
        {"messages": turn_two}
    )


def test_two_conversations_under_one_system_prompt_stay_apart():
    """The root includes the first user turn, so one agent's system prompt does
    not funnel every conversation it starts onto a single model."""
    shared = {"role": "system", "content": "s" * 400}
    a = server._affinity_key({"messages": [shared, {"role": "user", "content": "task A"}]})
    b = server._affinity_key({"messages": [shared, {"role": "user", "content": "task B"}]})
    assert a != b


def test_different_conversations_get_different_keys():
    def key(seed):
        return server._affinity_key({"messages": [
            {"role": "user", "content": seed},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "go on"},
        ]})
    assert key("alpha") != key("beta")


def test_substantial_system_prompt_is_cacheable_on_a_first_turn():
    """An agent resends its system prompt verbatim, so a first turn carrying one
    is already identifiable and does not have to wait for turn 2 to be pinned."""
    payload = {"messages": [
        {"role": "system", "content": "x" * server._AFFINITY_MIN_SYSTEM_CHARS},
        {"role": "user", "content": "hi"},
    ]}
    assert (server._affinity_key(payload) or "").startswith("conv:")


def test_short_system_prompt_is_not_worth_pinning_for():
    payload = {"messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]}
    assert server._affinity_key(payload) is None


def test_malformed_payloads_return_none():
    for payload in (None, {}, {"messages": "nope"}, {"messages": [None, 1]}):
        assert server._affinity_key(payload) is None


# — HRW properties —

def test_ordering_is_deterministic_for_one_key():
    pool = _cands(6)
    assert [_winner("k", pool) for _ in range(20)].count(_winner("k", pool)) == 20


def test_ordering_never_drops_a_candidate():
    pool = _cands(6)
    assert len(server._order_by_cache_affinity(pool, "k")) == len(pool)


def test_no_key_is_a_no_op():
    pool = _cands(6)
    assert server._order_by_cache_affinity(pool, None) == pool


def test_keys_spread_across_the_pool():
    """Affinity must not collapse every conversation onto one candidate."""
    pool = _cands(8)
    spread = Counter(_winner(f"k{i}", pool) for i in range(400))
    assert len(spread) == 8
    assert max(spread.values()) < 400 * 0.35


def test_growing_the_pool_reshuffles_minimally():
    """The reason for HRW over a modulo ring: churn stays near the 1/N ideal."""
    keys = [f"k{i}" for i in range(400)]
    small, large = _cands(8), _cands(9)
    moved = sum(1 for k in keys if _winner(k, small) != _winner(k, large))
    assert moved / len(keys) < 0.2  # ideal is 1/9 ≈ 11%; a modulo ring moves ~89%


def test_losing_a_candidate_only_moves_its_own_share():
    keys = [f"k{i}" for i in range(400)]
    full = _cands(8)
    reduced = [c for c in full if c[0] != "p3"]
    moved = sum(1 for k in keys if _winner(k, full) != _winner(k, reduced))
    pinned_to_lost = sum(1 for k in keys if _winner(k, full) == "p3")
    assert moved == pinned_to_lost


def test_rendezvous_rank_separator_prevents_collisions():
    assert server._rendezvous_rank("ab", "c") != server._rendezvous_rank("a", "bc")


# — integration with account expansion —

def test_accounts_are_pinned_per_conversation(monkeypatch):
    from llmproxy.config import Account

    accounts = [Account(id=f"a{i}", key=f"k{i}", key_raw=f"k{i}", label=None, priority=i)
                for i in range(4)]
    monkeypatch.setattr(server, "provider_accounts", lambda cfg: accounts)
    monkeypatch.setattr(server, "provider_account_strategy", lambda cfg: "round_robin")
    monkeypatch.setattr(server, "account_bound_cfg", lambda cfg, acct: {"_acct": acct.id})
    monkeypatch.setattr(server, "_is_candidate_saturated", lambda *a, **k: False)

    payload = {"prompt_cache_key": "conv-1", "messages": [{"role": "user", "content": "hi"}]}
    picks = {server._expand_accounts([("p", {}, "m")], payload)[0][1]["_acct"] for _ in range(20)}
    assert len(picks) == 1, "the same conversation must keep landing on one credential"

    others = {
        server._expand_accounts([("p", {}, "m")], {"prompt_cache_key": f"c{i}", "messages": []})[0][1]["_acct"]
        for i in range(40)
    }
    assert len(others) > 1, "different conversations must still spread"


# — reporting accuracy —

def test_affinity_is_not_reported_where_it_cannot_apply(monkeypatch):
    """A header claiming a pass fired when it changed nothing is worse than none."""
    monkeypatch.setattr(server, "provider_accounts", lambda cfg: ["only-one"])
    assert server._cache_affinity_applies([("p", {}, "m")], {}, False) is False


def test_affinity_applies_with_multiple_round_robin_credentials(monkeypatch):
    monkeypatch.setattr(server, "provider_accounts", lambda cfg: ["a", "b"])
    monkeypatch.setattr(server, "provider_account_strategy", lambda cfg: "round_robin")
    assert server._cache_affinity_applies([("p", {}, "m")], {}, False) is True


def test_priority_account_strategy_is_left_alone(monkeypatch):
    """An operator who ranked their credentials meant it."""
    monkeypatch.setattr(server, "provider_accounts", lambda cfg: ["a", "b"])
    monkeypatch.setattr(server, "provider_account_strategy", lambda cfg: "priority")
    assert server._cache_affinity_applies([("p", {}, "m")], {}, False) is False
