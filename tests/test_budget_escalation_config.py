"""Budget escalation is tunable, and its ceiling is clamped per model.

A `max_tokens: 5` ping took 30 seconds. Atria-Dawn-Preview is a reasoning
model: it spent all five tokens thinking, returned an empty completion with
`finish_reason: "length"`, and `_escalate_budget_if_starved` retried the same
candidate with a larger budget twice (5 -> 80 -> 1280) before it answered.

That behaviour is right — the alternative is handing the caller nothing — but
the three values governing it were hardcoded constants with no way to tune them.
They are `server` keys now, and the ceiling rose from 4096 to 65535.

That raise needs the clamp these tests pin. At 4096 the ceiling sat below almost
every model's output cap, so it could not ask for more than a model could give.
65535 is above what most models accept, and asking for more comes back as a 400
— which walks to the next candidate safely enough, but spends a round trip and
loses the answer the escalation existed to recover.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _load_server(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


def _server(tmp_path: Path, monkeypatch, *, model_context=None, **server_over):
    cfg = {
        "providers": {"p1": {"base_url": "http://p1.example/v1", "api_key": "k"}},
        "sync_believed_free_on_startup": False,
        "server": {"log_level": "ERROR", **server_over},
    }
    if model_context:
        cfg["model_context"] = model_context
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return _load_server(monkeypatch, tmp_path / "config.json")


@pytest.fixture
def S(tmp_path: Path, monkeypatch):
    return _server(tmp_path, monkeypatch)


def _ladder(S, start: int, provider: str | None = None, model: str = "m1") -> list[int]:
    """The full sequence of budgets escalation would walk from *start*."""
    payload = {"model": model, "max_tokens": start}
    out = [start]
    for _ in range(20):  # generous: a non-terminating bump must fail loudly
        bumped = S._bumped_budget(payload, provider)
        if bumped is None:
            return out
        payload = bumped
        out.append(payload["max_tokens"])
    raise AssertionError(f"budget did not terminate: {out}")


# ── defaults ────────────────────────────────────────────────────────────────

def test_the_factor_and_first_steps_are_unchanged(S):
    """The ladder from the logs. Only the ceiling moves in this change, so the
    early rungs must be exactly what production already does."""
    assert _ladder(S, 5)[:3] == [5, 80, 1280]


def test_the_raised_ceiling_is_reachable(S):
    """4096 was the old stopping point; it is not the stopping point now."""
    assert max(_ladder(S, 5)) > 4096


def test_the_ceiling_is_the_configured_default(S):
    assert _ladder(S, 5)[-1] == S._BUDGET_BUMP_CEILING == 65535


def test_an_uncapped_request_is_never_bumped(S):
    """GUARD: with no budget there is nothing to truncate on, so nothing to
    escalate."""
    assert S._bumped_budget({"model": "m1"}) is None


@pytest.mark.parametrize("budget", [0, -1, True, False, "80", None, 3.5])
def test_a_non_positive_or_non_int_budget_is_not_bumped(S, budget):
    """GUARD: `True` matters on its own — bool is an int subclass."""
    assert S._bumped_budget({"model": "m1", "max_tokens": budget}) is None


def test_max_completion_tokens_is_bumped_too(S):
    bumped = S._bumped_budget({"model": "m1", "max_completion_tokens": 5})
    assert bumped["max_completion_tokens"] == 80


# ── the per-model clamp ─────────────────────────────────────────────────────

def test_a_known_window_caps_the_bump(tmp_path, monkeypatch):
    """The guard that makes a 65535 ceiling safe. A model whose window is 8192
    must never be asked for more, however high the ceiling is set."""
    S = _server(tmp_path, monkeypatch, model_context={"p1/small": 8192})
    ladder = _ladder(S, 5, "p1", "small")
    assert max(ladder) == 8192
    assert all(step <= 8192 for step in ladder)


def test_an_unknown_window_leaves_the_ceiling_alone(tmp_path, monkeypatch):
    """GUARD: absent metadata is neutral, matching how context fit treats a
    missing context_length. It must never make things worse."""
    S = _server(tmp_path, monkeypatch, model_context={"p1/other": 8192})
    assert max(_ladder(S, 5, "p1", "unknown-model")) == 65535


def test_no_provider_name_leaves_the_ceiling_alone(S):
    """GUARD: callers that pass no provider (and every pre-existing test) keep
    the unclamped behaviour."""
    assert max(_ladder(S, 5, None, "m1")) == 65535


def test_a_bare_model_key_clamps_on_every_provider(tmp_path, monkeypatch):
    """model_context accepts a bare id, same as elsewhere in the config."""
    S = _server(tmp_path, monkeypatch, model_context={"shared": 2048})
    assert max(_ladder(S, 5, "p1", "shared")) == 2048


def test_a_budget_already_over_the_clamp_is_not_bumped(tmp_path, monkeypatch):
    """GUARD: no upward bump, and no downward rewrite either — the caller's
    budget is theirs, and llmproxy never clamps what it was given."""
    S = _server(tmp_path, monkeypatch, model_context={"p1/small": 8192})
    assert S._bumped_budget({"model": "small", "max_tokens": 9000}, "p1") is None


# ── the knobs ───────────────────────────────────────────────────────────────

def test_the_factor_is_configurable(tmp_path, monkeypatch):
    S = _server(tmp_path, monkeypatch, budget_escalation_factor=4)
    assert _ladder(S, 5)[:3] == [5, 20, 80]


def test_the_ceiling_is_configurable(tmp_path, monkeypatch):
    S = _server(tmp_path, monkeypatch, budget_escalation_ceiling=100)
    assert _ladder(S, 5) == [5, 80, 100]


@pytest.mark.parametrize("factor", [1, 0, -4])
def test_a_factor_that_makes_no_progress_terminates(tmp_path, monkeypatch, factor):
    """GUARD, and the one that would hang the proxy. A factor of 1 multiplies
    the budget by itself forever; `_ladder` raises rather than looping."""
    S = _server(tmp_path, monkeypatch, budget_escalation_factor=factor)
    assert _ladder(S, 5) == [5]


@pytest.mark.parametrize("bad", ["lots", None, [], {}, "16"])
def test_a_malformed_knob_falls_back_to_the_default(tmp_path, monkeypatch, bad):
    """GUARD: a typo costs tuning, not routing."""
    S = _server(tmp_path, monkeypatch, budget_escalation_factor=bad)
    assert _ladder(S, 5)[:3] == [5, 80, 1280]


# ── the retry loop ──────────────────────────────────────────────────────────

STARVED = json.dumps({
    "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
}).encode()
ANSWERED = json.dumps({
    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
}).encode()


def _resp(S, body: bytes, status: int = 200):
    from flask import Response
    return Response(body, status=status, content_type="application/json")


def _count_retries(S, monkeypatch, *, replies=None) -> int:
    """Run the escalation against a stub upstream; return the retry count."""
    calls = {"n": 0}
    queue = list(replies or [])

    def _fake(endpoint, pn, pc, payload, timeout, **kw):
        calls["n"] += 1
        return _resp(S, queue.pop(0) if queue else STARVED)

    monkeypatch.setattr(S, "_proxy_request", _fake)
    S._escalate_budget_if_starved(
        "chat/completions", "p1", {}, {"model": "m1", "max_tokens": 5},
        _resp(S, STARVED), 30, "test",
    )
    return calls["n"]


def test_the_retry_count_is_configurable(tmp_path, monkeypatch):
    S = _server(tmp_path, monkeypatch, budget_escalation_max_retries=1)
    assert _count_retries(S, monkeypatch) == 1


def test_escalation_can_be_switched_off(tmp_path, monkeypatch):
    """No second upstream call at all — the empty body goes straight back for
    ordinary failover to handle."""
    S = _server(tmp_path, monkeypatch, budget_escalation=False)
    assert _count_retries(S, monkeypatch) == 0


def test_zero_retries_is_the_same_as_off(tmp_path, monkeypatch):
    S = _server(tmp_path, monkeypatch, budget_escalation_max_retries=0)
    assert _count_retries(S, monkeypatch) == 0


def test_escalation_stops_as_soon_as_the_model_answers(tmp_path, monkeypatch):
    """GUARD: it spends no more calls than it needs."""
    S = _server(tmp_path, monkeypatch)
    assert _count_retries(S, monkeypatch, replies=[ANSWERED]) == 1


def test_a_non_starved_response_is_never_retried(S, monkeypatch):
    """GUARD: only an EMPTY body truncated on length qualifies. A model that
    produced output keeps its answer."""
    monkeypatch.setattr(S, "_proxy_request",
                        lambda *a, **k: pytest.fail("must not retry"))
    out = S._escalate_budget_if_starved(
        "chat/completions", "p1", {}, {"model": "m1", "max_tokens": 5},
        _resp(S, ANSWERED), 30, "test",
    )
    assert out.get_data() == ANSWERED


def test_the_cycle_deadline_still_wins(tmp_path, monkeypatch):
    """GUARD: this is the one place that spends several full timeouts on a
    single candidate, so an exhausted deadline must stop it whatever the retry
    budget says."""
    S = _server(tmp_path, monkeypatch, budget_escalation_max_retries=4)
    monkeypatch.setattr(S, "_timeout_for_candidate", lambda *a, **k: None)
    monkeypatch.setattr(S, "_proxy_request",
                        lambda *a, **k: pytest.fail("no room to retry"))
    out = S._escalate_budget_if_starved(
        "chat/completions", "p1", {}, {"model": "m1", "max_tokens": 5},
        _resp(S, STARVED), 30, "test", deadline=1.0,
    )
    assert out.get_data() == STARVED
