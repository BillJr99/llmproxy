"""Tests for tool-result signal extraction and the tier adjustment it drives.

These are pure-function tests over canned agentic transcripts — no network, no
config, no Flask. The scenarios are the ones that matter in practice: a plain
chat that must stay untouched, a short exchange that hit one error and must NOT
escalate, a genuine error spiral that must, a productive run that must not, and
the two hard overrides (context compaction, clean test finish).
"""

from __future__ import annotations

import pytest

from llmproxy import signals


def _assistant(*tool_calls):
    """An assistant turn invoking (name, arguments_json) tool calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(tool_calls)
        ],
    }


def _tool(text):
    return {"role": "tool", "tool_call_id": "call_0", "content": text}


def _user(text):
    return {"role": "user", "content": text}


def _spiral(turns=10, command="cargo build", result="error: compilation failed"):
    msgs = [_user("fix the build")]
    for _ in range(turns):
        msgs += [_assistant(("bash", f'{{"command": "{command}"}}')), _tool(result)]
    return {"messages": msgs}


# — extraction —

def test_plain_chat_produces_no_signal():
    sig = signals.extract_tool_signals({"messages": [_user("what is 2 + 2")]})
    assert not sig.has_signal
    assert signals.tier_adjustment({"messages": [_user("what is 2 + 2")]}) == (0, signals.SOURCE_NEUTRAL)


def test_malformed_payloads_never_raise():
    for payload in (None, {}, {"messages": None}, {"messages": [None, 3, "x"]},
                    {"messages": [{"role": "tool"}]}):
        assert signals.tier_adjustment(payload)[0] in (-1, 0, 1)


def test_tool_names_are_normalized_across_harnesses():
    payload = {"messages": [
        _user("go"),
        _assistant(("mcp__fs__write_file", "{}"), ("functions.Edit", "{}"), ("TodoWrite", "{}")),
        _tool("ok"),
    ]}
    sig = signals.extract_tool_signals(payload)
    assert sig.write_count == 1
    assert sig.edit_count == 1
    assert sig.plan_count == 1


@pytest.mark.parametrize("command,bucket", [
    ("cat file.txt", "read"),
    ("grep -r foo .", "read"),
    ("sed -i 's/a/b/' f.py", "edit"),
    ("cat > out.txt", "write"),
    ("echo hi | tee log.txt", "write"),
    # Redirection beats the read intent: this writes, it does not read.
    ("grep foo bar.txt > matches.txt", "write"),
])
def test_bash_intent_recovery(command, bucket):
    payload = {"messages": [_user("go"), _assistant(("bash", f'{{"command": "{command}"}}')), _tool("ok")]}
    sig = signals.extract_tool_signals(payload)
    assert getattr(sig, f"{bucket}_count") == 1


def test_zero_counts_do_not_read_as_failures():
    """"0 failed" is what success looks like; matching it would invert the signal."""
    assert signals._severity_of("test result: ok. 0 failed; 0 errors") == signals.SEVERITY_NONE
    assert signals._severity_of("3 failed") == signals.SEVERITY_SOFT


def test_mixed_test_output_is_not_a_pass():
    assert not signals._looks_like_tests_passed("1 failed, 4 passed")
    assert signals._looks_like_tests_passed("5 passed in 0.31s")


# — calibration —

def test_one_maxed_dimension_sits_below_the_threshold():
    """The corroboration property: no single signal may flip the tier alone."""
    import math
    one_signal = math.tanh(signals.SCORE_GAIN * signals.SIGNAL_UNIT * 1.0)
    assert one_signal < signals.DEFAULT_THRESHOLD
    assert one_signal == pytest.approx(0.462, abs=0.001)


def test_shallow_conversation_with_one_error_does_not_escalate():
    payload = {"messages": [
        _user("fix it"),
        _assistant(("bash", '{"command": "pytest"}')),
        _tool("Traceback (most recent call last): ..."),
    ]}
    delta, source = signals.tier_adjustment(payload)
    assert delta == 0
    assert source == signals.SOURCE_AMBIGUOUS


def test_stall_requires_minimum_turn_depth():
    shallow = _spiral(turns=signals.STALL_MIN_TURN_DEPTH - 3)
    deep = _spiral(turns=signals.STALL_MIN_TURN_DEPTH + 2)
    assert signals.tier_adjustment(shallow)[0] == 0
    assert signals.tier_adjustment(deep)[0] == 1


# — the scenarios —

def test_error_spiral_escalates():
    delta, source = signals.tier_adjustment(_spiral())
    assert delta == 1
    assert source == signals.SOURCE_DIMENSIONS


def test_deep_but_productive_run_does_not_escalate():
    msgs = [_user("build the feature")]
    for _ in range(10):
        msgs += [_assistant(("edit", '{"file": "a.py"}')), _tool("edited a.py")]
    assert signals.tier_adjustment({"messages": msgs})[0] == 0


def test_exploring_scores_lower_than_spinning():
    """Reading and planning is working the problem; neither is being stuck."""
    exploring = signals.dimensions_from_signals(signals.extract_tool_signals(
        _spiral(turns=10, command="cat main.rs", result="fn main() {}")))
    spinning = signals.dimensions_from_signals(signals.extract_tool_signals(
        _spiral(turns=10, command="cargo build", result="ok")))
    assert exploring.exploring == 1.0 and exploring.spinning == 0.0
    assert spinning.spinning == 1.0 and spinning.exploring == 0.0


def test_clean_finish_de_escalates():
    payload = {"messages": [
        _user("fix the bug"),
        _assistant(("write", '{"file": "a.py"}')), _tool("written"),
        _assistant(("bash", '{"command": "pytest"}')), _tool("5 passed in 0.31s"),
    ]}
    assert signals.tier_adjustment(payload) == (-1, signals.SOURCE_TESTS_PASSED)


def test_passing_tests_without_recent_writes_do_not_de_escalate():
    payload = {"messages": [
        _user("check the tests"),
        _assistant(("bash", '{"command": "pytest"}')), _tool("5 passed in 0.31s"),
    ]}
    assert signals.tier_adjustment(payload)[0] == 0


def test_critical_severity_overrides_everything():
    payload = {"messages": [
        _user("go"),
        _assistant(("edit", "{}")), _tool("edited"),
        _assistant(("bash", '{"command": "make"}')), _tool("fatal: out of memory"),
    ]}
    assert signals.tier_adjustment(payload) == (1, signals.SOURCE_OVERRIDE)


def test_compaction_escalates_and_is_self_latching():
    """A compaction summary erases the evidence, so it must escalate on its own.

    The summary stays in the prefix for the rest of the session, which is what
    keeps a hard task from snapping back to the weakest tier right after it is
    compacted.
    """
    payload = {"messages": [
        _user("This session is being continued from a previous conversation that ran out of context."),
        _user("keep going"),
    ]}
    assert signals.tier_adjustment(payload) == (1, signals.SOURCE_OVERRIDE)


def test_only_the_recent_window_establishes_a_clean_finish():
    """An old green test run says nothing about the current state."""
    msgs = [_user("go"), _assistant(("write", '{"f": "a"}')), _tool("written"),
            _assistant(("bash", '{"command": "pytest"}')), _tool("5 passed in 0.1s")]
    msgs += [_assistant(("bash", '{"command": "ls"}')), _tool("a.py")] * signals.RECENT_WINDOW_MESSAGES
    assert not signals.extract_tool_signals({"messages": msgs}).tests_passed
