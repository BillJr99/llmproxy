"""
Tool-result signals — inferring how an agentic conversation is *going*.

llmproxy's reasoning-tier triage (``server._target_reasoning_tier``) sizes a
request by its prompt length. That works for one-shot chat and fails for coding
agents: a short prompt can carry a hard task, and by turn twelve the thing that
actually predicts difficulty is not the prompt at all — it is whether the agent
is making progress. Agentic clients send that evidence with every request, in
the ``role: "tool"`` results already sitting in the message list.

This module reads those results and answers one question: should the request be
routed a tier *up* (the agent is stuck), a tier *down* (it is finishing
cleanly), or left where prompt size put it. It costs nothing — no extra model
call, no network, no config — and it is deliberately free of Flask, ``requests``
and ``load_config`` so it can be tested against canned transcripts.

Adapted from NVIDIA-NeMo/Switchyard 0.2.0 (Apache-2.0), specifically
``crates/libsy/src/algorithms/util/tool_signals.rs`` and ``.../stage.rs``.
Translated from Rust and reworked against llmproxy's canonical OpenAI-shaped
message list; the pattern tables are extended and maintained here, and
Switchyard's LLM-classifier consultation path is deliberately not carried over.
See ``THIRD_PARTY_NOTICES.md`` for the full attribution and change statement.

The scoring constants are Switchyard's and are load-bearing — see
``score_signals`` for why they are what they are.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

# — severity tiers —
#
# A tool result that failed is not automatically a hard signal. "exit code 1"
# from a linter is routine; an OOM kill is not. Three tiers keep that distinction
# rather than collapsing every failure into "error".
SEVERITY_NONE = 0.0
SEVERITY_SOFT = 0.3
SEVERITY_HARD = 0.7
SEVERITY_CRITICAL = 1.0

# Substrings matched against lowercased tool-result text, most severe first.
# Ordering matters: the first match wins, so a traceback that also mentions
# "error" is scored HARD rather than SOFT.
_ERROR_PATTERNS: tuple[tuple[str, float], ...] = (
    # Critical — the environment itself is broken. Retrying the same model on
    # the same box will not help; these mean escalate now.
    ("out of memory", SEVERITY_CRITICAL),
    ("oomkilled", SEVERITY_CRITICAL),
    ("connection refused", SEVERITY_CRITICAL),
    ("no space left on device", SEVERITY_CRITICAL),
    ("disk quota exceeded", SEVERITY_CRITICAL),
    ("segmentation fault", SEVERITY_CRITICAL),
    ("killed (signal", SEVERITY_CRITICAL),
    # Hard — a real, specific failure the agent has to reason about.
    ("traceback (most recent call last)", SEVERITY_HARD),
    ("modulenotfounderror", SEVERITY_HARD),
    ("importerror", SEVERITY_HARD),
    ("syntaxerror", SEVERITY_HARD),
    ("unhandled exception", SEVERITY_HARD),
    ("fatal error", SEVERITY_HARD),
    ("panic:", SEVERITY_HARD),
    ("compilation failed", SEVERITY_HARD),
    ("build failed", SEVERITY_HARD),
    ("cannot find module", SEVERITY_HARD),
    ("command not found", SEVERITY_HARD),
    ("permission denied", SEVERITY_HARD),
    ("timed out", SEVERITY_HARD),
    ("timeout after", SEVERITY_HARD),
    # "no such file or directory" is the highest-yield single pattern in agent
    # traces and also the easiest to over-fire on, since tools echo it while
    # probing for optional files. Kept at HARD deliberately: it is paired with
    # the corroboration requirement in score_signals rather than trusted alone.
    ("no such file or directory", SEVERITY_HARD),
    # Soft — something went wrong, but routinely and recoverably.
    ("error:", SEVERITY_SOFT),
    ("exit code 1", SEVERITY_SOFT),
    ("exit status 1", SEVERITY_SOFT),
    ("non-zero exit", SEVERITY_SOFT),
)

# Counts of zero are printed by almost every test runner on success
# ("0 failed", "0 errors"). Matching them as failures would invert the signal,
# so numeric failure phrases must carry a non-zero count to register.
_NONZERO_FAILURE_RE = re.compile(
    r"\b(?!0\b)(\d+)\s+(?:test(?:s)?\s+)?(?:failed|failures|errors)\b"
)

# Deliberately biased toward false negatives: missing a passing run costs one
# unnecessary tier, while a false "tests passed" would de-escalate a broken
# task down to the weakest model available.
_TEST_PASS_PHRASES: tuple[str, ...] = (
    "all tests passed",
    "test result: ok",
    "tests passed",
    "build succeeded",
    "0 failed",
    "✓ all",
)

_PYTEST_PASS_RE = re.compile(r"\b(\d+)\s+passed\b")

# — tool taxonomy —
#
# The same action has a different name in every harness. llmproxy serves
# OpenAI-SDK clients, opencode, Cursor and Anthropic-native clients through one
# canonical schema, so the taxonomy has to speak all of their vocabularies or the
# signal silently disappears for whichever client is not listed.
_EDIT_TOOLS = frozenset({
    "edit", "str_replace_editor", "str_replace_based_edit_tool", "apply_patch",
    "apply_diff", "multiedit", "edit_file", "replace_in_file", "patch",
})
_WRITE_TOOLS = frozenset({
    "write", "write_file", "create_file", "notebookedit", "create",
})
_READ_TOOLS = frozenset({
    "read", "read_file", "view", "cat", "glob", "grep", "search", "ls",
    "list_files", "find", "codebase_search", "file_search", "webfetch",
    "websearch",
})
_PLAN_TOOLS = frozenset({
    "todowrite", "update_plan", "task", "exit_plan_mode", "exitplanmode",
})
_BASH_TOOLS = frozenset({
    "bash", "shell", "shell_command", "local_shell_call", "terminal",
    "run_command", "run_terminal_cmd", "execute_command",
})

# A shell call is read-or-write depending on what it runs. Redirection and
# in-place editing win over a read intent: `grep foo bar > out.txt` writes.
_BASH_WRITE_PATTERNS: tuple[str, ...] = (
    " > ", ">>", "tee ", "<<'eof'", '<<"eof"', "<<eof", "mkdir ", "touch ",
    "cp ", "mv ", "install ",
)
_BASH_EDIT_PATTERNS: tuple[str, ...] = (
    "sed -i", "perl -pi", "perl -i", "patch ", "git apply",
)
_BASH_READ_PATTERNS: tuple[str, ...] = (
    "cat ", "head ", "tail ", "ls ", "grep ", "rg ", "find ", "git diff",
    "git log", "git status", "wc ", "pwd", "which ",
)

# Markers that the conversation carries a context-compaction summary rather than
# the original transcript.
_COMPACTION_MARKERS: tuple[str, ...] = (
    "this session is being continued from a previous conversation",
    "continued from a previous conversation",
    "context was compacted",
    "conversation summary:",
    "<summary>",
    "summary of the conversation so far",
)

# — windows and thresholds —

# Only the tail of a conversation says what is happening *now*. Everything
# before it is history that a recovered agent should not keep paying for.
RECENT_WINDOW_MESSAGES = 12

# A short conversation cannot be "stuck" — it has not had time to be. Below this
# depth the stall dimensions are held at zero, which is what stops a three-turn
# exchange that hit one error from being read as a spiral.
STALL_MIN_TURN_DEPTH = 8

# Writes/edits in the recent window that count as "fully producing". Kept low
# because agents batch edits: three in a window is a working agent.
PRODUCTION_SATURATION = 3

# — scoring constants (Switchyard's, retained deliberately) —
#
# These two are calibrated together against DEFAULT_THRESHOLD and should not be
# tuned independently. With SIGNAL_UNIT=0.10 and SCORE_GAIN=5.0, a single
# maxed dimension scores tanh(5.0 * 0.10 * 1.0) = 0.462 — just *under* a 0.5
# threshold. That is the whole point: one signal is never enough on its own, so
# a tier switch requires corroboration across dimensions. It also makes the
# threshold interpretable as "how many agreeing signals do I demand" — roughly
# 0.3 escalates on one, 0.5 on one-and-a-half, 0.7 on two.
SIGNAL_UNIT = 0.10
SCORE_GAIN = 5.0
DEFAULT_THRESHOLD = 0.5

# Decision sources, reported alongside the verdict so a routing choice can be
# explained after the fact rather than reverse-engineered from logs.
SOURCE_NEUTRAL = "neutral"
SOURCE_OVERRIDE = "override"
SOURCE_TESTS_PASSED = "tests_passed"
SOURCE_DIMENSIONS = "dimensions"
SOURCE_AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ToolSignals:
    """What the tool results in one request say about the agent's progress."""

    severity: float = 0.0
    turn_depth: int = 0
    compacted: bool = False
    tests_passed: bool = False
    edit_count: int = 0
    write_count: int = 0
    read_count: int = 0
    plan_count: int = 0
    recent_edit_count: int = 0
    recent_write_count: int = 0
    recent_read_count: int = 0
    recent_plan_count: int = 0
    tool_result_count: int = 0

    @property
    def has_signal(self) -> bool:
        """True when this request carried agentic evidence worth scoring.

        A plain chat completion produces an all-zero ``ToolSignals``; callers use
        this to leave prompt-size triage untouched rather than scoring noise.
        """
        return self.tool_result_count > 0 or self.turn_depth > 0 or self.compacted


@dataclass(frozen=True)
class CodingAgentDimensions:
    """``ToolSignals`` projected onto the four axes the scorer actually uses."""

    severity: float = 0.0
    spinning: float = 0.0
    exploring: float = 0.0
    production_intensity: float = 0.0


def _text_of(content) -> str:
    """Flatten canonical message content (str or content-part list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


def _normalize_tool_name(name) -> str:
    """Lowercase a tool name and strip the namespacing harnesses add.

    ``mcp__github__create_issue`` and ``functions.bash`` both reduce to their
    final segment so the taxonomy matches regardless of how a client namespaces.
    """
    if not isinstance(name, str):
        return ""
    lowered = name.strip().lower()
    for sep in ("__", "."):
        if sep in lowered:
            lowered = lowered.rsplit(sep, 1)[-1]
    return lowered


def _bash_intent(arguments) -> str | None:
    """Classify a shell invocation as ``write``/``edit``/``read``, or None.

    Redirection and in-place edits are checked before reads, because
    ``grep foo bar > out`` is a write that happens to mention a read command.
    """
    command = ""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            command = parsed.get("command") or parsed.get("cmd") or ""
        else:
            command = arguments
    elif isinstance(arguments, dict):
        command = arguments.get("command") or arguments.get("cmd") or ""
    if not isinstance(command, str) or not command:
        return None
    lowered = command.lower()
    if any(p in lowered for p in _BASH_EDIT_PATTERNS):
        return "edit"
    if any(p in lowered for p in _BASH_WRITE_PATTERNS):
        return "write"
    if any(p in lowered for p in _BASH_READ_PATTERNS):
        return "read"
    return None


def _severity_of(text: str) -> float:
    """Score one tool result's text against the error table (first match wins)."""
    if not text:
        return SEVERITY_NONE
    lowered = text.lower()
    for pattern, severity in _ERROR_PATTERNS:
        if pattern in lowered:
            return severity
    if _NONZERO_FAILURE_RE.search(lowered):
        return SEVERITY_SOFT
    return SEVERITY_NONE


def _looks_like_tests_passed(text: str) -> bool:
    """True when a tool result reads as a clean test or build run.

    Requires the absence of a non-zero failure count as well as a positive
    phrase, so "1 failed, 4 passed" does not read as success.
    """
    if not text:
        return False
    lowered = text.lower()
    if _NONZERO_FAILURE_RE.search(lowered):
        return False
    if any(phrase in lowered for phrase in _TEST_PASS_PHRASES):
        return True
    match = _PYTEST_PASS_RE.search(lowered)
    return bool(match and int(match.group(1)) > 0)


def extract_tool_signals(payload: dict) -> ToolSignals:
    """Walk a canonical request's messages and derive its ``ToolSignals``.

    Reads assistant ``tool_calls`` for *what the agent did* and ``role: "tool"``
    messages for *how it went*. Never raises on malformed input — a message list
    that does not parse simply contributes no signal, because a routing
    heuristic must not be able to fail a request.
    """
    if not isinstance(payload, dict):
        return ToolSignals()
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return ToolSignals()

    recent_start = max(0, len(messages) - RECENT_WINDOW_MESSAGES)
    counts = {"edit": 0, "write": 0, "read": 0, "plan": 0}
    recent_counts = {"edit": 0, "write": 0, "read": 0, "plan": 0}
    severity = SEVERITY_NONE
    tests_passed = False
    compacted = False
    turn_depth = 0
    tool_result_count = 0

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        is_recent = idx >= recent_start

        if role == "assistant":
            turn_depth += 1
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                fn = fn if isinstance(fn, dict) else {}
                name = _normalize_tool_name(fn.get("name") or call.get("name"))
                if not name:
                    continue
                bucket = None
                if name in _EDIT_TOOLS:
                    bucket = "edit"
                elif name in _WRITE_TOOLS:
                    bucket = "write"
                elif name in _PLAN_TOOLS:
                    bucket = "plan"
                elif name in _READ_TOOLS:
                    bucket = "read"
                elif name in _BASH_TOOLS:
                    bucket = _bash_intent(fn.get("arguments") or call.get("arguments"))
                if bucket:
                    counts[bucket] += 1
                    if is_recent:
                        recent_counts[bucket] += 1

        elif role == "tool":
            tool_result_count += 1
            text = _text_of(msg.get("content"))
            if is_recent:
                severity = max(severity, _severity_of(text))
                # Only the recent window can establish a clean finish; an old
                # green test run says nothing about the current state.
                if _looks_like_tests_passed(text):
                    tests_passed = True

        if not compacted:
            probe = _text_of(msg.get("content")).lower()
            if probe and any(marker in probe for marker in _COMPACTION_MARKERS):
                compacted = True

    return ToolSignals(
        severity=severity,
        turn_depth=turn_depth,
        compacted=compacted,
        tests_passed=tests_passed,
        edit_count=counts["edit"],
        write_count=counts["write"],
        read_count=counts["read"],
        plan_count=counts["plan"],
        recent_edit_count=recent_counts["edit"],
        recent_write_count=recent_counts["write"],
        recent_read_count=recent_counts["read"],
        recent_plan_count=recent_counts["plan"],
        tool_result_count=tool_result_count,
    )


def dimensions_from_signals(sig: ToolSignals) -> CodingAgentDimensions:
    """Project ``ToolSignals`` onto the four scoring axes.

    ``spinning`` and ``exploring`` partition the not-producing case: an agent
    that is neither writing nor reading is stuck, while one that is reading and
    planning is still working the problem. Both are gated on
    ``STALL_MIN_TURN_DEPTH`` so a short exchange is never read as a stall.
    """
    producing = sig.recent_edit_count + sig.recent_write_count
    production_intensity = min(1.0, producing / PRODUCTION_SATURATION)

    spinning = 0.0
    exploring = 0.0
    if sig.turn_depth >= STALL_MIN_TURN_DEPTH and producing == 0:
        if sig.recent_read_count or sig.recent_plan_count:
            exploring = 1.0
        else:
            spinning = 1.0

    return CodingAgentDimensions(
        severity=sig.severity,
        spinning=spinning,
        exploring=exploring,
        production_intensity=production_intensity,
    )


def score_signals(sig: ToolSignals) -> tuple[float, str]:
    """Score ``sig`` into ``(score, decision_source)``.

    ``score`` is in (-1, 1): positive means the request is going badly and wants
    a stronger model, negative means it is finishing cleanly. The magnitude is
    calibrated so that no single dimension can clear ``DEFAULT_THRESHOLD`` alone
    — see ``SIGNAL_UNIT`` / ``SCORE_GAIN``.
    """
    if not sig.has_signal:
        return 0.0, SOURCE_NEUTRAL

    dims = dimensions_from_signals(sig)
    raw = SIGNAL_UNIT * (
        dims.severity / SEVERITY_HARD
        + dims.spinning
        + dims.exploring
        - dims.production_intensity
    )
    score = math.tanh(SCORE_GAIN * raw)
    return score, SOURCE_DIMENSIONS


def tier_adjustment(payload: dict, threshold: float = DEFAULT_THRESHOLD) -> tuple[int, str]:
    """Return ``(delta, decision_source)`` for a canonical request.

    ``delta`` is -1, 0 or +1 tier steps. Two hard rules run before the scorer:

    * **Escalate on critical severity or compaction.** A compaction summary
      replaces the transcript the signals are derived from, so an agent that was
      struggling would otherwise snap back to the weakest tier at exactly the
      moment it least wants to. Escalating on ``compacted`` is self-latching,
      since the summary stays in the prefix for the rest of the session.
    * **De-escalate on a clean finish.** Tests passing, recent writes, and no
      errors is the one state where a cheaper model is clearly safe.

    Only if neither fires does the tanh score decide, and only if it clears
    ``threshold`` in either direction.
    """
    sig = extract_tool_signals(payload)
    if not sig.has_signal:
        return 0, SOURCE_NEUTRAL

    if sig.severity >= SEVERITY_CRITICAL or sig.compacted:
        return 1, SOURCE_OVERRIDE

    if sig.tests_passed and sig.recent_edit_count + sig.recent_write_count > 0 and sig.severity == 0:
        return -1, SOURCE_TESTS_PASSED

    score, source = score_signals(sig)
    if score >= threshold:
        return 1, source
    if score <= -threshold:
        return -1, source
    return 0, SOURCE_AMBIGUOUS
