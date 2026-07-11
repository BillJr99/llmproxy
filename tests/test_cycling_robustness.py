"""Tests for virtual-model cycling robustness.

Covers the failure classes that trigger failover beyond a plain HTTP error:
200-with-error / empty body, transient same-candidate retry/backoff, and the
streaming first-chunk error peek.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import requests
from flask import Response


def _load_server_with_config(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def server(monkeypatch, tmp_path: Path):
    cfg = {
        "providers": {
            "p1": {"base_url": "http://p1.example/v1", "api_key": "k", "model_filter": None},
            "p2": {"base_url": "http://p2.example/v1", "api_key": "k", "model_filter": None},
        },
        "believed_free": [],
        "model_reasoning": {},
        "model_capabilities": {},
        "free_limits": {},
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return _load_server_with_config(monkeypatch, p)


def _json_resp(body: dict, status: int = 200) -> Response:
    return Response(json.dumps(body), status=status, content_type="application/json")


_OK = {"choices": [{"message": {"content": "ok"}}]}


class FakeStreamResp:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(self, status: int, chunks: list[bytes], content: bytes = b"", headers=None):
        self.status_code = status
        self._chunks = list(chunks)
        self.content = content
        self.headers = headers or {"Content-Type": "text/event-stream"}
        self.closed = False

    def iter_content(self, chunk_size=None):
        yield from self._chunks

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


# ── pure detector helpers ───────────────────────────────────────────────────

def test_is_transient_status(server):
    f = server._is_transient_status
    assert all(f(s) for s in (429, 500, 502, 503, 504))
    assert not any(f(s) for s in (400, 401, 403, 404, 422))


def test_response_unusable(server):
    f = server._response_unusable
    assert f(b"not json") is True
    assert f(json.dumps({"error": {"message": "x"}}).encode()) is True
    assert f(json.dumps({"choices": []}).encode()) is True
    assert f(json.dumps({"choices": None}).encode()) is True
    assert f(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()) is False
    # empty content but tool_calls present is still usable
    assert f(json.dumps(
        {"choices": [{"message": {"content": "", "tool_calls": [{"id": "1"}]}}]}
    ).encode()) is False
    # empty / whitespace-only content with no tool call is unusable — a reasoning
    # model that burned a tight max_tokens budget on thinking must fail over.
    assert f(json.dumps({"choices": [{"message": {"content": ""}}]}).encode()) is True
    assert f(json.dumps({"choices": [{"message": {"content": "   "}}]}).encode()) is True
    assert f(json.dumps({"choices": [{"message": {"content": None}}]}).encode()) is True
    assert f(json.dumps({"choices": [{"message": {}}]}).encode()) is True
    # a refusal is a usable answer
    assert f(json.dumps({"choices": [{"message": {"content": "", "refusal": "no"}}]}).encode()) is False
    # multimodal content parts count when any part has text
    assert f(json.dumps(
        {"choices": [{"message": {"content": [{"type": "text", "text": "hi"}]}}]}
    ).encode()) is False
    assert f(json.dumps(
        {"choices": [{"message": {"content": [{"type": "text", "text": ""}]}}]}
    ).encode()) is True
    # a second choice with real output keeps the body usable
    assert f(json.dumps(
        {"choices": [{"message": {"content": ""}}, {"message": {"content": "hi"}}]}
    ).encode()) is False


def test_sse_prefix_is_error(server):
    f = server._sse_prefix_is_error
    assert f(b'data: {"error":{"message":"x"}}\n\n') is True
    assert f(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n') is False
    assert f(b"data: [DONE]\n\n") is False
    assert f(b": keep-alive\n\n") is False


def test_peek_stream_preserves_first_chunk(server):
    first = b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
    done = b"data: [DONE]\n\n"
    resp = FakeStreamResp(200, [first, done])
    err, prefix, rest = server._peek_stream(resp)
    assert err is None
    assert prefix == first
    # the peeked prefix plus the remainder reconstructs the original stream
    assert prefix + b"".join(rest) == first + done


def test_peek_stream_detects_error(server):
    err_chunk = b'data: {"error":{"message":"boom"}}\n\n'
    resp = FakeStreamResp(200, [err_chunk])
    err, prefix, _rest = server._peek_stream(resp)
    assert err == err_chunk
    assert prefix == err_chunk


# ── non-streaming failover ──────────────────────────────────────────────────

def test_failover_on_200_error_body(server, monkeypatch):
    calls: list[str] = []

    def fake(endpoint, pn, cfg, payload, timeout):
        calls.append(payload["model"])
        if payload["model"] == "m1":
            return _json_resp({"error": {"message": "rate limited"}})
        return _json_resp(_OK)

    monkeypatch.setattr(server, "_proxy_request", fake)
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    resp = server._proxy_cycling_non_streaming("chat/completions", "t", candidates, {}, 5)
    assert calls == ["m1", "m2"]
    assert resp.status_code == 200
    assert b"ok" in resp.get_data()


def test_failover_on_empty_choices(server, monkeypatch):
    calls: list[str] = []

    def fake(endpoint, pn, cfg, payload, timeout):
        calls.append(payload["model"])
        return _json_resp({"choices": []}) if payload["model"] == "m1" else _json_resp(_OK)

    monkeypatch.setattr(server, "_proxy_request", fake)
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    resp = server._proxy_cycling_non_streaming("chat/completions", "t", candidates, {}, 5)
    assert calls == ["m1", "m2"]
    assert b"ok" in resp.get_data()


def test_failover_on_empty_content(server, monkeypatch):
    # A 200 whose only choice has empty content and no tool call (a reasoning
    # model that spent a tight max_tokens budget on thinking) must fail over to a
    # candidate that returns a real answer, not hand the client a blank reply.
    calls: list[str] = []

    def fake(endpoint, pn, cfg, payload, timeout):
        calls.append(payload["model"])
        if payload["model"] == "m1":
            return _json_resp({"choices": [{"message": {"content": ""}}]})
        return _json_resp(_OK)

    monkeypatch.setattr(server, "_proxy_request", fake)
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    resp = server._proxy_cycling_non_streaming("chat/completions", "t", candidates, {}, 5)
    assert calls == ["m1", "m2"]
    assert b"ok" in resp.get_data()


# ── auto budget escalation ──────────────────────────────────────────────────

def test_is_budget_truncated_empty(server):
    f = server._is_budget_truncated_empty
    # empty content cut off on length → recoverable with more budget
    assert f(json.dumps(
        {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
    ).encode()) is True
    assert f(json.dumps(
        {"choices": [{"message": {"content": "   "}, "finish_reason": "max_tokens"}]}
    ).encode()) is True
    # empty but NOT truncated (stopped normally) → not a budget problem
    assert f(json.dumps(
        {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
    ).encode()) is False
    # truncated but already has content → usable, leave it alone
    assert f(json.dumps(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "length"}]}
    ).encode()) is False
    # empty + tool call is usable output, not a starved body
    assert f(json.dumps(
        {"choices": [{"message": {"content": "", "tool_calls": [{"id": "1"}]},
                      "finish_reason": "length"}]}
    ).encode()) is False
    assert f(b"not json") is False


def test_bumped_budget(server):
    f = server._bumped_budget
    factor = server._BUDGET_BUMP_FACTOR
    ceiling = server._BUDGET_BUMP_CEILING
    assert f({"max_tokens": 8})["max_tokens"] == 8 * factor
    assert f({"max_completion_tokens": 8})["max_completion_tokens"] == 8 * factor
    # clamps at the ceiling
    assert f({"max_tokens": ceiling // 2 + 1})["max_tokens"] == ceiling
    # already at/over the ceiling → no bump
    assert f({"max_tokens": ceiling}) is None
    # no budget field / uncapped request → nothing to bump
    assert f({}) is None
    assert f({"max_tokens": 0}) is None
    # a bool is not a real budget (True == 1 in Python) — must be ignored
    assert f({"max_tokens": True}) is None


def test_budget_escalation_recovers_answer(server, monkeypatch):
    # A reasoning model returns an empty, length-truncated body at max_tokens=8,
    # then answers once the budget is bumped — all on the SAME candidate, with no
    # failover to a weaker model.
    calls: list[dict] = []
    starved = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}

    def fake(endpoint, pn, cfg, payload, timeout):
        calls.append(dict(payload))
        if payload.get("max_tokens", 0) <= 8:
            return _json_resp(starved)
        return _json_resp(_OK)

    monkeypatch.setattr(server, "_proxy_request", fake)
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    resp = server._proxy_cycling_non_streaming(
        "chat/completions", "t", candidates, {"max_tokens": 8}, 5
    )
    # Only m1 was tried; the second call carried the bumped budget.
    assert [c["model"] for c in calls] == ["m1", "m1"]
    assert calls[1]["max_tokens"] == 8 * server._BUDGET_BUMP_FACTOR
    assert b"ok" in resp.get_data()


def test_budget_escalation_is_bounded_then_fails_over(server, monkeypatch):
    # A model that stays empty no matter the budget must not loop forever: the
    # escalation is capped, then the loop fails over to the next candidate.
    calls: list[dict] = []
    starved = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}

    def fake(endpoint, pn, cfg, payload, timeout):
        calls.append(dict(payload))
        return _json_resp(starved) if payload["model"] == "m1" else _json_resp(_OK)

    monkeypatch.setattr(server, "_proxy_request", fake)
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    resp = server._proxy_cycling_non_streaming(
        "chat/completions", "t", candidates, {"max_tokens": 8}, 5
    )
    m1_calls = [c for c in calls if c["model"] == "m1"]
    # one initial + at most _BUDGET_BUMP_MAX_RETRIES bumped attempts, then failover
    assert len(m1_calls) == 1 + server._BUDGET_BUMP_MAX_RETRIES
    assert calls[-1]["model"] == "m2"
    assert b"ok" in resp.get_data()


def test_no_budget_escalation_without_max_tokens(server, monkeypatch):
    # An uncapped request that comes back empty+truncated can't be bumped (no
    # budget field), so it fails over immediately without extra same-candidate hits.
    calls: list[dict] = []
    starved = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}

    def fake(endpoint, pn, cfg, payload, timeout):
        calls.append(dict(payload))
        return _json_resp(starved) if payload["model"] == "m1" else _json_resp(_OK)

    monkeypatch.setattr(server, "_proxy_request", fake)
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    resp = server._proxy_cycling_non_streaming("chat/completions", "t", candidates, {}, 5)
    assert [c["model"] for c in calls] == ["m1", "m2"]
    assert b"ok" in resp.get_data()


def test_transient_fails_over_immediately_when_alternatives_remain(server, monkeypatch):
    # A 429/5xx with another candidate available must fail over at once (no
    # same-candidate retry) so a rate-limited model never stalls the pipeline.
    calls: list[str] = []
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    def fake(endpoint, pn, cfg, payload, timeout):
        calls.append(payload["model"])
        if payload["model"] == "m1":
            return _json_resp({"error": "overloaded"}, status=503)
        return _json_resp(_OK)

    monkeypatch.setattr(server, "_proxy_request", fake)
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    resp = server._proxy_cycling_non_streaming("chat/completions", "t", candidates, {}, 5)
    assert calls == ["m1", "m2"]  # m1 tried once, immediately failed over to m2
    assert resp.status_code == 200


def test_transient_retry_only_on_last_candidate(server, monkeypatch):
    # The last candidate has no fallback, so it gets the same-candidate retries.
    calls: list[str] = []
    state = {"first": True}
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)

    def fake(endpoint, pn, cfg, payload, timeout):
        calls.append(payload["model"])
        # m1 (not last) always 503; m2 (last) 503 once then succeeds.
        if payload["model"] == "m1":
            return _json_resp({"error": "overloaded"}, status=503)
        if payload["model"] == "m2" and state["first"]:
            state["first"] = False
            return _json_resp({"error": "overloaded"}, status=503)
        return _json_resp(_OK)

    monkeypatch.setattr(server, "_proxy_request", fake)
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    resp = server._proxy_cycling_non_streaming("chat/completions", "t", candidates, {}, 5)
    assert calls == ["m1", "m2", "m2"]  # m1 once, then m2 retried on itself
    assert resp.status_code == 200


def test_non_transient_does_not_retry(server, monkeypatch):
    calls: list[str] = []

    def fake(endpoint, pn, cfg, payload, timeout):
        calls.append(payload["model"])
        return _json_resp({"error": "bad request"}, status=400) if payload["model"] == "m1" else _json_resp(_OK)

    monkeypatch.setattr(server, "_proxy_request", fake)
    candidates = [("p1", {}, "m1"), ("p2", {}, "m2")]
    resp = server._proxy_cycling_non_streaming("chat/completions", "t", candidates, {}, 5)
    assert calls == ["m1", "m2"]  # m1 tried once (no retry on 400), failed over
    assert resp.status_code == 200


# ── streaming failover ──────────────────────────────────────────────────────

def test_streaming_failover_on_first_chunk_error(server, monkeypatch):
    responses = iter([
        FakeStreamResp(200, [b'data: {"error":{"message":"boom"}}\n\n']),
        FakeStreamResp(200, [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]),
    ])
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: next(responses))
    recorded: list[tuple[str, str]] = []
    candidates = [
        ("p1", {"base_url": "http://p1/v1", "api_key": "k"}, "m1"),
        ("p2", {"base_url": "http://p2/v1", "api_key": "k"}, "m2"),
    ]
    # A request context is needed because committing the stream uses
    # flask.stream_with_context, which validates the context eagerly.
    with server.app.test_request_context():
        resp = server._proxy_cycling_streaming(
            "chat/completions", "t", candidates, {"messages": []}, 5,
            on_success=lambda pn, um, *a: recorded.append((pn, um)),
        )
    assert recorded == [("p2", "m2")]  # committed only the healthy candidate
    assert resp.status_code == 200


def test_streaming_connect_timeout_causes_failover(server, monkeypatch):
    # A Timeout on the initial requests.post() must fail over immediately to
    # the next candidate; the client should never see the timeout error.
    calls: list[str] = []
    ok_resp = FakeStreamResp(200, [b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', b"data: [DONE]\n\n"])

    def fake_post(*a, **k):
        if not calls:
            calls.append("p1")
            raise requests.exceptions.Timeout("connect timed out")
        calls.append("p2")
        return ok_resp

    monkeypatch.setattr(server.requests, "post", fake_post)
    recorded: list[tuple[str, str]] = []
    candidates = [
        ("p1", {"base_url": "http://p1/v1", "api_key": "k"}, "m1"),
        ("p2", {"base_url": "http://p2/v1", "api_key": "k"}, "m2"),
    ]
    with server.app.test_request_context():
        resp = server._proxy_cycling_streaming(
            "chat/completions", "t", candidates, {"messages": []}, 5,
            on_success=lambda pn, um, *a: recorded.append((pn, um)),
        )
    assert calls == ["p1", "p2"]
    assert recorded == [("p2", "m2")]
    assert resp.status_code == 200


def test_streaming_connect_error_as_connectionerror_causes_failover(server, monkeypatch):
    # A ConnectionError (e.g. ReadTimeout wrapped via urllib3 MaxRetryError) on
    # the initial requests.post() must also fail over — not surface to the client.
    calls: list[str] = []
    ok_resp = FakeStreamResp(200, [b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', b"data: [DONE]\n\n"])

    def fake_post(*a, **k):
        if not calls:
            calls.append("p1")
            raise requests.exceptions.ConnectionError(
                "HTTPSConnectionPool(host='p1.example', port=443): Read timed out."
            )
        calls.append("p2")
        return ok_resp

    monkeypatch.setattr(server.requests, "post", fake_post)
    recorded: list[tuple[str, str]] = []
    candidates = [
        ("p1", {"base_url": "http://p1/v1", "api_key": "k"}, "m1"),
        ("p2", {"base_url": "http://p2/v1", "api_key": "k"}, "m2"),
    ]
    with server.app.test_request_context():
        resp = server._proxy_cycling_streaming(
            "chat/completions", "t", candidates, {"messages": []}, 5,
            on_success=lambda pn, um, *a: recorded.append((pn, um)),
        )
    assert calls == ["p1", "p2"]
    assert recorded == [("p2", "m2")]
    assert resp.status_code == 200


def test_streaming_mid_stream_connection_error_clean_message(server, monkeypatch):
    # A ConnectionError (ReadTimeout-as-ConnectionError) that fires mid-stream
    # after the peek must yield the clean "timed out" message, not the raw
    # socket exception string (which leaks implementation details to the client).
    class HangingStreamResp(FakeStreamResp):
        def iter_content(self, chunk_size=None):
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            raise requests.exceptions.ConnectionError(
                "HTTPSConnectionPool(host='p1.example', port=443): Read timed out."
            )

    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: HangingStreamResp(200, []))
    candidates = [("p1", {"base_url": "http://p1/v1", "api_key": "k"}, "m1")]
    with server.app.test_request_context():
        resp = server._proxy_cycling_streaming(
            "chat/completions", "t", candidates, {"messages": []}, 5,
        )
    body = b"".join(resp.response)
    assert b"Upstream stream timed out." in body
    assert b"HTTPSConnectionPool" not in body


def test_streaming_all_first_chunk_errors_returns_last(server, monkeypatch):
    responses = iter([
        FakeStreamResp(200, [b'data: {"error":{"message":"a"}}\n\n']),
        FakeStreamResp(200, [b'data: {"error":{"message":"b"}}\n\n']),
    ])
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: next(responses))
    recorded: list = []
    candidates = [
        ("p1", {"base_url": "http://p1/v1", "api_key": "k"}, "m1"),
        ("p2", {"base_url": "http://p2/v1", "api_key": "k"}, "m2"),
    ]
    resp = server._proxy_cycling_streaming(
        "chat/completions", "t", candidates, {"messages": []}, 5,
        on_success=lambda pn, um, *a: recorded.append((pn, um)),
    )
    assert recorded == []  # nothing ever committed
    assert resp.status_code == 502
    assert b"b" in resp.get_data()


# ── multiple accounts per provider ───────────────────────────────────────────

def _multi_acct_cfg():
    # priority strategy keeps account order deterministic (a before b).
    return {
        "base_url": "http://p1/v1",
        "account_strategy": "priority",
        "accounts": [
            {"key": "k1", "label": "a", "priority": 1},
            {"key": "k2", "label": "b", "priority": 2},
        ],
    }


def test_expand_accounts_single_is_noop(server):
    cands = [("p1", {"base_url": "http://p1/v1", "api_key": "k"}, "m1")]
    out = server._expand_accounts(cands)
    assert out == cands  # lone credential — cfg untouched, no _account_id


def test_expand_accounts_fans_out_adjacent(server):
    out = server._expand_accounts([("p1", _multi_acct_cfg(), "m1")])
    assert [(pn, c["_account_id"], um) for pn, c, um in out] == [
        ("p1", "a", "m1"), ("p1", "b", "m1"),
    ]
    # Each bound cfg resolves its own key through the existing accessor.
    assert server.provider_api_key(out[0][1]) == "k1"
    assert server.provider_api_key(out[1][1]) == "k2"


def test_quota_rotates_to_next_account_then_is_sticky(server, monkeypatch):
    server._reset_usage()
    pc = _multi_acct_cfg()
    seen: list = []

    def fake(endpoint, pn, cfg, payload, timeout):
        seen.append(cfg.get("_account_id"))
        if cfg.get("_account_id") == "a":   # account a is rate-limited
            return _json_resp({"error": {"type": "rate_limit_exceeded"}}, status=429)
        return _json_resp(_OK)

    monkeypatch.setattr(server, "_proxy_request", fake)
    recs: list = []

    order1 = server._expand_accounts([("p1", pc, "m1")])
    assert [c["_account_id"] for _pn, c, _um in order1] == ["a", "b"]
    resp = server._proxy_cycling_non_streaming(
        "chat/completions", "t", order1, {}, 5,
        on_success=lambda pn, um, body=None, acct=None: recs.append((pn, um, acct)),
    )
    assert resp.status_code == 200
    assert seen == ["a", "b"]                 # rotated account a -> b, same model
    assert recs == [("p1", "m1", "b")]        # usage recorded under account b
    # Account a is cooled; the next request deprioritizes it (sticky rotation).
    assert server._is_candidate_saturated("p1", "m1", "a") is True
    assert server._is_candidate_saturated("p1", "m1", "b") is False
    order2 = server._expand_accounts([("p1", pc, "m1")])
    assert [c["_account_id"] for _pn, c, _um in order2] == ["b", "a"]


def test_all_accounts_saturated_still_reachable(server, monkeypatch):
    server._reset_usage()
    pc = _multi_acct_cfg()
    server._mark_saturated(server._usage_key("p1", "m1", "a"), retry_after="60")
    server._mark_saturated(server._usage_key("p1", "m1", "b"), retry_after="60")
    out = server._expand_accounts([("p1", pc, "m1")])
    # Both cooling — order is stable (priority) and neither is dropped.
    assert {c["_account_id"] for _pn, c, _um in out} == {"a", "b"}
