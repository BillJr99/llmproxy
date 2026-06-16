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
from flask import Response


def _load_server_with_config(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
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
            on_success=lambda pn, um: recorded.append((pn, um)),
        )
    assert recorded == [("p2", "m2")]  # committed only the healthy candidate
    assert resp.status_code == 200


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
        on_success=lambda pn, um: recorded.append((pn, um)),
    )
    assert recorded == []  # nothing ever committed
    assert resp.status_code == 502
    assert b"b" in resp.get_data()
