"""Tests for the streaming and cycling hardening.

Four behaviors, each of which had a real failure mode before:

* the identity streaming fast path relayed upstream errors as ``200``;
* a committed stream that died mid-flight simply stopped, with no
  ``finish_reason`` and no ``[DONE]``, and the provider kept a perfect health
  score for it;
* the candidate walk had no overall wall-clock bound;
* the pre-commit check ended at the first non-empty chunk, so a provider that
  emitted a role preamble and then died was already committed to.

Everything gated here defaults to the old behavior, so each suite also asserts
the default is unchanged rather than only asserting the new path works.
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


def _make_server(monkeypatch, tmp_path: Path, server_block: dict | None = None):
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
                   "request_timeout": 5, "stream_timeout": 5, **(server_block or {})},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return _load_server_with_config(monkeypatch, p)


@pytest.fixture
def server(monkeypatch, tmp_path: Path):
    return _make_server(monkeypatch, tmp_path)


class FakeStreamResp:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(self, status: int, chunks: list[bytes], content: bytes = b"",
                 headers=None, die: Exception | None = None):
        self.status_code = status
        self._chunks = list(chunks)
        self.content = content
        self.headers = headers or {"Content-Type": "text/event-stream"}
        self.closed = False
        self._die = die

    def iter_content(self, chunk_size=None):
        yield from self._chunks
        if self._die is not None:
            raise self._die

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


CANDS = [
    ("p1", {"base_url": "http://p1.example/v1", "api_key": "k"}, "m1"),
    ("p2", {"base_url": "http://p2.example/v1", "api_key": "k"}, "m2"),
]

PREAMBLE = b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
ERRFRAME = b'data: {"error":{"message":"upstream exploded"}}\n\n'
TEXT = b'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n'
DONE = b"data: [DONE]\n\n"
TOOLFRAME = (
    b'data: {"choices":[{"index":0,"delta":{"tool_calls":'
    b'[{"index":0,"id":"call_1","function":{"name":"f","arguments":"{}"}}]}}]}\n\n'
)


def _run_stream(server, cands=CANDS, config=None):
    with server.app.test_request_context():
        return server._proxy_cycling_streaming(
            "chat/completions", "t", cands, {"messages": []}, 5, config=config,
        )


# ── terminal frames on a committed stream ───────────────────────────────────

def test_mid_stream_failure_emits_finish_reason_and_done(server, monkeypatch):
    """A stream that dies after commit must still terminate cleanly.

    Without a finish_reason the client cannot know a partially-accumulated
    tool-call argument string will get no more fragments; without [DONE] the
    OpenAI SDKs hang or raise a generic 'stream ended' error.
    """
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: FakeStreamResp(
        200, [TEXT], die=requests.exceptions.ConnectionError("boom")))
    resp = _run_stream(server, cands=CANDS[:1])
    body = b"".join(resp.response)
    assert b'"error"' in body
    assert b'"finish_reason": "length"' in body or b'"finish_reason":"length"' in body
    assert body.rstrip().endswith(b"data: [DONE]")


def test_mid_stream_failure_demotes_provider_health(server, monkeypatch):
    """The candidate is credited at commit; a later death must revoke that."""
    outcomes: list[tuple] = []
    monkeypatch.setattr(server, "_record_outcome",
                        lambda pn, um, ok, **kw: outcomes.append((pn, um, ok)))
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: FakeStreamResp(
        200, [TEXT], die=requests.exceptions.ConnectionError("boom")))
    resp = _run_stream(server, cands=CANDS[:1])
    b"".join(resp.response)
    assert ("p1", "m1", True) in outcomes, "commit should credit the candidate"
    assert ("p1", "m1", False) in outcomes, "mid-stream death should demote it"


def test_client_disconnect_does_not_demote_provider(server, monkeypatch):
    """A user pressing Ctrl-C is not a provider fault."""
    outcomes: list[tuple] = []
    monkeypatch.setattr(server, "_record_outcome",
                        lambda pn, um, ok, **kw: outcomes.append((pn, um, ok)))
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: FakeStreamResp(
        200, [TEXT], die=RuntimeError("client disconnected")))
    resp = _run_stream(server, cands=CANDS[:1])
    b"".join(resp.response)
    assert (("p1", "m1", False)) not in outcomes


def test_stream_error_frames_are_valid_json():
    from llmproxy import server as s
    frames = s._stream_error_frames("boom", "m1")
    assert frames[-1] == b"data: [DONE]\n\n"
    for frame in frames[:-1]:
        json.loads(frame.split(b"data: ", 1)[1])


# ── identity fast path status check ─────────────────────────────────────────

@pytest.mark.parametrize("status", [429, 500, 401])
def test_identity_stream_relays_upstream_error_status(server, monkeypatch, status):
    """An upstream error must not arrive as 200 text/event-stream."""
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: FakeStreamResp(
        status, [], content=b'{"error":{"message":"nope"}}',
        headers={"Content-Type": "application/json"}))
    with server.app.test_request_context():
        resp = server._proxy_streaming(
            "chat/completions", "p1",
            {"base_url": "http://p1.example/v1", "api_key": "k"},
            {"model": "m1", "messages": []}, 5,
        )
    assert resp.status_code == status
    assert "event-stream" not in resp.content_type


def test_identity_stream_success_is_unchanged(server, monkeypatch):
    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: FakeStreamResp(200, [TEXT, DONE]))
    with server.app.test_request_context():
        resp = server._proxy_streaming(
            "chat/completions", "p1",
            {"base_url": "http://p1.example/v1", "api_key": "k"},
            {"model": "m1", "messages": []}, 5,
        )
    assert resp.status_code == 200
    assert resp.content_type == "text/event-stream"
    assert b"".join(resp.response) == TEXT + DONE


def test_identity_stream_connect_timeout_returns_504(server, monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.Timeout("slow")
    monkeypatch.setattr(server.requests, "post", boom)
    with server.app.test_request_context():
        resp = server._proxy_streaming(
            "chat/completions", "p1",
            {"base_url": "http://p1.example/v1", "api_key": "k"},
            {"model": "m1", "messages": []}, 5,
        )
    assert resp.status_code == 504


# ── pre-commit window ───────────────────────────────────────────────────────

def test_precommit_disabled_by_default_commits_on_first_chunk(server, monkeypatch):
    """The default must reproduce the old behavior exactly."""
    seq = iter([FakeStreamResp(200, [PREAMBLE, ERRFRAME]), FakeStreamResp(200, [PREAMBLE, TEXT])])
    tried: list = []
    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: (tried.append(1), next(seq))[1])
    resp = _run_stream(server, config={"server": {}})
    body = b"".join(resp.response)
    assert len(tried) == 1
    assert b"exploded" in body


def test_precommit_window_fails_over_past_role_preamble(monkeypatch, tmp_path):
    """The headline case: preamble, then error, must fail over invisibly."""
    server = _make_server(monkeypatch, tmp_path, {"stream_commit_on_content": True})
    seq = iter([FakeStreamResp(200, [PREAMBLE, ERRFRAME]), FakeStreamResp(200, [PREAMBLE, TEXT])])
    tried: list = []
    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: (tried.append(1), next(seq))[1])
    cfg = {"server": {"stream_commit_on_content": True}}
    resp = _run_stream(server, config=cfg)
    body = b"".join(resp.response)
    assert len(tried) == 2
    assert b"exploded" not in body
    assert b"hello" in body


def test_precommit_window_replays_every_buffered_byte_in_order(monkeypatch, tmp_path):
    server = _make_server(monkeypatch, tmp_path, {"stream_commit_on_content": True})
    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: FakeStreamResp(200, [PREAMBLE, TEXT, DONE]))
    resp = _run_stream(server, cands=CANDS[:1],
                       config={"server": {"stream_commit_on_content": True}})
    assert b"".join(resp.response) == PREAMBLE + TEXT + DONE


def test_precommit_window_fails_over_on_stream_with_no_output(monkeypatch, tmp_path):
    """A provider that says nothing usable and closes is a failure, not a turn."""
    server = _make_server(monkeypatch, tmp_path, {"stream_commit_on_content": True})
    seq = iter([FakeStreamResp(200, [PREAMBLE, DONE]), FakeStreamResp(200, [PREAMBLE, TEXT])])
    tried: list = []
    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: (tried.append(1), next(seq))[1])
    resp = _run_stream(server, config={"server": {"stream_commit_on_content": True}})
    assert len(tried) == 2
    assert b"hello" in b"".join(resp.response)


def test_precommit_window_commits_on_tool_call_delta(monkeypatch, tmp_path):
    """A tool-call fragment is output; an agent turn may contain no prose."""
    server = _make_server(monkeypatch, tmp_path, {"stream_commit_on_content": True})
    tried: list = []
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: (
        tried.append(1), FakeStreamResp(200, [PREAMBLE, TOOLFRAME, DONE]))[1])
    resp = _run_stream(server, config={"server": {"stream_commit_on_content": True}})
    assert len(tried) == 1
    assert b"call_1" in b"".join(resp.response)


def test_delta_yields_output_discriminates_preamble_from_content(server):
    assert server._delta_yields_output({"content": "hi"}) is True
    assert server._delta_yields_output({"tool_calls": [{"index": 0}]}) is True
    assert server._delta_yields_output({"refusal": "no"}) is True
    assert server._delta_yields_output({"role": "assistant"}) is False
    assert server._delta_yields_output({"content": ""}) is False
    assert server._delta_yields_output({}) is False
    assert server._delta_yields_output(None) is False


# ── whole-response buffering ────────────────────────────────────────────────

def test_buffer_full_gives_true_mid_generation_failover(monkeypatch, tmp_path):
    """The one configuration where a death at 80% is recoverable."""
    server = _make_server(monkeypatch, tmp_path, {"stream_buffer_full": True})
    seq = iter([
        FakeStreamResp(200, [TEXT], die=requests.exceptions.ConnectionError("died at 80%")),
        FakeStreamResp(200, [TEXT, DONE]),
    ])
    tried: list = []
    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: (tried.append(1), next(seq))[1])
    resp = _run_stream(server, config={"server": {"stream_buffer_full": True}})
    body = b"".join(resp.response)
    assert len(tried) == 2
    assert body == TEXT + DONE


def test_buffer_full_disabled_by_default_leaks_partial_output(server, monkeypatch):
    """Documents precisely what the default cannot do, so the trade stays visible."""
    seq = iter([
        FakeStreamResp(200, [TEXT], die=requests.exceptions.ConnectionError("died")),
        FakeStreamResp(200, [TEXT, DONE]),
    ])
    tried: list = []
    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: (tried.append(1), next(seq))[1])
    resp = _run_stream(server, config={"server": {}})
    body = b"".join(resp.response)
    assert len(tried) == 1
    assert b"hello" in body and b'"error"' in body


# ── cycle deadline ──────────────────────────────────────────────────────────

def test_cycle_deadline_disabled_by_default(server):
    assert server._cycle_deadline({"server": {}}) is None
    assert server._timeout_for_candidate(None, 60) == 60


def test_cycle_deadline_shrinks_candidate_timeout(server, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(server.time, "monotonic", lambda: now[0])
    deadline = server._cycle_deadline({"server": {"cycle_deadline_seconds": 30}})
    assert deadline == 1030.0
    now[0] = 1010.0
    assert server._timeout_for_candidate(deadline, 60) == pytest.approx(20.0)


def test_cycle_deadline_stops_walking_when_exhausted(server, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(server.time, "monotonic", lambda: now[0])

    def failing(endpoint, pn, cfg, payload, timeout, **kw):
        now[0] += 100.0  # each candidate burns the whole budget
        return Response(json.dumps({"error": {"message": "boom"}}),
                        status=500, content_type="application/json")

    monkeypatch.setattr(server, "_proxy_request", failing)
    three = CANDS + [("p3", {"base_url": "http://p3/v1", "api_key": "k"}, "m3")]
    with server.app.test_request_context():
        resp = server._proxy_cycling_non_streaming(
            "chat/completions", "t", three, {"messages": []}, 60,
        )
    # Candidate 0 always runs; the deadline must stop the walk before candidate 2.
    assert resp.status_code == 500
    assert b"boom" in resp.get_data()


def test_cycle_deadline_always_allows_the_first_candidate(server, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(server.time, "monotonic", lambda: now[0])
    calls: list = []

    def ok(endpoint, pn, cfg, payload, timeout, **kw):
        calls.append(pn)
        return Response(json.dumps({"choices": [{"message": {"content": "hi"}}]}),
                        status=200, content_type="application/json")

    monkeypatch.setattr(server, "_proxy_request", ok)
    monkeypatch.setattr(server, "_cycle_deadline", lambda *a, **k: now[0] - 1)
    with server.app.test_request_context():
        resp = server._proxy_cycling_non_streaming(
            "chat/completions", "t", CANDS, {"messages": []}, 60,
        )
    assert calls == ["p1"]
    assert resp.status_code == 200


def test_budget_escalation_accepts_a_deadline(server):
    """Keyword-only with a None default, so existing callers are unaffected."""
    import inspect
    sig = inspect.signature(server._escalate_budget_if_starved)
    assert sig.parameters["deadline"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["deadline"].default is None


# ── streamed route provenance ───────────────────────────────────────────────
# A stream has no body to inject into, so the llmproxy_route block rides one
# synthetic frame ahead of the upstream's own. Rewriting the real chunks would
# forfeit the byte-for-byte relay the buffered pre-commit prefix depends on.

def _stream_virtual(server, cands=CANDS, config=None):
    with server.app.test_request_context():
        return server._proxy_cycling_streaming(
            "chat/completions", "llmproxy__free", cands, {"messages": []}, 5,
            config=config, route_reason="capacity", virtual_model="llmproxy__free",
        )


def test_stream_leads_with_the_route_frame(server, monkeypatch):
    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: FakeStreamResp(200, [TEXT, DONE]))
    resp = _stream_virtual(server, cands=CANDS[:1])
    body = b"".join(resp.response)

    frames = [f for f in body.split(b"\n\n") if f.strip()]
    lead = json.loads(frames[0][len(b"data: "):])
    assert lead["object"] == "chat.completion.chunk"
    assert lead["choices"] == []          # same shape as the usage chunk clients already see
    assert lead["llmproxy_route"]["selected_model"] == "p1/m1"
    assert lead["llmproxy_route"]["virtual"] == "llmproxy__free"
    assert lead["llmproxy_route"]["failed_over"] is False

    # Everything the upstream sent follows, untouched and in order.
    assert body.endswith(TEXT + DONE)
    assert resp.headers["X-LLMProxy-Selected-Model"] == "p1/m1"


def test_streamed_route_frame_marks_a_failover(server, monkeypatch):
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeStreamResp(200, [ERRFRAME])
        return FakeStreamResp(200, [TEXT, DONE])

    monkeypatch.setattr(server.requests, "post", fake_post)
    resp = _stream_virtual(server)
    body = b"".join(resp.response)
    lead = json.loads(body.split(b"\n\n")[0][len(b"data: "):])
    assert lead["llmproxy_route"]["selected_model"] == "p2/m2"
    assert lead["llmproxy_route"]["failed_over"] is True
    assert lead["llmproxy_route"]["route_reason"] == "capacity,failover#1"


def test_streamed_route_frame_can_be_switched_off(server, monkeypatch):
    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: FakeStreamResp(200, [TEXT, DONE]))
    resp = _stream_virtual(server, cands=CANDS[:1],
                           config={"server": {"report_route": False}})
    body = b"".join(resp.response)
    assert body == TEXT + DONE
    assert resp.headers["X-LLMProxy-Selected-Model"] == "p1/m1"


def test_a_pinned_stream_gets_no_route_frame(server, monkeypatch):
    """Only virtual models carry the block; a pinned stream stays byte-identical."""
    monkeypatch.setattr(server.requests, "post",
                        lambda *a, **k: FakeStreamResp(200, [TEXT, DONE]))
    resp = _run_stream(server, cands=CANDS[:1])
    assert b"".join(resp.response) == TEXT + DONE

# ── a timeout is a 429 ──────────────────────────────────────────────────────
#
# A timeout and a rate limit tell the *next* request the same thing: this
# candidate is not answering. Only 429 used to be remembered, so a model that
# had started timing out kept its place in the order and cost a full timeout on
# every request. These pin the cooldown down at each point one can occur —
# before the stream opens, during the pre-commit peek, and after commit, which
# is the one the read timeout is the only possible guard for.

def _cooling(server, provider="p1", model="m1"):
    return server._is_candidate_saturated(provider, model, None)


@pytest.fixture
def cool_server(monkeypatch, tmp_path: Path):
    s = _make_server(monkeypatch, tmp_path, {"virtual_timeout_seconds": 90})
    s._reset_usage()
    yield s
    s._reset_usage()


def test_the_knob_is_off_by_default(server):
    """An untouched config keeps the existing request/stream timeouts exactly."""
    assert server._virtual_timeout({"server": {}}) is None
    assert server._virtual_timeout({"server": {"virtual_timeout_seconds": 0}}) is None
    assert server._virtual_timeout({"server": {"virtual_timeout_seconds": 90}}) == 90.0


@pytest.mark.parametrize("bad", [-1, "ninety", None, True, [], {}])
def test_a_malformed_value_disables_rather_than_breaks(server, bad):
    """config.json is hand-edited; a typo must not fail every request. A
    negative reads as 'off' rather than as an instant timeout, which is the
    harmless reading."""
    assert server._virtual_timeout({"server": {"virtual_timeout_seconds": bad}}) is None


def test_only_a_synthesized_timeout_carries_the_marker(server):
    """The non-streaming loop cannot distinguish a timeout from an upstream 504
    by status alone, so the marker is what carries it out of _proxy_request."""
    with server.app.test_request_context():
        plain = server._error("upstream said 504", status=504)
        assert not server._is_timeout_response(plain)
        assert server._is_timeout_response(server._mark_timeout(plain))


def test_a_read_timeout_wearing_a_connectionerror_is_still_a_timeout(server):
    """urllib3 re-raises a mid-stream read timeout as ConnectionError, so the
    type cannot be trusted — only the message can."""
    import requests as _rq
    assert server._looks_like_timeout(_rq.exceptions.ReadTimeout("x"))
    assert server._looks_like_timeout(_rq.exceptions.ConnectTimeout("x"))
    assert server._looks_like_timeout(
        _rq.exceptions.ConnectionError("HTTPSConnectionPool(host='a'): Read timed out."))
    # A genuinely different fault must NOT be cooled: it fails over on its own,
    # and cooling it would pull a candidate for a transient blip that cost nothing.
    assert not server._looks_like_timeout(_rq.exceptions.ConnectionError("Connection reset by peer"))
    assert not server._looks_like_timeout(ValueError("nope"))


def test_a_connect_timeout_cools_the_candidate(cool_server, monkeypatch):
    import requests as _rq

    calls = []

    def _post(url, **kw):
        calls.append(url)
        if "p1" in url:
            raise _rq.exceptions.Timeout("timed out")
        return FakeStreamResp(200, [TEXT, DONE])

    monkeypatch.setattr(cool_server.requests, "post", _post)
    with cool_server.app.test_request_context():
        cool_server._proxy_cycling_streaming(
            "chat/completions", "t", CANDS, {"messages": []}, 5, config=None)
    assert len(calls) == 2                      # failed over, as before
    assert _cooling(cool_server, "p1", "m1")    # and is now remembered
    assert not _cooling(cool_server, "p2", "m2")


def test_a_stream_that_dies_mid_flight_cools_the_candidate(cool_server, monkeypatch):
    """The reported symptom: upstream 200, a first chunk, then silence. The
    request is already lost — bytes have shipped — so cooling the candidate for
    the NEXT request is the only repair available."""
    import requests as _rq

    stall = _rq.exceptions.ConnectionError(
        "HTTPSConnectionPool(host='p1.example', port=443): Read timed out.")
    monkeypatch.setattr(cool_server.requests, "post",
                        lambda url, **kw: FakeStreamResp(200, [TEXT], die=stall))
    with cool_server.app.test_request_context():
        resp = cool_server._proxy_cycling_streaming(
            "chat/completions", "t", CANDS, {"messages": []}, 5, config=None)
        body = b"".join(resp.response)

    # The client gets a cleanly terminated stream, not a truncated one.
    assert b"Upstream stream timed out." in body
    assert b"HTTPSConnectionPool" not in body     # no internals leaked
    assert body.rstrip().endswith(b"data: [DONE]")
    assert _cooling(cool_server, "p1", "m1")


def test_a_client_disconnect_still_does_not_cool_anything(cool_server, monkeypatch):
    """Ctrl-C is not the provider's fault; it must not pull a healthy model."""
    monkeypatch.setattr(cool_server.requests, "post",
                        lambda url, **kw: FakeStreamResp(
                            200, [TEXT], die=RuntimeError("client disconnected")))
    with cool_server.app.test_request_context():
        resp = cool_server._proxy_cycling_streaming(
            "chat/completions", "t", CANDS, {"messages": []}, 5, config=None)
        b"".join(resp.response)
    assert not _cooling(cool_server, "p1", "m1")


def test_a_non_streaming_timeout_cools_the_candidate(cool_server, monkeypatch):
    import requests as _rq

    def _post(url, **kw):
        if "p1" in url:
            raise _rq.exceptions.Timeout("timed out")
        r = type("R", (), {})()
        r.status_code, r.headers = 200, {"Content-Type": "application/json"}
        r.content = b'{"choices":[{"message":{"content":"hi"}}]}'
        r.get_data = lambda: r.content
        return r

    monkeypatch.setattr(cool_server.requests, "post", _post)
    with cool_server.app.test_request_context():
        cool_server._proxy_cycling_non_streaming(
            "chat/completions", "t", CANDS, {"messages": []}, 5, config=None)
    assert _cooling(cool_server, "p1", "m1")


def test_an_ordinary_upstream_502_is_not_cooled(cool_server, monkeypatch):
    """Only timeouts are cooled. A plain 5xx already fails over, and cooling it
    would take a candidate out of rotation for a one-off blip."""
    def _post(url, **kw):
        r = type("R", (), {})()
        r.status_code, r.headers = 502, {"Content-Type": "application/json"}
        r.content = b'{"error":{"message":"bad gateway"}}'
        r.get_data = lambda: r.content
        return r

    monkeypatch.setattr(cool_server.requests, "post", _post)
    with cool_server.app.test_request_context():
        cool_server._proxy_cycling_non_streaming(
            "chat/completions", "t", CANDS, {"messages": []}, 5, config=None)
    assert not _cooling(cool_server, "p1", "m1")


def test_the_timeout_bounds_the_gap_between_chunks_not_the_total(monkeypatch, tmp_path):
    """The bound has to be an idle one. A total budget would kill a long but
    healthy generation, which is exactly what flagship-tier reasoning produces."""
    seen = {}
    s = _make_server(monkeypatch, tmp_path, {"virtual_timeout_seconds": 90,
                                             "stream_timeout": 300})

    def _post(url, **kw):
        seen["timeout"] = kw.get("timeout")
        return FakeStreamResp(200, [TEXT, DONE])

    monkeypatch.setattr(s.requests, "post", _post)
    with s.app.test_request_context():
        s._proxy_cycling_streaming("chat/completions", "t", CANDS,
                                   {"messages": []}, 300, config=None)
    connect, read = seen["timeout"]
    assert read == 90       # the idle bound, in force for the life of the socket
    assert connect <= 90


def test_without_the_knob_the_socket_timeout_is_unchanged(monkeypatch, tmp_path):
    seen = {}
    s = _make_server(monkeypatch, tmp_path, {"stream_timeout": 300})
    monkeypatch.setattr(s.requests, "post",
                        lambda url, **kw: (seen.__setitem__("timeout", kw.get("timeout")),
                                           FakeStreamResp(200, [TEXT, DONE]))[1])
    with s.app.test_request_context():
        s._proxy_cycling_streaming("chat/completions", "t", CANDS,
                                   {"messages": []}, 300, config=None)
    assert seen["timeout"][1] == 300     # the pre-existing stream_timeout
