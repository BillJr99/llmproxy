"""Every way a streamed candidate can fail must be visible and must count.

`GET /v1/failures` was added to answer "which models are failing, and why", but
it was wired into only the HTTP-status path. The streaming loop has eight ways
to fail over and six recorded nothing at all — including a connect timeout,
the single most common failure a free-tier pool produces.

The cost was not only a blind report. Those paths also left the loop's roll-call
of what it had tried empty, so a walk in which every candidate timed out fell
through to a bare `503`. That read as "there was nothing to try" when five
candidates had just been tried for sixty seconds each, and told the caller
nothing about which models had burned five minutes of its time.

Each test names the path it covers, because the failure mode was precisely that
paths existed which nobody had enumerated.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import requests


def _load_server(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def S(tmp_path: Path, monkeypatch):
    cfg = {
        "providers": {
            "p1": {"base_url": "http://p1.example/v1", "api_key": "k", "model_filter": None},
            "p2": {"base_url": "http://p2.example/v1", "api_key": "k", "model_filter": None},
        },
        "sync_believed_free_on_startup": False,
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    server = _load_server(monkeypatch, path)
    server._reset_failures()
    return server


class FakeStreamResp:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(self, status: int, chunks: list[bytes], content: bytes = b"",
                 headers=None, die: Exception | None = None):
        self.status_code = status
        self._chunks = list(chunks)
        self.content = content
        self.headers = headers or {"Content-Type": "text/event-stream"}
        self._die = die

    def iter_content(self, chunk_size=None):
        yield from self._chunks
        if self._die is not None:
            raise self._die

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


CANDS = [
    ("p1", {"base_url": "http://p1.example/v1", "api_key": "k"}, "m1"),
    ("p2", {"base_url": "http://p2.example/v1", "api_key": "k"}, "m2"),
]

ERRFRAME = b'data: {"error":{"message":"upstream exploded"}}\n\n'
TEXT = b'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n'


def _run(S, cands=CANDS, config=None):
    with S.app.test_request_context():
        return S._proxy_cycling_streaming(
            "chat/completions", "flagship__free", cands, {"messages": []}, 5,
            config=config, virtual_model="llmproxy__flagship/free",
        )


def _raise(exc):
    def _boom(*a, **k):
        raise exc
    return _boom


# ── the reported incident: every candidate times out at connect ─────────────

def test_an_all_timeout_walk_reports_502_not_a_bare_503(S, monkeypatch):
    """Five candidates x a 60s timeout is an exhausted pool, not an empty one."""
    monkeypatch.setattr(S.requests, "post",
                        _raise(requests.exceptions.Timeout("Read timed out.")))
    resp = _run(S)
    assert resp.status_code == 502
    listed = json.loads(resp.get_data())["error"]["llmproxy_candidates"]
    assert [c["target"] for c in listed] == ["p1/m1", "p2/m2"]
    assert all(c["status"] is None for c in listed)


def test_an_all_timeout_walk_is_visible_in_the_failure_report(S, monkeypatch):
    """The report exists to answer exactly this, and could not see it."""
    monkeypatch.setattr(S.requests, "post",
                        _raise(requests.exceptions.Timeout("Read timed out.")))
    _run(S)
    rows = S._failure_records()
    assert {r["target"] for r in rows} == {"p1/m1", "p2/m2"}
    assert {r["kind"] for r in rows} == {"timeout"}
    assert all(r["duration_ms"] is not None for r in rows)
    assert all(r["virtual_model"] == "llmproxy__flagship/free" for r in rows)


def test_the_report_surfaces_how_long_each_candidate_burned(S):
    """A pool burning a full timeout and one returning fast 404s differ."""
    S._record_failure("p1", "m1", status=None, kind="timeout", duration_ms=60012.4)
    with S.app.test_client() as c:
        entry = c.get("/v1/failures").get_json()["by_model"][0]
    assert entry["slowest_ms"] == 60012.4


def test_a_connection_error_is_distinguished_from_a_timeout(S, monkeypatch):
    """One upstream went quiet; the other was never there."""
    monkeypatch.setattr(S.requests, "post",
                        _raise(requests.exceptions.ConnectionError("no such host")))
    _run(S)
    assert {r["kind"] for r in S._failure_records()} == {"connection"}


def test_an_unexpected_connect_exception_is_still_recorded(S, monkeypatch):
    """A TLS or protocol error must not vanish for not being a Timeout."""
    monkeypatch.setattr(S.requests, "post", _raise(ValueError("bad TLS handshake")))
    assert _run(S).status_code == 502
    assert len(S._failure_records()) == 2


# ── the stream-level paths ──────────────────────────────────────────────────

def test_a_200_that_opens_with_an_sse_error_is_recorded(S, monkeypatch):
    monkeypatch.setattr(S.requests, "post",
                        lambda *a, **k: FakeStreamResp(200, [ERRFRAME]))
    _run(S)
    rows = S._failure_records()
    assert len(rows) == 2
    assert {r["kind"] for r in rows} == {"stream"}
    assert "upstream exploded" in rows[0]["detail"]


def test_a_200_that_produces_no_output_is_recorded(S, monkeypatch):
    """The streaming counterpart of a 200 with an unusable body.

    Detecting it at all requires `stream_commit_on_content`: with the default
    one-chunk peek there is nothing to notice, so the stream commits and the
    client receives an empty reply. That is pre-existing behaviour the widened
    window exists to fix; what is asserted here is that once the window DOES
    catch it, the candidate is recorded rather than silently skipped.
    """
    monkeypatch.setattr(S.requests, "post", lambda *a, **k: FakeStreamResp(200, []))
    _run(S, config={"server": {"stream_commit_on_content": True}})
    rows = S._failure_records()
    assert len(rows) == 2
    assert {r["kind"] for r in rows} == {"stream"}


def test_a_stream_that_dies_while_opening_is_recorded(S, monkeypatch):
    """Died mid-peek: no status, no body, and previously no record either."""
    monkeypatch.setattr(S.requests, "post", lambda *a, **k: FakeStreamResp(
        200, [], die=requests.exceptions.Timeout("read timed out")))
    _run(S)
    rows = S._failure_records()
    assert len(rows) == 2
    assert {r["kind"] for r in rows} == {"timeout"}


def test_a_buffered_stream_that_dies_mid_generation_is_recorded(S, monkeypatch):
    """Under stream_buffer_full the death is caught pre-commit and failed over."""
    monkeypatch.setattr(S.requests, "post", lambda *a, **k: FakeStreamResp(
        200, [TEXT], die=requests.exceptions.ConnectionError("died at token 500")))
    _run(S, config={"server": {"stream_buffer_full": True}})
    rows = S._failure_records()
    assert len(rows) == 2
    assert {r["kind"] for r in rows} == {"stream"}
    assert "failed before completion" in rows[0]["detail"]


# ── guards ──────────────────────────────────────────────────────────────────

def test_a_unanimous_5xx_is_still_relayed_untouched(S, monkeypatch):
    """A 5xx already says "server side"; relaying preserves the diagnostic."""
    monkeypatch.setattr(S.requests, "post", lambda *a, **k: FakeStreamResp(
        503, [], content=b'{"error":{"message":"overloaded"}}',
        headers={"Content-Type": "application/json"}))
    resp = _run(S)
    assert resp.status_code == 503
    assert {r["status"] for r in S._failure_records()} == {503}


def test_a_unanimous_404_still_becomes_502(S, monkeypatch):
    """The original reported bug, on the streaming path."""
    monkeypatch.setattr(S.requests, "post", lambda *a, **k: FakeStreamResp(
        404, [], content=b'{"error":{"message":"No endpoints support tool use."}}',
        headers={"Content-Type": "application/json"}))
    assert _run(S).status_code == 502


def test_an_empty_candidate_list_still_reports_503(S):
    """503 keeps meaning "there was nothing to try"."""
    resp = _run(S, cands=[])
    assert resp.status_code == 503
    assert S._failure_records() == []
