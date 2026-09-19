"""Per-request audit records (``server.request_log``).

The two human log lines say what llmproxy is doing; a record says what it did.
Three properties carry the feature and are asserted directly here: a record is
emitted exactly once per request and carries a correlation id the client also
receives; a *streamed* record is emitted when the stream ends rather than when
the headers go out, so its duration and body are the real ones; and ``full``
mode never writes credentials down, however much content it is capturing.
"""

from __future__ import annotations

import datetime
import importlib
import json
import logging
import time
from pathlib import Path

import pytest


def _make_server(monkeypatch, tmp_path: Path, server_block: dict | None = None):
    cfg = {
        "providers": {"p1": {"base_url": "http://p1.example/v1", "api_key": "sk-SECRET"}},
        "believed_free": [],
        "model_reasoning": {},
        "server": {"log_level": "ERROR", **(server_block or {})},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("LLMPROXY_CONFIG", str(p))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    with server_mod._model_route_cache_lock:
        server_mod._model_route_cache.update({"p1__m1": ("p1", "m1")})
    return server_mod


class _Captured(logging.Handler):
    """Collect the JSON records the request channel emits."""

    def __init__(self):
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record):
        self.records.append(json.loads(record.getMessage()))


@pytest.fixture
def captured(monkeypatch):
    def _attach(server):
        h = _Captured()
        server.request_logger.handlers = [h]
        return h
    return _attach


class _Json:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    content = b'{"choices":[{"message":{"content":"four"}}],"usage":{"total_tokens":11}}'
    elapsed = datetime.timedelta(milliseconds=12)


class _Stream:
    status_code = 200
    headers = {"Content-Type": "text/event-stream"}

    def iter_content(self, chunk_size=None):
        for chunk in (b'data: {"choices":[{"delta":{"content":"four"}}]}\n\n',
                      b"data: [DONE]\n\n"):
            time.sleep(0.02)   # real elapsed time, so duration_ms means something
            yield chunk

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _chat(server, **over):
    payload = {"model": "p1/m1", "messages": [{"role": "user", "content": "2+2?"}], **over}
    return server.app.test_client().post("/v1/chat/completions", json=payload)


# ── the knob ────────────────────────────────────────────────────────────────

def test_off_by_default(monkeypatch, tmp_path, captured):
    """An untouched config records nothing: this decides whether user content is
    written down, so it cannot arrive switched on."""
    s = _make_server(monkeypatch, tmp_path)
    h = captured(s)
    assert s._request_log_mode({"server": {}}) == "off"
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: _Json())
    _chat(s)
    assert h.records == []


@pytest.mark.parametrize("raw,expected", [
    ("metadata", "metadata"), ("full", "full"), ("FULL", "full"), (" full ", "full"),
    ("off", "off"), (True, "full"), (False, "off"),
    # Anything unrecognised must fail CLOSED — silently logging prompts because
    # of a typo is the one failure mode this must not have.
    ("verbose", "off"), ("", "off"), (None, "off"), (7, "off"), ([], "off"),
])
def test_mode_parsing_fails_closed(monkeypatch, tmp_path, raw, expected):
    s = _make_server(monkeypatch, tmp_path)
    assert s._request_log_mode({"server": {"request_log": raw}}) == expected


# ── one record per request, correlated ──────────────────────────────────────

def test_one_record_per_request_with_an_id_the_client_also_gets(
        monkeypatch, tmp_path, captured):
    """The '→'/'←' pair cannot be paired under concurrency. The id is what makes
    a record, its log lines, and a user's bug report refer to one request."""
    s = _make_server(monkeypatch, tmp_path, {"request_log": "metadata"})
    h = captured(s)
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: _Json())
    resp = _chat(s)

    assert len(h.records) == 1
    rec = h.records[0]
    assert rec["object"] == "request.record"
    assert rec["id"] == resp.headers["X-LLMProxy-Request-Id"]
    assert rec["method"] == "POST" and rec["path"] == "/v1/chat/completions"
    assert rec["status"] == 200 and rec["duration_ms"] >= 0
    assert rec["model"] == "p1/m1" and rec["selected_model"] == "p1/m1"
    assert rec["streamed"] is False


def test_metadata_mode_records_no_content(monkeypatch, tmp_path, captured):
    s = _make_server(monkeypatch, tmp_path, {"request_log": "metadata"})
    h = captured(s)
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: _Json())
    _chat(s)
    blob = json.dumps(h.records[0])
    assert "2+2?" not in blob and "four" not in blob
    assert "request_body" not in h.records[0]


def test_full_mode_records_both_bodies(monkeypatch, tmp_path, captured):
    s = _make_server(monkeypatch, tmp_path, {"request_log": "full"})
    h = captured(s)
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: _Json())
    _chat(s)
    rec = h.records[0]
    assert rec["request_body"]["messages"][0]["content"] == "2+2?"
    assert rec["response_body"]["choices"][0]["message"]["content"] == "four"


def test_the_model_recorded_is_the_one_the_client_wrote(
        monkeypatch, tmp_path, captured):
    """The proxy canonicalises payload['model'] in place. The record must say
    what was ASKED for; what it resolved to is selected_model."""
    s = _make_server(monkeypatch, tmp_path, {"request_log": "full"})
    h = captured(s)
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: _Json())
    s.app.test_client().post("/v1/chat/completions",
                             json={"model": "p1/m1", "messages": []})
    assert h.records[0]["model"] == "p1/m1"


# ── streaming: the record has to outlive after_request ──────────────────────

def test_a_streamed_record_waits_for_the_stream_to_finish(
        monkeypatch, tmp_path, captured):
    """Flask runs after_request BEFORE the WSGI server pulls a chunk, so a record
    written there reports time-to-headers and an empty body."""
    s = _make_server(monkeypatch, tmp_path, {"request_log": "full"})
    h = captured(s)
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: _Stream())
    resp = _chat(s, stream=True)

    assert h.records == [], "record emitted before the stream was consumed"
    body = b"".join(resp.response)
    assert len(h.records) == 1

    rec = h.records[0]
    assert rec["streamed"] is True
    assert rec["duration_ms"] >= 40          # both 20ms chunks, not time-to-headers
    assert rec["response_body"] == body.decode()
    assert "four" in rec["response_body"]


def test_a_client_hangup_mid_stream_still_produces_a_record(
        monkeypatch, tmp_path, captured):
    """A disconnect is an outcome worth recording, not a reason to lose the row."""
    s = _make_server(monkeypatch, tmp_path, {"request_log": "full"})
    h = captured(s)
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: _Stream())
    resp = _chat(s, stream=True)
    it = iter(resp.response)
    next(it)                                  # take one chunk, then walk away
    it.close()
    assert len(h.records) == 1
    assert h.records[0]["streamed"] is True


# ── credentials never land in the stream ────────────────────────────────────

def test_admin_bodies_are_never_captured(monkeypatch, tmp_path, captured):
    """The admin API takes API keys in the clear — that is how you set one — so
    full mode must not turn a config change into a plaintext key on stdout."""
    s = _make_server(monkeypatch, tmp_path, {"request_log": "full"})
    h = captured(s)
    s.app.test_client().post(
        "/admin/api/providers",
        json={"name": "p2", "base_url": "http://p2.example/v1",
              "api_key": "sk-PLAINTEXT-SHOULD-NEVER-BE-LOGGED"},
    )
    assert h.records, "an admin request must still be recorded"
    blob = json.dumps(h.records)
    assert "sk-PLAINTEXT-SHOULD-NEVER-BE-LOGGED" not in blob
    assert h.records[0]["bodies_omitted"] == "credential-bearing path"
    assert h.records[0]["path"].startswith("/admin")


def test_request_headers_are_never_recorded(monkeypatch, tmp_path, captured):
    """Headers carry the caller's Authorization. A record stream that has to be
    handled as credential material is one nobody will keep."""
    s = _make_server(monkeypatch, tmp_path, {"request_log": "full"})
    h = captured(s)
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: _Json())
    s.app.test_client().post(
        "/v1/chat/completions",
        json={"model": "p1/m1", "messages": []},
        headers={"Authorization": "Bearer sk-CALLER-TOKEN"},
    )
    blob = json.dumps(h.records)
    assert "sk-CALLER-TOKEN" not in blob and "Authorization" not in blob


# ── robustness ──────────────────────────────────────────────────────────────

def test_a_body_cap_truncates_and_says_so(monkeypatch, tmp_path, captured):
    s = _make_server(monkeypatch, tmp_path,
                     {"request_log": "full", "request_log_max_body_bytes": 40})
    h = captured(s)
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: _Json())
    _chat(s, messages=[{"role": "user", "content": "x" * 500}])
    rec = h.records[0]
    assert rec["request_body"]["truncated"] is True
    assert rec["request_body"]["bytes"] > 40
    assert len(rec["request_body"]["text"]) <= 40


def test_a_non_json_body_is_kept_as_text_not_dropped(monkeypatch, tmp_path):
    s = _make_server(monkeypatch, tmp_path, {"request_log": "full"})
    assert s._json_or_text(b"data: [DONE]\n\n", 0) == "data: [DONE]\n\n"
    assert s._json_or_text(b'{"a":1}', 0) == {"a": 1}
    assert s._json_or_text(b"", 0) is None
    assert s._json_or_text(None, 0) is None


def test_a_broken_record_never_breaks_the_reply(monkeypatch, tmp_path):
    """Auditing is never allowed to be what fails a request."""
    s = _make_server(monkeypatch, tmp_path, {"request_log": "full"})
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: _Json())

    def _boom(*a, **k):
        raise RuntimeError("log sink exploded")

    monkeypatch.setattr(s.request_logger, "info", _boom)
    assert _chat(s).status_code == 200


def test_every_request_is_recorded_not_just_the_proxy_surface(
        monkeypatch, tmp_path, captured):
    s = _make_server(monkeypatch, tmp_path, {"request_log": "metadata"})
    h = captured(s)
    client = s.app.test_client()
    client.get("/health")
    client.get("/v1/models")
    assert [r["path"] for r in h.records] == ["/health", "/v1/models"]
    assert all(r["model"] is None for r in h.records)
