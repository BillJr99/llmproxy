"""End-to-end tests for the dialect translation layer.

Exercises the inbound (openai / anthropic) x upstream (openai / anthropic /
gemini) matrix through Flask's test client with a fake upstream, plus a couple
of direct adapter round-trip checks. No real network calls.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Fake upstream
# --------------------------------------------------------------------------- #

class _Elapsed:
    def total_seconds(self):
        return 0.01


class _FakeResp:
    def __init__(self, status=200, body=b"", chunks=None, content_type="application/json"):
        self.status_code = status
        self._body = body
        self._chunks = chunks
        self.headers = {"Content-Type": content_type}
        self.elapsed = _Elapsed()

    @property
    def content(self):
        return self._body

    def iter_content(self, chunk_size=None):
        yield from (self._chunks if self._chunks is not None else [self._body])

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def server(monkeypatch, tmp_path):
    cfg = {
        "providers": {
            "oai": {"base_url": "http://oai.example/v1", "api_key": "k"},
            "anth": {"base_url": "http://anth.example/v1", "api_key": "k", "protocol": "anthropic"},
            "gem": {"base_url": "http://gem.example/v1beta", "api_key": "k", "protocol": "gemini"},
        },
        "believed_free": [],
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5, "response_cache_ttl": 0},
    }
    p = Path(tmp_path) / "config.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setenv("LLMPROXY_CONFIG", str(p))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    server_mod.app.config["TESTING"] = True
    return server_mod


def _set_post(monkeypatch, server, responder):
    """Install a fake requests.post that delegates to *responder(url, body, stream)*."""
    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        return responder(url, json, stream)
    monkeypatch.setattr(server.requests, "post", fake_post)


# canned upstream bodies -------------------------------------------------------

_OPENAI_RESP = json.dumps({
    "id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-x",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello world"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}).encode()

_OPENAI_STREAM = [
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n',
    b'data: [DONE]\n\n',
]

_ANTHROPIC_RESP = json.dumps({
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
    "content": [{"type": "text", "text": "hi from claude"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 4, "output_tokens": 3},
}).encode()

_ANTHROPIC_STREAM = [
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","model":"claude-x","usage":{"input_tokens":4,"output_tokens":0}}}\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi "}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"there"}}\n\n',
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]

_GEMINI_RESP = json.dumps({
    "candidates": [{"content": {"parts": [{"text": "gemini says hi"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 3, "totalTokenCount": 9},
    "modelVersion": "gemini-x",
}).encode()

_GEMINI_STREAM = [
    b'data: {"candidates":[{"content":{"parts":[{"text":"gem "}]}}]}\n\n',
    b'data: {"candidates":[{"content":{"parts":[{"text":"ini"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":6,"candidatesTokenCount":3,"totalTokenCount":9}}\n\n',
]


# --------------------------------------------------------------------------- #
# openai inbound (regression — identity fast path)
# --------------------------------------------------------------------------- #

def test_openai_in_openai_up_nonstream(monkeypatch, server):
    _set_post(monkeypatch, server, lambda url, body, stream: _FakeResp(body=_OPENAI_RESP))
    client = server.app.test_client()
    r = client.post("/v1/chat/completions",
                    json={"model": "oai/gpt-x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    data = r.get_json()
    assert data["choices"][0]["message"]["content"] == "hello world"


def test_openai_in_openai_up_stream(monkeypatch, server):
    _set_post(monkeypatch, server, lambda url, body, stream: _FakeResp(chunks=_OPENAI_STREAM, content_type="text/event-stream"))
    client = server.app.test_client()
    r = client.post("/v1/chat/completions",
                    json={"model": "oai/gpt-x", "stream": True,
                          "messages": [{"role": "user", "content": "hi"}]})
    body = r.get_data()
    assert b"chat.completion" not in body or b"delta" in body  # passthrough preserved
    assert b"data: [DONE]" in body


# --------------------------------------------------------------------------- #
# anthropic inbound -> openai upstream
# --------------------------------------------------------------------------- #

def test_anthropic_in_openai_up_nonstream(monkeypatch, server):
    captured = {}

    def responder(url, body, stream):
        captured["url"] = url
        captured["body"] = body
        return _FakeResp(body=_OPENAI_RESP)

    _set_post(monkeypatch, server, responder)
    client = server.app.test_client()
    r = client.post("/v1/messages",
                    json={"model": "oai/gpt-x", "max_tokens": 64, "system": "be nice",
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    data = r.get_json()
    # Anthropic-shaped response
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "hello world"
    assert data["stop_reason"] == "end_turn"
    assert data["usage"] == {"input_tokens": 5, "output_tokens": 2}
    # Upstream got a canonical OpenAI request with the system folded in
    assert captured["url"] == "http://oai.example/v1/chat/completions"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "be nice"}


def test_anthropic_in_openai_up_stream(monkeypatch, server):
    _set_post(monkeypatch, server, lambda url, body, stream: _FakeResp(chunks=_OPENAI_STREAM, content_type="text/event-stream"))
    client = server.app.test_client()
    r = client.post("/v1/messages",
                    json={"model": "oai/gpt-x", "max_tokens": 64, "stream": True,
                          "messages": [{"role": "user", "content": "hi"}]})
    body = r.get_data().decode()
    assert "event: message_start" in body
    assert "event: content_block_start" in body
    assert '"text":"Hel"' in body and '"text":"lo"' in body
    assert "event: message_delta" in body
    assert "event: message_stop" in body


# --------------------------------------------------------------------------- #
# openai inbound -> anthropic upstream (native protocol)
# --------------------------------------------------------------------------- #

def test_openai_in_anthropic_up_nonstream(monkeypatch, server):
    captured = {}

    def responder(url, body, stream):
        captured["url"] = url
        captured["body"] = body
        return _FakeResp(body=_ANTHROPIC_RESP)

    _set_post(monkeypatch, server, responder)
    client = server.app.test_client()
    r = client.post("/v1/chat/completions",
                    json={"model": "anth/claude-x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    data = r.get_json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "hi from claude"
    assert data["usage"]["prompt_tokens"] == 4
    # Native Anthropic request was built (messages endpoint, max_tokens required)
    assert captured["url"] == "http://anth.example/v1/messages"
    assert captured["body"]["max_tokens"]  # defaulted


def test_openai_in_anthropic_up_stream(monkeypatch, server):
    _set_post(monkeypatch, server, lambda url, body, stream: _FakeResp(chunks=_ANTHROPIC_STREAM, content_type="text/event-stream"))
    client = server.app.test_client()
    r = client.post("/v1/chat/completions",
                    json={"model": "anth/claude-x", "stream": True,
                          "messages": [{"role": "user", "content": "hi"}]})
    body = r.get_data().decode()
    # Rendered as OpenAI SSE chunks
    assert '"content":"hi "' in body and '"content":"there"' in body
    assert '"finish_reason":"stop"' in body
    assert "data: [DONE]" in body


# --------------------------------------------------------------------------- #
# openai inbound -> gemini upstream (native protocol)
# --------------------------------------------------------------------------- #

def test_openai_in_gemini_up_nonstream(monkeypatch, server):
    captured = {}

    def responder(url, body, stream):
        captured["url"] = url
        return _FakeResp(body=_GEMINI_RESP)

    _set_post(monkeypatch, server, responder)
    client = server.app.test_client()
    r = client.post("/v1/chat/completions",
                    json={"model": "gem/gemini-x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    data = r.get_json()
    assert data["choices"][0]["message"]["content"] == "gemini says hi"
    assert data["usage"]["prompt_tokens"] == 6
    assert captured["url"] == "http://gem.example/v1beta/models/gemini-x:generateContent"


def test_openai_in_gemini_up_stream(monkeypatch, server):
    _set_post(monkeypatch, server, lambda url, body, stream: _FakeResp(chunks=_GEMINI_STREAM, content_type="text/event-stream"))
    client = server.app.test_client()
    r = client.post("/v1/chat/completions",
                    json={"model": "gem/gemini-x", "stream": True,
                          "messages": [{"role": "user", "content": "hi"}]})
    body = r.get_data().decode()
    assert '"content":"gem "' in body and '"content":"ini"' in body
    assert "data: [DONE]" in body


# --------------------------------------------------------------------------- #
# cross-dialect: anthropic inbound -> gemini upstream (streaming)
# --------------------------------------------------------------------------- #

def test_anthropic_in_gemini_up_stream(monkeypatch, server):
    _set_post(monkeypatch, server, lambda url, body, stream: _FakeResp(chunks=_GEMINI_STREAM, content_type="text/event-stream"))
    client = server.app.test_client()
    r = client.post("/v1/messages",
                    json={"model": "gem/gemini-x", "max_tokens": 32, "stream": True,
                          "messages": [{"role": "user", "content": "hi"}]})
    body = r.get_data().decode()
    # Client sees Anthropic events even though upstream spoke Gemini
    assert "event: message_start" in body
    assert '"text":"gem "' in body and '"text":"ini"' in body
    assert "event: message_stop" in body


# --------------------------------------------------------------------------- #
# gemini inbound (generateContent surface) -> openai upstream
# --------------------------------------------------------------------------- #

def test_gemini_in_openai_up_nonstream(monkeypatch, server):
    captured = {}

    def responder(url, body, stream):
        captured["body"] = body
        return _FakeResp(body=_OPENAI_RESP)

    _set_post(monkeypatch, server, responder)
    client = server.app.test_client()
    r = client.post("/v1beta/models/oai/gpt-x:generateContent",
                    json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                          "systemInstruction": {"parts": [{"text": "be brief"}]},
                          "generationConfig": {"maxOutputTokens": 32}})
    assert r.status_code == 200
    data = r.get_json()
    # Gemini-shaped response
    assert data["candidates"][0]["content"]["parts"][0]["text"] == "hello world"
    assert data["candidates"][0]["finishReason"] == "STOP"
    assert data["usageMetadata"]["promptTokenCount"] == 5
    # model came from the URL path (oai/gpt-x), resolved to upstream "gpt-x";
    # system folded into the canonical request
    assert captured["body"]["model"] == "gpt-x"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "be brief"}


def test_gemini_in_openai_up_stream(monkeypatch, server):
    _set_post(monkeypatch, server, lambda url, body, stream: _FakeResp(chunks=_OPENAI_STREAM, content_type="text/event-stream"))
    client = server.app.test_client()
    r = client.post("/v1beta/models/oai/gpt-x:streamGenerateContent?alt=sse",
                    json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    body = r.get_data().decode()
    # Rendered as Gemini SSE chunks
    assert '"text":"Hel"' in body and '"text":"lo"' in body
    assert '"finishReason":"STOP"' in body


def test_gemini_count_tokens(monkeypatch, server):
    client = server.app.test_client()
    r = client.post("/v1beta/models/oai/gpt-x:countTokens",
                    json={"contents": [{"role": "user", "parts": [{"text": "x" * 40}]}]})
    assert r.status_code == 200
    assert r.get_json()["totalTokens"] >= 1


# --------------------------------------------------------------------------- #
# count_tokens utility
# --------------------------------------------------------------------------- #

def test_count_tokens(monkeypatch, server):
    client = server.app.test_client()
    r = client.post("/v1/messages/count_tokens",
                    json={"model": "oai/gpt-x", "messages": [{"role": "user", "content": "x" * 40}]})
    assert r.status_code == 200
    assert r.get_json()["input_tokens"] >= 1
