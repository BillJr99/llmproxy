"""Tests for streamed chat-chunk normalization (``llmproxy.stream_normalize``).

The motivating failure: opencode (Vercel AI SDK, strict zod chunk schema)
aborted with "Invalid ... stream event" on a ``llmproxy/flagship__free`` stream
that failed over to Cloudflare Workers AI, whose first chunk was

    {"id":"id-1795715353439","created":1795715353,
     "model":"@cf/meta/llama-4-scout-17b-16e-instruct",
     "object":"chat.completion.chunk",
     "choices":[{"delta":{"content":"","role":"assistant"}}]}

with no ``choices[].index``. Cloudflare's tool-call deltas also vary (missing
``index`` or ``id``, ``function.arguments`` as an object) and it can send a
usage-only chunk with no ``choices``.
"""

from __future__ import annotations

import copy
import json

import pytest

from llmproxy.stream_normalize import (
    ChunkNormalizer,
    normalize_chat_chunk,
    normalize_sse_stream,
)

from .test_stream_hardening import CANDS, FakeStreamResp, _make_server

CF_MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct"

CF_FIRST = {
    "id": "id-1795715353439",
    "created": 1795715353,
    "model": CF_MODEL,
    "object": "chat.completion.chunk",
    "choices": [{"delta": {"content": "", "role": "assistant"}}],
}

VALID = {
    "id": "chatcmpl-abc",
    "object": "chat.completion.chunk",
    "created": 1700000000,
    "model": "gpt-x",
    "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
}


def _frame(obj) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


def _frames(body: bytes) -> list[dict | str]:
    out: list = []
    for block in body.split(b"\n\n"):
        block = block.strip()
        if not block.startswith(b"data:"):
            continue
        data = block[len(b"data:"):].strip()
        out.append("[DONE]" if data == b"[DONE]" else json.loads(data))
    return out


# ── normalize_chat_chunk ────────────────────────────────────────────────────

def test_valid_chunk_is_unchanged():
    chunk = copy.deepcopy(VALID)
    fixed, changed = normalize_chat_chunk(chunk)
    assert changed is False
    assert fixed == VALID


def test_cloudflare_first_chunk_gets_choice_index():
    fixed, changed = normalize_chat_chunk(copy.deepcopy(CF_FIRST))
    assert changed is True
    assert fixed["choices"][0]["index"] == 0
    assert fixed["choices"][0]["delta"] == {"content": "", "role": "assistant"}
    assert fixed["id"] == "id-1795715353439"
    assert fixed["model"] == CF_MODEL


def test_missing_choice_indices_use_position_and_delta_is_added():
    chunk = {"choices": [{"delta": {"content": "a"}}, {"finish_reason": "stop"}]}
    fixed, changed = normalize_chat_chunk(chunk, ChunkNormalizer(model="m", created=5))
    assert changed is True
    assert [c["index"] for c in fixed["choices"]] == [0, 1]
    assert fixed["choices"][1]["delta"] == {}
    assert fixed["object"] == "chat.completion.chunk"
    assert fixed["created"] == 5
    assert fixed["model"] == "m"
    assert isinstance(fixed["id"], str) and fixed["id"]


def test_metadata_only_gap_is_left_alone():
    # id/created/model are nullish in strict client schemas; such a chunk is
    # otherwise valid and must be relayed exactly as sent.
    chunk = {"choices": [{"index": 0, "delta": {"content": "hello"}}]}
    fixed, changed = normalize_chat_chunk(copy.deepcopy(chunk))
    assert changed is False and fixed == chunk


def test_top_level_id_is_stable_across_a_stream():
    state = ChunkNormalizer(model="m")
    a, _ = normalize_chat_chunk({"id": "first", "choices": [{"delta": {}}]}, state)
    b, _ = normalize_chat_chunk({"choices": [{"delta": {"content": "x"}}]}, state)
    assert a["id"] == b["id"] == "first"


def test_tool_call_arguments_object_becomes_string():
    chunk = {
        "id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
        "choices": [{"index": 0, "delta": {"tool_calls": [
            {"id": "call_1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        ]}}],
    }
    fixed, changed = normalize_chat_chunk(chunk)
    call = fixed["choices"][0]["delta"]["tool_calls"][0]
    assert changed is True
    assert call["index"] == 0
    assert call["id"] == "call_1"
    assert call["type"] == "function"
    assert call["function"]["arguments"] == '{"city": "Paris"}'
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


@pytest.mark.parametrize("args, expected", [(None, ""), ([1, 2], "[1, 2]"), ("{}", "{}")])
def test_tool_call_arguments_are_always_strings(args, expected):
    chunk = {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": args}},
    ]}}]}
    fixed, _ = normalize_chat_chunk(chunk)
    assert fixed["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == expected


def test_tool_call_existing_type_and_id_are_kept():
    chunk = copy.deepcopy(VALID)
    chunk["choices"][0]["delta"] = {"tool_calls": [
        {"index": 0, "id": "call_9", "type": "function",
         "function": {"name": "f", "arguments": "{}"}},
    ]}
    before = copy.deepcopy(chunk)
    fixed, changed = normalize_chat_chunk(chunk)
    assert changed is False
    assert fixed == before


def test_parallel_tool_calls_without_index_get_distinct_indices():
    state = ChunkNormalizer(model="m")
    a, _ = normalize_chat_chunk({"choices": [{"delta": {"tool_calls": [
        {"id": "call_a", "function": {"name": "f", "arguments": ""}}]}}]}, state)
    b, _ = normalize_chat_chunk({"choices": [{"delta": {"tool_calls": [
        {"id": "call_b", "function": {"name": "g", "arguments": ""}}]}}]}, state)
    c, _ = normalize_chat_chunk({"choices": [{"delta": {"tool_calls": [
        {"function": {"arguments": '{"x":1}'}}]}}]}, state)
    assert a["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert b["choices"][0]["delta"]["tool_calls"][0]["index"] == 1
    # an id-less fragment continues the call in progress
    assert c["choices"][0]["delta"]["tool_calls"][0]["index"] == 1
    assert "type" not in c["choices"][0]["delta"]["tool_calls"][0]


def test_usage_only_chunk_gets_empty_choices():
    chunk = {"usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}
    fixed, changed = normalize_chat_chunk(chunk, ChunkNormalizer(model="m"))
    assert changed is True
    assert fixed["choices"] == []
    assert fixed["usage"]["total_tokens"] == 7
    assert fixed["object"] == "chat.completion.chunk"


def test_finish_reason_non_string_becomes_null():
    chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": 7}]}
    fixed, _ = normalize_chat_chunk(chunk)
    assert fixed["choices"][0]["finish_reason"] is None
    chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    fixed, _ = normalize_chat_chunk(chunk)
    assert fixed["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("payload", [[1, 2], "text", 3, {"foo": "bar"}])
def test_non_chunk_payloads_are_dropped(payload):
    fixed, _ = normalize_chat_chunk(payload)
    assert fixed is None


def test_error_payload_passes_through():
    err = {"error": {"message": "quota"}}
    fixed, changed = normalize_chat_chunk(copy.deepcopy(err))
    assert fixed == err and changed is False


def test_content_text_is_never_changed():
    text = '  {"not": "json"} é\n'
    chunk = {"choices": [{"delta": {"content": text}}]}
    fixed, _ = normalize_chat_chunk(chunk)
    assert fixed["choices"][0]["delta"]["content"] == text


# ── normalize_sse_stream ────────────────────────────────────────────────────

def test_valid_stream_is_byte_identical_even_when_split_mid_event():
    raw = _frame(VALID) + b": keep-alive\n\n" + _frame(VALID) + b"data: [DONE]\n\n"
    pieces = [raw[i:i + 7] for i in range(0, len(raw), 7)]
    assert b"".join(normalize_sse_stream(pieces, model="gpt-x")) == raw


def test_crlf_framed_valid_stream_is_byte_identical():
    raw = _frame(VALID).replace(b"\n", b"\r\n") + b"data: [DONE]\r\n\r\n"
    assert b"".join(normalize_sse_stream([raw])) == raw


def test_cloudflare_stream_is_repaired():
    usage = {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
    tool = {
        "id": "id-1", "created": 1795715353, "model": CF_MODEL,
        "object": "chat.completion.chunk",
        "choices": [{"delta": {"tool_calls": [
            {"function": {"name": "f", "arguments": {"q": "x"}}}]}}],
    }
    raw = (_frame(CF_FIRST) + _frame(tool) + _frame(usage)
           + b"data: not json\n\n" + b"data: [DONE]\n\n")
    frames = _frames(b"".join(normalize_sse_stream([raw], model=CF_MODEL)))
    assert frames[-1] == "[DONE]"
    first, second, third = frames[:3]
    assert len(frames) == 4  # the non-JSON event was dropped
    assert first["choices"][0]["index"] == 0
    call = second["choices"][0]["delta"]["tool_calls"][0]
    assert call == {"function": {"name": "f", "arguments": '{"q": "x"}'},
                    "index": 0, "type": "function"}
    assert third["choices"] == [] and third["id"] == "id-1795715353439"


def test_comments_stay_comments_and_event_lines_are_kept():
    raw = b": ping\n\nevent: message\ndata: " + json.dumps(CF_FIRST).encode() + b"\n\n"
    out = b"".join(normalize_sse_stream([raw]))
    assert out.startswith(b": ping\n\n")
    assert b"event: message\ndata: {" in out


def test_trailing_unterminated_event_is_flushed():
    raw = _frame(VALID).rstrip(b"\n")
    assert b"".join(normalize_sse_stream([raw])) == raw


# ── wired into the relay ────────────────────────────────────────────────────

def test_cycling_stream_relays_normalized_cloudflare_chunks(monkeypatch, tmp_path):
    server = _make_server(monkeypatch, tmp_path)
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: FakeStreamResp(
        200, [_frame(CF_FIRST), b"data: [DONE]\n\n"]))
    with server.app.test_request_context():
        resp = server._proxy_cycling_streaming(
            "chat/completions", "t", CANDS[:1], {"messages": []}, 5, config={"server": {}},
        )
    frames = _frames(b"".join(resp.response))
    assert frames[0]["choices"][0]["index"] == 0
    assert frames[-1] == "[DONE]"


def test_pinned_stream_relays_normalized_chunks(monkeypatch, tmp_path):
    server = _make_server(monkeypatch, tmp_path)
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: FakeStreamResp(
        200, [_frame(CF_FIRST), b"data: [DONE]\n\n"]))
    with server.app.test_request_context():
        resp = server._proxy_streaming(
            "chat/completions", "p1", {"base_url": "http://p1.example/v1", "api_key": "k"},
            {"model": "m1", "messages": []}, 5,
        )
    assert _frames(b"".join(resp.response))[0]["choices"][0]["index"] == 0


def test_normalization_can_be_switched_off(monkeypatch, tmp_path):
    server = _make_server(monkeypatch, tmp_path, {"stream_normalize": False})
    raw = _frame(CF_FIRST) + b"data: [DONE]\n\n"
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: FakeStreamResp(200, [raw]))
    with server.app.test_request_context():
        resp = server._proxy_cycling_streaming(
            "chat/completions", "t", CANDS[:1], {"messages": []}, 5,
            config={"server": {"stream_normalize": False}},
        )
    assert b"".join(resp.response) == raw


def test_non_chat_endpoint_is_not_normalized(monkeypatch, tmp_path):
    server = _make_server(monkeypatch, tmp_path)
    raw = b'data: {"choices":[{"text":"hi"}],"object":"text_completion"}\n\n'
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: FakeStreamResp(200, [raw]))
    with server.app.test_request_context():
        resp = server._proxy_streaming(
            "completions", "p1", {"base_url": "http://p1.example/v1", "api_key": "k"},
            {"model": "m1", "prompt": "x"}, 5,
        )
    assert b"".join(resp.response) == raw


# --- review round 1 (agy): regressions ---------------------------------------

def _events(stream: bytes, split: int | None = None):
    parts = [stream] if split is None else [stream[i:i + split] for i in range(0, len(stream), split)]
    return list(normalize_sse_stream(parts))


def test_unicode_line_separators_in_content_are_not_split():
    chunk = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
             "choices": [{"index": 0, "delta": {"content": "a b c\u0085d"}, "finish_reason": None}]}
    raw = ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()
    assert b"".join(_events(raw)) == raw


def test_malformed_string_indices_do_not_raise():
    for bad in ("--1", "²", "-1", -1, "1.5"):
        chunk = {"choices": [{"index": bad, "delta": {"tool_calls": [{"index": bad, "id": "x",
                 "function": {"name": "f", "arguments": "{}"}}]}}]}
        fixed, changed = ChunkNormalizer().normalize(chunk)
        assert changed
        assert fixed["choices"][0]["index"] == 0
        assert fixed["choices"][0]["delta"]["tool_calls"][0]["index"] == 0


def test_idless_named_calls_in_separate_chunks_get_distinct_indices():
    st = ChunkNormalizer()
    a, _ = st.normalize({"choices": [{"delta": {"tool_calls": [{"function": {"name": "a", "arguments": "{"}}]}}]})
    frag, _ = st.normalize({"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "}"}}]}}]})
    b, _ = st.normalize({"choices": [{"delta": {"tool_calls": [{"function": {"name": "b", "arguments": "{}"}}]}}]})
    assert a["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert frag["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert b["choices"][0]["delta"]["tool_calls"][0]["index"] == 1


def test_idless_first_call_then_id_call_do_not_collide():
    st = ChunkNormalizer()
    st.normalize({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "a", "arguments": "{}"}}]}}]})
    b, _ = st.normalize({"choices": [{"delta": {"tool_calls": [{"id": "call_2", "function": {"name": "b", "arguments": "{}"}}]}}]})
    assert b["choices"][0]["delta"]["tool_calls"][0]["index"] == 1


def test_empty_type_is_repaired():
    fixed, changed = ChunkNormalizer().normalize(
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "x", "type": "",
         "function": {"name": "f", "arguments": "{}"}}]}}]})
    assert changed and fixed["choices"][0]["delta"]["tool_calls"][0]["type"] == "function"


def test_mixed_terminators_split_into_separate_events():
    def c(t):
        return json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                           "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})
    raw = f"data: {c('x')}\r\n\ndata: {c('y')}\n\r\ndata: [DONE]\n\n".encode()
    out = _events(raw)
    assert len(out) == 3 and b"".join(out) == raw


def test_large_single_buffer_is_linear():
    import time as _t
    ev = ("data: " + json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
          "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]}) + "\n\n").encode()
    raw = ev * 20000
    t0 = _t.perf_counter()
    out = _events(raw)
    assert len(out) == 20000
    assert _t.perf_counter() - t0 < 2.0


# --- review round 2 (codex follow-up) ----------------------------------------

def test_cr_only_event_separators():
    ev = json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                     "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]})
    raw = f"data: {ev}\r\rdata: {ev}\r\r".encode()
    out = _events(raw)
    assert len(out) == 2 and b"".join(out) == raw


def test_crlf_within_event_is_not_a_separator():
    raw = b'event: x\r\ndata: {"choices":[{"index":0,"delta":{"content":"a"}}]}\r\n\r\n'
    out = _events(raw, split=3)
    assert b"".join(out) == raw


def test_lone_surrogate_is_preserved_when_the_frame_needs_repair():
    raw = b'data: {"choices":[{"delta":{"content":"\\ud800"}}]}\n\n'
    fixed, = _frames(b"".join(_events(raw)))
    assert fixed["choices"] == [{"index": 0, "delta": {"content": "\ud800"}}]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (42, "42"),
        (1.25, "1.25"),
        (True, "true"),
        (False, "false"),
        ({"text": "untyped object"}, None),
        ([True, {"type": 5}, {"text": "untyped"}, {"type": "text", "text": "keep"}],
         [{"type": "text", "text": "keep"}]),
    ],
)
def test_bad_content_relay(content, expected):
    chunk = copy.deepcopy(VALID)
    chunk["choices"][0]["delta"]["content"] = content
    fixed, = _frames(b"".join(_events(_frame(chunk))))
    delta = fixed["choices"][0]["delta"]
    if expected is None:
        assert "content" not in delta
    else:
        assert delta["content"] == expected


@pytest.mark.parametrize(("field", "value"), [("id", 123), ("model", 9), ("created", "17"), ("created", True)])
def test_wrong_metadata_type_alone_triggers_repair(field, value):
    chunk = copy.deepcopy(VALID)
    chunk[field] = value
    fixed, = _frames(b"".join(normalize_sse_stream([_frame(chunk)], model="fallback")))
    assert isinstance(fixed[field], {"id": str, "model": str, "created": int}[field])
    assert not isinstance(fixed[field], bool)
    if value == "17":
        assert fixed[field] == 17


def test_invalid_role_reasoning_and_non_array_tool_calls_are_dropped():
    chunk = copy.deepcopy(VALID)
    chunk["choices"][0]["delta"].update(
        role="user", reasoning=5, reasoning_content=False, tool_calls={"function": {}},
    )
    fixed, = _frames(b"".join(_events(_frame(chunk))))
    assert fixed["choices"][0]["delta"] == {"content": "Hi"}


def test_malformed_calls_drop_without_inventing_tool_ids():
    chunk = copy.deepcopy(VALID)
    chunk["choices"][0]["delta"]["tool_calls"] = [
        0, {"id": "missing"}, {"function": "wrong"},
        {"id": 42, "function": {"name": 9, "arguments": "{}"},
         "extra_content": {"google": {"thought_signature": 42}}},
        {"id": "real", "function": {"name": "fn", "arguments": "{}"},
         "extra_content": {"google": {"thought_signature": "sig"}}},
    ]
    fixed, = _frames(b"".join(_events(_frame(chunk))))
    malformed, real = fixed["choices"][0]["delta"]["tool_calls"]
    assert "id" not in malformed
    assert malformed["function"] == {"arguments": "{}"}
    assert malformed["extra_content"] == {"google": {}}
    assert real["id"] == "real"
    assert real["function"]["name"] == "fn"
    assert real["extra_content"] == {"google": {"thought_signature": "sig"}}


@pytest.mark.parametrize("extra", [False, {"google": "invalid"}])
def test_malformed_tool_extra_content_is_removed(extra):
    chunk = copy.deepcopy(VALID)
    chunk["choices"][0]["delta"]["tool_calls"] = [
        {"index": 0, "id": "real", "type": "function", "function": {"arguments": "{}"}, "extra_content": extra},
    ]
    fixed, = _frames(b"".join(_events(_frame(chunk))))
    call = fixed["choices"][0]["delta"]["tool_calls"][0]
    assert call.get("extra_content") == ({} if isinstance(extra, dict) else None)


def test_choice_and_usage_repairs_preserve_numeric_usage_and_unknown_fields():
    chunk = copy.deepcopy(VALID)
    chunk["choices"] = [False, None, chunk["choices"][0]]
    chunk["usage"] = {
        "prompt_tokens": "7",
        "completion_tokens": True,
        "total_tokens": 9.25,
        "billed_tokens": "unmodeled",
        "prompt_tokens_details": {"cached_tokens": "6.5e0", "provider": "keep"},
        "completion_tokens_details": {
            "reasoning_tokens": "unknown",
            "accepted_prediction_tokens": "1e999",
            "rejected_prediction_tokens": 2,
        },
    }
    fixed, = _frames(b"".join(_events(_frame(chunk))))
    assert fixed["choices"] == VALID["choices"]
    assert fixed["usage"] == {
        "prompt_tokens": 7,
        "total_tokens": 9.25,
        "billed_tokens": "unmodeled",
        "prompt_tokens_details": {"cached_tokens": 6.5, "provider": "keep"},
        "completion_tokens_details": {"rejected_prediction_tokens": 2},
    }


def test_invalid_usage_objects_are_dropped_and_valid_typed_content_is_untouched():
    chunk = copy.deepcopy(VALID)
    chunk["usage"] = "invalid"
    fixed, = _frames(b"".join(_events(_frame(chunk))))
    assert "usage" not in fixed
    chunk["usage"] = {"prompt_tokens_details": "invalid", "completion_tokens_details": None}
    fixed, = _frames(b"".join(_events(_frame(chunk))))
    assert fixed["usage"] == {"completion_tokens_details": None}

    chunk["usage"] = {"prompt_tokens": 4.25, "completion_tokens": None}
    chunk["created"] = 1700000000.5
    chunk["choices"][0]["delta"] = {
        "content": [{"type": "text", "text": " keep spacing "}],
        "role": "",
    }
    raw = b"data: " + json.dumps(chunk, indent=None).encode() + b"\r\n\r\n"
    assert b"".join(_events(raw)) == raw


@pytest.mark.parametrize(
    "chunk",
    [
        {"choices": [{"index": 0, "delta": {"content": None, "role": "", "tool_calls": None}}]},
        {"id": None, "created": None, "model": None, "choices": []},
        {"error": {"message": "limited", "type": None, "code": 429}},
    ],
)
def test_valid_nullish_metadata_content_and_error_bytes_are_preserved(chunk):
    raw = b"data: " + json.dumps(chunk).encode() + b"\r\n\r\n"
    assert b"".join(_events(raw)) == raw


@pytest.mark.parametrize("function", [{}, {"arguments": None}])
def test_legacy_tool_argument_default_is_retained(function):
    chunk = {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": function}]}}]}
    fixed, = _frames(b"".join(_events(_frame(chunk))))
    assert fixed["choices"][0]["delta"]["tool_calls"][0]["function"] == {"arguments": ""}
