"""Tests for the ``responses`` outbound protocol (canonical chat -> /v1/responses)."""

from __future__ import annotations

import json

from llmproxy.dialects import get_outbound
from llmproxy.providers import get_provider_templates


def _sse(*events: dict) -> list[bytes]:
    return [
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    ]


def test_build_request_targets_responses_with_bearer():
    out = get_outbound("responses")
    url, headers, body = out.build_request(
        "chat/completions", "https://api.openai.com/v1",
        {"api_key": "sk-test"}, {"model": "gpt-5.5-pro", "messages": []},
        stream=False, forwarded_headers={"User-Agent": "ua"},
    )
    assert url == "https://api.openai.com/v1/responses"
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["User-Agent"] == "ua"
    assert body["store"] is False


def test_request_maps_messages_tools_and_limits():
    out = get_outbound("responses")
    payload = {
        "model": "gpt-5.5-pro",
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 256,
        "reasoning_effort": "high",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://x/img.png", "detail": "low"}},
            ]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "lookup", "description": "d", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "s", "schema": {"type": "object"}}},
    }
    _url, _h, body = out.build_request(
        "chat/completions", "https://b/v1", {"api_key": "k"}, payload,
        stream=True, forwarded_headers={},
    )
    assert body["instructions"] == "Be terse."
    assert body["max_output_tokens"] == 256
    assert body["reasoning"] == {"effort": "high"}
    assert body["stream"] is True
    assert "stream_options" not in body and "messages" not in body
    assert body["temperature"] == 0.2
    user, call, result = body["input"]
    assert user["role"] == "user"
    assert user["content"][0] == {"type": "input_text", "text": "What is this?"}
    assert user["content"][1] == {"type": "input_image", "image_url": "https://x/img.png",
                                  "detail": "low"}
    assert call == {"type": "function_call", "call_id": "call_1", "name": "lookup",
                    "arguments": "{\"q\":\"x\"}"}
    assert result == {"type": "function_call_output", "call_id": "call_1", "output": "result"}
    assert body["tools"] == [{"type": "function", "name": "lookup", "description": "d",
                              "parameters": {"type": "object"}}]
    assert body["tool_choice"] == {"type": "function", "name": "lookup"}
    assert body["text"] == {"format": {"type": "json_schema", "name": "s",
                                       "schema": {"type": "object"}}}


def test_text_only_content_stays_a_string():
    out = get_outbound("responses")
    _u, _h, body = out.build_request(
        "chat/completions", "https://b/v1", {}, {"model": "m", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": "hello"},
        ]}, stream=False, forwarded_headers={},
    )
    assert body["input"] == [{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "hello"}]


def test_translate_text_response_and_usage():
    out = get_outbound("responses")
    native = {
        "id": "resp_1", "object": "response", "created_at": 1, "status": "completed",
        "model": "gpt-5.5-pro",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Hello", "annotations": []}]},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 4},
                  "output_tokens_details": {"reasoning_tokens": 3}},
    }
    data = json.loads(out.translate_response(json.dumps(native).encode()))
    choice = data["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello"}
    assert choice["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 10
    assert data["usage"]["completion_tokens"] == 5
    assert data["usage"]["prompt_tokens_details"] == {"cached_tokens": 4}
    assert data["usage"]["completion_tokens_details"] == {"reasoning_tokens": 3}


def test_translate_tool_call_and_incomplete():
    out = get_outbound("responses")
    native = {"status": "completed", "model": "m", "output": [
        {"type": "function_call", "id": "fc_1", "call_id": "call_9", "name": "f",
         "arguments": "{}"}]}
    choice = json.loads(out.translate_response(json.dumps(native).encode()))["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["id"] == "call_9"
    assert choice["message"]["tool_calls"][0]["function"] == {"name": "f", "arguments": "{}"}

    cut = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
           "model": "m", "output": []}
    assert json.loads(out.translate_response(json.dumps(cut).encode()))[
        "choices"][0]["finish_reason"] == "length"


def test_error_body_passes_through():
    out = get_outbound("responses")
    err = json.dumps({"error": {"message": "bad", "type": "invalid_request_error"}}).encode()
    assert out.translate_response(err) == err


def test_stream_text_and_usage():
    out = get_outbound("responses")
    chunks = list(out.parse_stream(_sse(
        {"type": "response.created", "response": {"model": "gpt-5.5-pro"}},
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {"type": "response.completed", "response": {
            "status": "completed", "model": "gpt-5.5-pro",
            "usage": {"input_tokens": 3, "output_tokens": 2}}},
    )))
    assert chunks[-1] is None
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[1:-1])
    assert text == "Hello"
    final = chunks[-2]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_stream_tool_call():
    out = get_outbound("responses")
    chunks = list(out.parse_stream(_sse(
        {"type": "response.created", "response": {"model": "m"}},
        {"type": "response.output_item.added", "output_index": 1,
         "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                  "name": "lookup", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{\"q\""},
        {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": ":1}"},
        {"type": "response.completed", "response": {"status": "completed", "model": "m"}},
    )))
    opened = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    assert opened["index"] == 0 and opened["id"] == "call_1"
    assert opened["function"]["name"] == "lookup"
    args = "".join(c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                   for c in chunks[2:4])
    assert args == "{\"q\":1}"
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_failure_surfaces_an_error_chunk():
    out = get_outbound("responses")
    chunks = list(out.parse_stream(_sse(
        {"type": "response.created", "response": {"model": "m"}},
        {"type": "response.failed", "response": {
            "status": "failed", "error": {"code": "server_error", "message": "boom"}}},
    )))
    errors = [c for c in chunks if c and c.get("error")]
    assert errors and errors[0]["error"]["message"] == "boom"
    assert chunks[-1] is None


def test_templates_carry_responses_and_jetstream():
    by_key = {t["key"]: t for t in get_provider_templates()}
    assert by_key["openai-responses"]["protocol"] == "responses"
    assert by_key["openai-responses"]["base_url"] == "https://api.openai.com/v1"
    assert "protocol" not in by_key["jetstream2"]
    assert by_key["jetstream2"]["base_url"] == "https://llm.jetstream-cloud.org/api"
