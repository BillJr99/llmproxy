"""Tests for the OpenAI Responses API inbound dialect (``POST /v1/responses``).

Before this dialect existed a Responses request fell through to the generic
``/v1/<subpath>`` passthrough, which resolves a provider directly and has no
cycling engine behind it — so a virtual model like ``llmproxy/free`` could not
be reached from a Responses client at all. These tests cover the three shape
reconciliations that make routing it possible: input items to chat messages,
one chat choice fanning out to several output items, and the typed streaming
event vocabulary.
"""

from __future__ import annotations

import json

import pytest

from llmproxy.dialects import get_inbound
from llmproxy.dialects import responses as R


@pytest.fixture(autouse=True)
def _clean_store():
    R.STORE.clear()
    R._take_pending()
    yield
    R.STORE.clear()


ADAPTER = get_inbound("responses")


def _canonical(body: dict) -> dict:
    return ADAPTER.to_canonical_request(body)


# ── registration ────────────────────────────────────────────────────────────

def test_dialect_is_registered():
    assert get_inbound("responses").name == "responses"


def test_route_exists():
    from llmproxy import server
    rules = {r.rule for r in server.app.url_map.iter_rules()}
    assert "/v1/responses" in rules
    assert "/v1/responses/<response_id>" in rules


# ── request: input shapes ───────────────────────────────────────────────────

def test_string_input_becomes_a_user_message():
    assert _canonical({"model": "m", "input": "hi"})["messages"] == [
        {"role": "user", "content": "hi"}]


def test_instructions_become_a_leading_system_message():
    msgs = _canonical({"model": "m", "instructions": "Be terse.", "input": "hi"})["messages"]
    assert msgs[0] == {"role": "system", "content": "Be terse."}


def test_typed_input_text_parts_flatten():
    msgs = _canonical({"model": "m", "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "a"},
                                     {"type": "input_text", "text": "b"}]}]})["messages"]
    assert msgs == [{"role": "user", "content": "ab"}]


def test_input_image_stays_a_parts_list_so_vision_routing_sees_it():
    from llmproxy import server
    payload = _canonical({"model": "m", "input": [
        {"role": "user", "content": [
            {"type": "input_text", "text": "what is this"},
            {"type": "input_image", "image_url": "http://x/i.png"}]}]})
    assert isinstance(payload["messages"][0]["content"], list)
    assert server._request_has_image(payload) is True


def test_function_call_items_become_assistant_tool_calls():
    msgs = _canonical({"model": "m", "input": [
        {"role": "user", "content": "go"},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a":1}'},
        {"type": "function_call_output", "call_id": "c1", "output": "done"},
    ]})["messages"]
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["id"] == "c1"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "f"
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "done"}


def test_parallel_function_calls_collapse_onto_one_assistant_message():
    msgs = _canonical({"model": "m", "input": [
        {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
        {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"},
    ]})["messages"]
    assert len(msgs) == 1
    assert len(msgs[0]["tool_calls"]) == 2


def test_non_string_function_output_is_serialized():
    msgs = _canonical({"model": "m", "input": [
        {"type": "function_call_output", "call_id": "c", "output": {"ok": True}}]})["messages"]
    assert json.loads(msgs[0]["content"]) == {"ok": True}


def test_reasoning_items_are_dropped():
    msgs = _canonical({"model": "m", "input": [
        {"type": "reasoning", "summary": []},
        {"role": "user", "content": "hi"}]})["messages"]
    assert msgs == [{"role": "user", "content": "hi"}]


# ── request: parameters ─────────────────────────────────────────────────────

def test_max_output_tokens_maps_to_max_tokens():
    assert _canonical({"model": "m", "input": "x", "max_output_tokens": 64})["max_tokens"] == 64


def test_reasoning_effort_maps_through():
    p = _canonical({"model": "m", "input": "x", "reasoning": {"effort": "high"}})
    assert p["reasoning_effort"] == "high"


def test_flat_tools_become_nested_chat_tools():
    p = _canonical({"model": "m", "input": "x", "tools": [
        {"type": "function", "name": "f", "description": "d",
         "parameters": {"type": "object"}}]})
    assert p["tools"] == [{"type": "function", "function": {
        "name": "f", "description": "d", "parameters": {"type": "object"}}}]


def test_builtin_tools_are_dropped():
    """Nothing behind llmproxy can run OpenAI's server-side tools."""
    p = _canonical({"model": "m", "input": "x", "tools": [
        {"type": "web_search"}, {"type": "function", "name": "f", "parameters": {}}]})
    assert len(p["tools"]) == 1


def test_tool_choice_function_is_renested():
    p = _canonical({"model": "m", "input": "x",
                    "tools": [{"type": "function", "name": "f", "parameters": {}}],
                    "tool_choice": {"type": "function", "name": "f"}})
    assert p["tool_choice"] == {"type": "function", "function": {"name": "f"}}


def test_text_format_json_schema_becomes_response_format():
    p = _canonical({"model": "m", "input": "x", "text": {"format": {
        "type": "json_schema", "name": "S", "schema": {"type": "object"}}}})
    assert p["response_format"]["type"] == "json_schema"
    assert p["response_format"]["json_schema"]["name"] == "S"


# ── response rendering ──────────────────────────────────────────────────────

def _chat(content="hi", tool_calls=None, finish="stop", usage=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    body = {"model": "up/m", "created": 7,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}]}
    if usage:
        body["usage"] = usage
    return json.dumps(body).encode()


def test_render_response_shape():
    out = json.loads(ADAPTER.render_response(_chat()))
    assert out["object"] == "response"
    assert out["status"] == "completed"
    assert out["output"][0]["type"] == "message"
    assert out["output"][0]["content"][0]["type"] == "output_text"
    assert out["output_text"] == "hi"


def test_render_response_fans_out_tool_calls_as_items():
    out = json.loads(ADAPTER.render_response(_chat(
        content="calling", finish="tool_calls",
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}}])))
    kinds = [i["type"] for i in out["output"]]
    assert kinds == ["message", "function_call"]
    assert out["output"][1]["call_id"] == "c1"


def test_render_response_maps_usage_names():
    out = json.loads(ADAPTER.render_response(_chat(
        usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})))
    assert out["usage"] == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}


def test_render_response_marks_truncation_incomplete():
    out = json.loads(ADAPTER.render_response(_chat(finish="length")))
    assert out["status"] == "incomplete"
    assert out["incomplete_details"] == {"reason": "max_output_tokens"}


def test_render_response_passes_upstream_errors_through():
    body = json.dumps({"error": {"message": "nope"}}).encode()
    assert json.loads(ADAPTER.render_response(body))["error"]["message"] == "nope"


# ── streaming ───────────────────────────────────────────────────────────────

def _chunks(*deltas, finish="stop", usage=None):
    out = [{"model": "up/m", "choices": [{"index": 0, "delta": d}]} for d in deltas]
    tail = {"model": "up/m", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    if usage:
        tail["usage"] = usage
    out.append(tail)
    out.append(None)
    return iter(out)


def _events(raw: bytes) -> list[str]:
    return [line[7:] for line in raw.decode().splitlines() if line.startswith("event: ")]


def _datas(raw: bytes) -> list[dict]:
    return [json.loads(line[6:]) for line in raw.decode().splitlines()
            if line.startswith("data: ")]


def test_stream_emits_the_text_lifecycle_in_order():
    raw = b"".join(ADAPTER.render_stream(_chunks({"role": "assistant"}, {"content": "he"},
                                                 {"content": "llo"})))
    assert _events(raw) == [
        "response.created", "response.in_progress",
        "response.output_item.added", "response.content_part.added",
        "response.output_text.delta", "response.output_text.delta",
        "response.output_text.done", "response.content_part.done",
        "response.output_item.done", "response.completed",
    ]


def test_stream_sequence_numbers_are_monotonic():
    raw = b"".join(ADAPTER.render_stream(_chunks({"content": "a"}, {"content": "b"})))
    seqs = [d["sequence_number"] for d in _datas(raw)]
    assert seqs == list(range(len(seqs)))


def test_stream_reassembles_tool_call_arguments():
    raw = b"".join(ADAPTER.render_stream(_chunks(
        {"tool_calls": [{"index": 0, "id": "c9", "function": {"name": "f", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"p":'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '1}'}}]},
        finish="tool_calls")))
    final = _datas(raw)[-1]["response"]
    call = [i for i in final["output"] if i["type"] == "function_call"][0]
    assert call["arguments"] == '{"p":1}'
    assert call["call_id"] == "c9"
    assert "response.function_call_arguments.done" in _events(raw)


def test_stream_final_event_carries_usage():
    raw = b"".join(ADAPTER.render_stream(_chunks(
        {"content": "x"}, usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})))
    assert _datas(raw)[-1]["response"]["usage"]["total_tokens"] == 3


def test_stream_truncation_emits_response_incomplete():
    raw = b"".join(ADAPTER.render_stream(_chunks({"content": "x"}, finish="length")))
    assert _events(raw)[-1] == "response.incomplete"


def test_stream_done_sentinel_has_no_responses_analogue():
    raw = b"".join(ADAPTER.render_stream(iter([None])))
    assert b"[DONE]" not in raw


# ── statefulness ────────────────────────────────────────────────────────────

def test_previous_response_id_replays_the_transcript():
    _canonical({"model": "m", "instructions": "Be terse.", "input": "hello"})
    first = json.loads(ADAPTER.render_response(_chat(content="hi")))
    again = _canonical({"model": "m", "previous_response_id": first["id"], "input": "more?"})
    assert [m["role"] for m in again["messages"]] == ["system", "user", "assistant", "user"]


def test_unknown_previous_response_id_raises_rather_than_guessing():
    with pytest.raises(R.UnknownPreviousResponse):
        _canonical({"model": "m", "previous_response_id": "resp_missing", "input": "x"})


def test_store_false_opts_out():
    _canonical({"model": "m", "input": "secret", "store": False})
    out = json.loads(ADAPTER.render_response(_chat()))
    assert R.STORE.get(out["id"]) is None


def test_store_is_bounded():
    store = R._ResponseStore(limit=3)
    for i in range(5):
        store.save(f"r{i}", [{"role": "user", "content": str(i)}])
    assert store.get("r0") is None and store.get("r1") is None
    assert store.get("r4") is not None


def test_store_delete():
    store = R._ResponseStore()
    store.save("r", [])
    assert store.delete("r") is True
    assert store.delete("r") is False


def test_streaming_turn_is_stored_before_the_terminal_event():
    """A client may follow up the instant it sees response.completed."""
    _canonical({"model": "m", "input": "hi"})
    raw = b"".join(ADAPTER.render_stream(_chunks({"content": "yo"})))
    rid = _datas(raw)[-1]["response"]["id"]
    assert R.STORE.get(rid) is not None
