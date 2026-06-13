"""Anthropic Messages dialect — inbound and outbound.

* **Inbound** powers the ``/v1/messages`` route: an Anthropic Messages request
  is converted to the canonical OpenAI schema (so all existing routing, virtual
  models, and usage accounting apply unchanged), and the canonical response is
  rendered back into the Anthropic shape — including the Anthropic streaming
  event format.
* **Outbound** lets a provider with ``"protocol": "anthropic"`` be reached over
  its native ``/messages`` endpoint (``x-api-key`` + ``anthropic-version``),
  translating the canonical OpenAI request/response/stream to/from Messages.

Covers text, tool definitions + tool calls + tool results, and token usage.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator

from ..config import provider_api_key
from .base import (
    InboundAdapter,
    OutboundAdapter,
    iter_sse_events,
    register_inbound,
    register_outbound,
    sse_event,
)

ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
    None: "end_turn",
}
_STOP_TO_FINISH = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
    "refusal": "stop",
    None: "stop",
}


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text", ""))
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return "" if content is None else str(content)


def _system_text(system) -> str:
    """Anthropic ``system`` may be a string or a list of text blocks."""
    if isinstance(system, list):
        return "".join(
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
        )
    return system or ""


# ---------------------------------------------------------------------------
# canonical OpenAI  ->  Anthropic Messages (request body)
# ---------------------------------------------------------------------------

def _openai_to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system_chunks: list[str] = []
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_chunks.append(_text_of(msg.get("content")))
            continue
        if role == "tool":
            out.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": _text_of(msg.get("content")),
            }]})
            continue
        if role == "assistant":
            blocks: list[dict] = []
            text = _text_of(msg.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (ValueError, TypeError):
                    args = {}
                blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                               "name": fn.get("name"), "input": args})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            continue
        # user (and any other role) -> user text block
        out.append({"role": "user", "content": [{"type": "text", "text": _text_of(msg.get("content"))}]})
    return "\n".join(c for c in system_chunks if c), out


def _to_anthropic_request(payload: dict) -> dict:
    system, messages = _openai_to_anthropic_messages(payload.get("messages", []))
    body: dict = {
        "model": payload.get("model", ""),
        "max_tokens": payload.get("max_tokens") or payload.get("max_completion_tokens")
        or _DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    if system:
        body["system"] = system
    if payload.get("temperature") is not None:
        body["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        body["top_p"] = payload["top_p"]
    if payload.get("stop") is not None:
        stop = payload["stop"]
        body["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    if payload.get("stream"):
        body["stream"] = True
    tools = []
    for tool in payload.get("tools") or []:
        if tool.get("type") == "function" and tool.get("function"):
            fn = tool["function"]
            tools.append({
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
    if tools:
        body["tools"] = tools
    return body


# ---------------------------------------------------------------------------
# Anthropic Messages (response)  <->  canonical OpenAI (response)
# ---------------------------------------------------------------------------

def _anthropic_response_to_openai(data: dict) -> dict:
    text_out = []
    tool_calls = []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            text_out.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input", {}), separators=(",", ":")),
                },
            })
    message: dict = {"role": "assistant", "content": "".join(text_out) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = _STOP_TO_FINISH.get(data.get("stop_reason"), "stop")
    usage_in = data.get("usage") or {}
    out = {
        "id": data.get("id", "chatcmpl-anthropic"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", ""),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": usage_in.get("input_tokens", 0),
            "completion_tokens": usage_in.get("output_tokens", 0),
            "total_tokens": usage_in.get("input_tokens", 0) + usage_in.get("output_tokens", 0),
        },
    }
    return out


def _openai_response_to_anthropic(data: dict) -> dict:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content_blocks: list[dict] = []
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            inp = {}
        content_blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                               "name": fn.get("name"), "input": inp})
    usage = data.get("usage") or {}
    return {
        "id": data.get("id", "msg_proxy"),
        "type": "message",
        "role": "assistant",
        "model": data.get("model", ""),
        "content": content_blocks,
        "stop_reason": _FINISH_TO_STOP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Anthropic Messages (request)  ->  canonical OpenAI (request)   [inbound]
# ---------------------------------------------------------------------------

def _anthropic_to_openai_request(body: dict) -> dict:
    messages: list[dict] = []
    system = _system_text(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for block in content or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {}), separators=(",", ":")),
                    },
                })
            elif btype == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _text_of(block.get("content")),
                })
        if role == "assistant":
            m: dict = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                m["tool_calls"] = tool_calls
            messages.append(m)
        else:  # user
            if text_parts:
                messages.append({"role": "user", "content": "".join(text_parts)})
            messages.extend(tool_results)

    payload: dict = {"model": body.get("model", ""), "messages": messages}
    if body.get("max_tokens") is not None:
        payload["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]
    if body.get("stream"):
        payload["stream"] = True
    tools = []
    for tool in body.get("tools") or []:
        tools.append({"type": "function", "function": {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        }})
    if tools:
        payload["tools"] = tools
        if isinstance(body.get("tool_choice"), dict):
            tc = body["tool_choice"]
            if tc.get("type") == "tool" and tc.get("name"):
                payload["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
            elif tc.get("type") in ("any", "auto"):
                payload["tool_choice"] = "required" if tc["type"] == "any" else "auto"
    return payload


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class AnthropicOutbound(OutboundAdapter):
    name = "anthropic"

    def build_request(self, endpoint, base_url, provider_cfg, payload, *, stream, forwarded_headers):
        url = f"{base_url}/messages"
        headers = {"Content-Type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
        api_key = provider_api_key(provider_cfg)
        if api_key:
            headers["x-api-key"] = api_key
        return url, headers, _to_anthropic_request(payload)

    def translate_response(self, content: bytes) -> bytes:
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return content
        if data.get("type") == "error":
            return content  # leave upstream error bodies intact
        return json.dumps(_anthropic_response_to_openai(data)).encode("utf-8")

    def parse_stream(self, byte_iter: Iterable[bytes]) -> Iterator[dict | None]:
        model = ""
        finish = None
        usage: dict = {}
        block_tool: dict[int, int] = {}      # anthropic block index -> openai tool index
        next_tool = 0
        for event, data in iter_sse_events(byte_iter):
            try:
                obj = json.loads(data)
            except (ValueError, TypeError):
                continue
            etype = event or obj.get("type")
            if etype == "message_start":
                msg = obj.get("message", {})
                model = msg.get("model", "")
                usage.update(msg.get("usage") or {})
                yield {"object": "chat.completion.chunk", "model": model,
                       "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            elif etype == "content_block_start":
                block = obj.get("content_block", {})
                if block.get("type") == "tool_use":
                    idx = obj.get("index", 0)
                    oai_i = next_tool
                    block_tool[idx] = oai_i
                    next_tool += 1
                    yield {"object": "chat.completion.chunk", "model": model, "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": [{"index": oai_i, "id": block.get("id", ""),
                                                  "type": "function",
                                                  "function": {"name": block.get("name"), "arguments": ""}}]},
                        "finish_reason": None}]}
            elif etype == "content_block_delta":
                delta = obj.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield {"object": "chat.completion.chunk", "model": model, "choices": [{
                        "index": 0, "delta": {"content": delta.get("text", "")}, "finish_reason": None}]}
                elif delta.get("type") == "input_json_delta":
                    oai_i = block_tool.get(obj.get("index", 0), 0)
                    yield {"object": "chat.completion.chunk", "model": model, "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": [{"index": oai_i,
                                                  "function": {"arguments": delta.get("partial_json", "")}}]},
                        "finish_reason": None}]}
            elif etype == "message_delta":
                finish = _STOP_TO_FINISH.get((obj.get("delta") or {}).get("stop_reason"), "stop")
                usage.update(obj.get("usage") or {})
            elif etype == "message_stop":
                break
        final = {"object": "chat.completion.chunk", "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": finish or "stop"}]}
        if usage:
            final["usage"] = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            }
        yield final
        yield None


class AnthropicInbound(InboundAdapter):
    name = "anthropic"

    def to_canonical_request(self, body: dict) -> dict:
        return _anthropic_to_openai_request(body)

    def render_response(self, canonical: bytes) -> bytes:
        try:
            data = json.loads(canonical)
        except (ValueError, TypeError):
            return canonical
        if "error" in data and "choices" not in data:
            return canonical
        return json.dumps(_openai_response_to_anthropic(data)).encode("utf-8")

    def render_stream(self, chunks: Iterator[dict | None]) -> Iterator[bytes]:
        started = False
        text_open = False
        text_index = None
        next_index = 0
        tool_index: dict[int, int] = {}   # openai tool index -> anthropic block index
        model = ""
        finish = None
        usage: dict = {}
        msg_id = f"msg_{int(time.time()*1000):x}"

        def _start():
            nonlocal started
            started = True
            return sse_event("message_start", {"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": 0}}})

        for chunk in chunks:
            if chunk is None:
                break
            model = chunk.get("model") or model
            if chunk.get("usage"):
                usage = chunk["usage"]
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            if not started:
                yield _start()
            content = delta.get("content")
            if content:
                if not text_open:
                    text_index = next_index
                    next_index += 1
                    text_open = True
                    yield sse_event("content_block_start", {
                        "type": "content_block_start", "index": text_index,
                        "content_block": {"type": "text", "text": ""}})
                yield sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": text_index,
                    "delta": {"type": "text_delta", "text": content}})
            for tc in delta.get("tool_calls") or []:
                oai_i = tc.get("index", 0)
                fn = tc.get("function") or {}
                if oai_i not in tool_index:
                    ai = next_index
                    next_index += 1
                    tool_index[oai_i] = ai
                    yield sse_event("content_block_start", {
                        "type": "content_block_start", "index": ai,
                        "content_block": {"type": "tool_use", "id": tc.get("id") or f"toolu_{ai}",
                                          "name": fn.get("name"), "input": {}}})
                ai = tool_index[oai_i]
                if fn.get("arguments"):
                    yield sse_event("content_block_delta", {
                        "type": "content_block_delta", "index": ai,
                        "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}})

        if not started:
            yield _start()
        if text_open:
            yield sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index})
        for ai in sorted(tool_index.values()):
            yield sse_event("content_block_stop", {"type": "content_block_stop", "index": ai})
        yield sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": _FINISH_TO_STOP.get(finish, "end_turn"), "stop_sequence": None},
            "usage": {"output_tokens": usage.get("completion_tokens", 0)}})
        yield sse_event("message_stop", {"type": "message_stop"})


register_outbound(AnthropicOutbound())
register_inbound(AnthropicInbound())
