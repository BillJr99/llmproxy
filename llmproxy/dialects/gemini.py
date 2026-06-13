"""Outbound adapter for Google Gemini's native ``generateContent`` protocol.

Translates the canonical OpenAI chat schema to/from Gemini's
``models/{model}:generateContent`` (and ``:streamGenerateContent?alt=sse``)
shape, covering text, tool/function calls, and token usage.
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
    sse_data,
)

_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
}

# OpenAI finish_reason -> Gemini finishReason (inbound rendering).
_FINISH_MAP_OUT = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "tool_calls": "STOP",
    "content_filter": "SAFETY",
    None: "STOP",
}


def _text_of(content) -> str:
    """Flatten OpenAI message content (str or list of parts) to plain text."""
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


def _to_gemini_request(payload: dict) -> dict:
    """Build a Gemini generateContent body from a canonical OpenAI payload."""
    contents: list[dict] = []
    system_parts: list[dict] = []
    for msg in payload.get("messages", []):
        role = msg.get("role")
        if role == "system":
            system_parts.append({"text": _text_of(msg.get("content"))})
            continue
        if role == "tool":
            # OpenAI tool result -> Gemini functionResponse (carried on a user turn).
            try:
                resp = json.loads(msg.get("content") or "null")
            except (ValueError, TypeError):
                resp = msg.get("content")
            contents.append({
                "role": "user",
                "parts": [{"functionResponse": {
                    "name": msg.get("name") or msg.get("tool_call_id") or "tool",
                    "response": {"result": resp},
                }}],
            })
            continue
        parts: list[dict] = []
        text = _text_of(msg.get("content"))
        if text:
            parts.append({"text": text})
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {}
            parts.append({"functionCall": {"name": fn.get("name"), "args": args}})
        if not parts:
            parts.append({"text": ""})
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})

    body: dict = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}

    gen: dict = {}
    max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
    if max_tokens is not None:
        gen["maxOutputTokens"] = max_tokens
    if payload.get("temperature") is not None:
        gen["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        gen["topP"] = payload["top_p"]
    if payload.get("stop") is not None:
        stop = payload["stop"]
        gen["stopSequences"] = [stop] if isinstance(stop, str) else stop
    if gen:
        body["generationConfig"] = gen

    decls = []
    for tool in payload.get("tools") or []:
        if tool.get("type") == "function" and tool.get("function"):
            fn = tool["function"]
            decl = {"name": fn.get("name")}
            if fn.get("description"):
                decl["description"] = fn["description"]
            if fn.get("parameters"):
                decl["parameters"] = fn["parameters"]
            decls.append(decl)
    if decls:
        body["tools"] = [{"functionDeclarations": decls}]
    return body


def _parts_to_message(parts: list[dict]) -> tuple[str, list[dict]]:
    """Split Gemini content parts into (text, tool_calls)."""
    text_out = []
    tool_calls = []
    for part in parts or []:
        if "text" in part:
            text_out.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append({
                "id": f"call_{len(tool_calls)}_{fc.get('name', 'fn')}",
                "type": "function",
                "function": {
                    "name": fc.get("name"),
                    "arguments": json.dumps(fc.get("args", {}), separators=(",", ":")),
                },
            })
    return "".join(text_out), tool_calls


def _usage(meta: dict | None) -> dict | None:
    if not meta:
        return None
    return {
        "prompt_tokens": meta.get("promptTokenCount", 0),
        "completion_tokens": meta.get("candidatesTokenCount", 0),
        "total_tokens": meta.get("totalTokenCount", 0),
    }


class GeminiOutbound(OutboundAdapter):
    name = "gemini"

    def build_request(self, endpoint, base_url, provider_cfg, payload, *, stream, forwarded_headers):
        model = payload.get("model", "")
        verb = "streamGenerateContent" if stream else "generateContent"
        url = f"{base_url}/models/{model}:{verb}"
        if stream:
            url += "?alt=sse"
        headers = {"Content-Type": "application/json"}
        api_key = provider_api_key(provider_cfg)
        if api_key:
            headers["x-goog-api-key"] = api_key
        return url, headers, _to_gemini_request(payload)

    def translate_response(self, content: bytes) -> bytes:
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return content
        candidates = data.get("candidates") or [{}]
        cand = candidates[0]
        text, tool_calls = _parts_to_message((cand.get("content") or {}).get("parts", []))
        message: dict = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish = _FINISH_MAP.get(cand.get("finishReason", "STOP"), "stop")
        if tool_calls:
            finish = "tool_calls"
        out = {
            "id": data.get("responseId", "chatcmpl-gemini"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("modelVersion", ""),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }
        usage = _usage(data.get("usageMetadata"))
        if usage:
            out["usage"] = usage
        return json.dumps(out).encode("utf-8")

    def parse_stream(self, byte_iter: Iterable[bytes]) -> Iterator[dict | None]:
        sent_role = False
        for _event, data in iter_sse_events(byte_iter):
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except (ValueError, TypeError):
                continue
            cand = (obj.get("candidates") or [{}])[0]
            text, tool_calls = _parts_to_message((cand.get("content") or {}).get("parts", []))
            delta: dict = {}
            if not sent_role:
                delta["role"] = "assistant"
                sent_role = True
            if text:
                delta["content"] = text
            if tool_calls:
                delta["tool_calls"] = [
                    {"index": i, **tc} for i, tc in enumerate(tool_calls)
                ]
            chunk: dict = {
                "object": "chat.completion.chunk",
                "model": obj.get("modelVersion", ""),
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            finish = cand.get("finishReason")
            if finish:
                chunk["choices"][0]["finish_reason"] = (
                    "tool_calls" if tool_calls else _FINISH_MAP.get(finish, "stop")
                )
            usage = _usage(obj.get("usageMetadata"))
            if usage:
                chunk["usage"] = usage
            yield chunk
        yield None


register_outbound(GeminiOutbound())


# ---------------------------------------------------------------------------
# Gemini generateContent (request)  ->  canonical OpenAI   [inbound]
# ---------------------------------------------------------------------------

def _gemini_to_openai_request(body: dict) -> dict:
    messages: list[dict] = []
    system = body.get("systemInstruction") or body.get("system_instruction")
    if system:
        text = "".join(p.get("text", "") for p in system.get("parts", []))
        if text:
            messages.append({"role": "system", "content": text})

    for content in body.get("contents", []):
        gem_role = content.get("role", "user")
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_msgs: list[dict] = []
        for part in content.get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{fc.get('name', 'fn')}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name"),
                        "arguments": json.dumps(fc.get("args", {}), separators=(",", ":")),
                    },
                })
            elif "functionResponse" in part:
                fr = part["functionResponse"]
                tool_msgs.append({
                    "role": "tool",
                    "tool_call_id": f"call_{fr.get('name', 'fn')}",
                    "name": fr.get("name"),
                    "content": json.dumps(fr.get("response", {}), separators=(",", ":")),
                })
        if gem_role == "model":
            m: dict = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                m["tool_calls"] = tool_calls
            messages.append(m)
        else:
            if text_parts:
                messages.append({"role": "user", "content": "".join(text_parts)})
            messages.extend(tool_msgs)

    payload: dict = {"messages": messages}
    gen = body.get("generationConfig") or body.get("generation_config") or {}
    if gen.get("maxOutputTokens") is not None:
        payload["max_tokens"] = gen["maxOutputTokens"]
    if gen.get("temperature") is not None:
        payload["temperature"] = gen["temperature"]
    if gen.get("topP") is not None:
        payload["top_p"] = gen["topP"]
    if gen.get("stopSequences"):
        payload["stop"] = gen["stopSequences"]
    tools: list[dict] = []
    for tool in body.get("tools") or []:
        for decl in tool.get("functionDeclarations", []):
            tools.append({"type": "function", "function": {
                "name": decl.get("name"),
                "description": decl.get("description", ""),
                "parameters": decl.get("parameters", {"type": "object", "properties": {}}),
            }})
    if tools:
        payload["tools"] = tools
    return payload


def _openai_to_gemini_response(data: dict) -> dict:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    parts: list[dict] = []
    if message.get("content"):
        parts.append({"text": message["content"]})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        parts.append({"functionCall": {"name": fn.get("name"), "args": args}})
    usage = data.get("usage") or {}
    out = {
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": _FINISH_MAP_OUT.get(choice.get("finish_reason"), "STOP"),
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
    }
    if data.get("model"):
        out["modelVersion"] = data["model"]
    return out


class GeminiInbound(InboundAdapter):
    """Inbound Google Gemini generateContent surface.

    Exposed at /v1beta/models/{model}:generateContent (+ :streamGenerateContent,
    :countTokens), so the Google GenAI SDK can point at llmproxy. The model id is
    carried in the URL path, so the route injects it into the canonical payload.
    """

    name = "gemini"

    def to_canonical_request(self, body: dict) -> dict:
        return _gemini_to_openai_request(body)

    def render_response(self, canonical: bytes) -> bytes:
        try:
            data = json.loads(canonical)
        except (ValueError, TypeError):
            return canonical
        if "error" in data and "choices" not in data:
            return canonical
        return json.dumps(_openai_to_gemini_response(data)).encode("utf-8")

    def render_stream(self, chunks):
        tool_acc: dict[int, dict] = {}
        finish = None
        usage: dict | None = None
        model = ""
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
            content = delta.get("content")
            if content:
                yield sse_data({"candidates": [{
                    "content": {"role": "model", "parts": [{"text": content}]}, "index": 0}]})
            for tc in delta.get("tool_calls") or []:
                acc = tool_acc.setdefault(tc.get("index", 0), {"name": None, "args": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["name"] = fn["name"]
                if fn.get("arguments"):
                    acc["args"] += fn["arguments"]
        # Final chunk: any buffered tool calls, finishReason, and usage.
        parts: list[dict] = []
        for i in sorted(tool_acc):
            acc = tool_acc[i]
            try:
                args = json.loads(acc["args"] or "{}")
            except (ValueError, TypeError):
                args = {}
            parts.append({"functionCall": {"name": acc["name"], "args": args}})
        final: dict = {"candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": _FINISH_MAP_OUT.get(finish, "STOP"), "index": 0}]}
        if usage:
            final["usageMetadata"] = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            }
        if model:
            final["modelVersion"] = model
        yield sse_data(final)


register_inbound(GeminiInbound())
