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
from .base import OutboundAdapter, iter_sse_events, register_outbound

_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
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
