"""Identity adapters — OpenAI is the canonical dialect.

The outbound adapter reproduces the original passthrough exactly (``{base_url}/
{endpoint}`` with ``Authorization: Bearer``). The inbound adapter is a no-op for
requests/responses but still provides a real ``render_stream`` so that an
OpenAI-speaking client can sit in front of a *non*-OpenAI upstream (the
canonical chunks are re-encoded as standard ``data:`` SSE frames).

This module also defines the ``openai-completions`` *inbound* dialect that
backs the legacy ``/v1/completions`` fallback: it wraps a legacy ``prompt`` as a
single chat message on the way in, and renders the canonical chat response (and
stream) back into the legacy ``text_completion`` shape on the way out. The
``/v1/completions`` route only reaches for it when the upstream lacks a native
legacy endpoint (see ``server.completions``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from ..config import provider_api_key
from .base import InboundAdapter, OutboundAdapter, register_inbound, register_outbound, sse_data


class OpenAIOutbound(OutboundAdapter):
    name = "openai"
    is_identity = True

    def build_request(self, endpoint, base_url, provider_cfg, payload, *, stream, forwarded_headers):
        url = f"{base_url}/{endpoint}"
        headers = {"Content-Type": "application/json", **forwarded_headers}
        api_key = provider_api_key(provider_cfg)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return url, headers, payload


class OpenAIInbound(InboundAdapter):
    name = "openai"
    is_identity = True

    def render_stream(self, chunks: Iterator[dict | None]) -> Iterator[bytes]:
        for chunk in chunks:
            if chunk is None:
                yield b"data: [DONE]\n\n"
            else:
                yield sse_data(chunk)


# ---------------------------------------------------------------------------
# Legacy /v1/completions inbound (prompt <-> chat) — fallback dialect
# ---------------------------------------------------------------------------

# Request keys that carry over unchanged from a legacy completion to a chat
# request. Legacy-only keys the chat schema rejects or reinterprets (``prompt``,
# ``suffix``, ``echo``, ``best_of``, and the integer ``logprobs``) are dropped.
_LEGACY_PASSTHROUGH_KEYS = (
    "model", "stream", "stream_options", "temperature", "top_p", "n", "stop",
    "presence_penalty", "frequency_penalty", "logit_bias", "user", "seed",
    "max_tokens", "max_completion_tokens",
)


def _prompt_to_text(prompt) -> str:
    """Flatten a legacy ``prompt`` (string or array of strings) to one string.

    OpenAI's legacy endpoint also accepts token-id arrays, which can't be
    decoded back to text without the model's tokenizer; those elements are
    skipped rather than guessed at.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, list):
                continue  # token-id array — no tokenizer to decode it
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if prompt is None else str(prompt)


def _legacy_to_openai_request(body: dict) -> dict:
    payload = {k: body[k] for k in _LEGACY_PASSTHROUGH_KEYS if k in body}
    payload["messages"] = [{"role": "user", "content": _prompt_to_text(body.get("prompt", ""))}]
    return payload


def _openai_response_to_legacy(data: dict) -> dict:
    choices = []
    for choice in data.get("choices") or []:
        message = choice.get("message") or {}
        choices.append({
            "text": message.get("content") or "",
            "index": choice.get("index", 0),
            "logprobs": None,
            "finish_reason": choice.get("finish_reason"),
        })
    if not choices:
        choices = [{"text": "", "index": 0, "logprobs": None, "finish_reason": None}]
    out = {
        "id": data.get("id", "cmpl-proxy"),
        "object": "text_completion",
        "created": data.get("created", int(time.time())),
        "model": data.get("model", ""),
        "choices": choices,
    }
    if data.get("usage"):
        out["usage"] = data["usage"]
    if data.get("system_fingerprint"):
        out["system_fingerprint"] = data["system_fingerprint"]
    return out


class OpenAILegacyCompletionsInbound(InboundAdapter):
    """Legacy ``/v1/completions`` surface rendered over canonical chat.

    The request's ``prompt`` becomes a single user message so it can flow
    through the ordinary chat routing (virtual models, cycling, native
    upstreams), and the canonical chat response/stream is rendered back into the
    ``text_completion`` shape the legacy client expects.
    """

    name = "openai-completions"

    def to_canonical_request(self, body: dict) -> dict:
        return _legacy_to_openai_request(body)

    def render_response(self, canonical: bytes) -> bytes:
        try:
            data = json.loads(canonical)
        except (ValueError, TypeError):
            return canonical
        if "error" in data and "choices" not in data:
            return canonical  # leave upstream error bodies intact
        return json.dumps(_openai_response_to_legacy(data)).encode("utf-8")

    def render_stream(self, chunks: Iterator[dict | None]) -> Iterator[bytes]:
        model = ""
        created = int(time.time())
        cmpl_id = f"cmpl-{created:x}"
        for chunk in chunks:
            if chunk is None:
                break
            model = chunk.get("model") or model
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
            finish = choice.get("finish_reason")
            # A frame per content delta, plus the terminal frame that carries the
            # finish_reason. Usage-only chunks (empty choices/delta) fall through.
            if text or finish is not None:
                yield sse_data({
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish,
                    }],
                })
        yield b"data: [DONE]\n\n"


register_outbound(OpenAIOutbound())
register_inbound(OpenAIInbound())
register_inbound(OpenAILegacyCompletionsInbound())
