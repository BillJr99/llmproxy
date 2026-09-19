"""Provenance block naming the model and provider that served a virtual request.

A virtual model like ``llmproxy__deep/free`` stands for a pool, and the request
cycles through it until one candidate answers. Which one did is useful to the
caller: it explains a cost line, a latency spike or a change in answer quality,
and it is the first thing anyone asks when a free-pool answer looks off.

The ``X-LLMProxy-Selected-Model`` response header carries the same fact and is
the channel that works everywhere, including streamed and non-OpenAI-dialect
replies. This module adds the in-body form for the common case, because most
SDK clients surface a parsed body and never expose response headers at all.

The block is *additive*: it occupies a top-level ``llmproxy_route`` key, which
strict OpenAI clients ignore. Nothing here may ever fail a served response, so
every function degrades to the untouched input on error, mirroring
``llmproxy.fusion``.
"""

from __future__ import annotations

import json
import time
import traceback

from .dialects.base import sse_data

# Top-level body key and the ``object`` discriminator inside it.
ROUTE_FIELD = "llmproxy_route"
ROUTE_OBJECT = "route.report"


def build_route_report(
    *,
    virtual: str,
    provider: str,
    model: str,
    route_reason: str | None = None,
    attempt_index: int = 0,
) -> dict:
    """Build the additive ``llmproxy_route`` block.

    ``virtual`` is the pool the client asked for, ``provider``/``model`` the
    candidate that answered. ``attempt_index`` is 0 for the first-ranked
    candidate, so ``failed_over`` distinguishes a ranked pick from one settled
    for after earlier candidates failed.
    """
    return {
        "object": ROUTE_OBJECT,
        "virtual": virtual,
        "provider": provider,
        "model": model,
        "selected_model": f"{provider}/{model}",
        "route_reason": route_reason,
        "attempt": attempt_index,
        "failed_over": attempt_index > 0,
    }


def inject_route_report(body_bytes: bytes, report: dict) -> bytes:
    """Return *body_bytes* (an OpenAI chat completion) with *report* attached.

    On any parse or serialization failure the original bytes are returned
    unchanged, so an edge case costs the caller its provenance rather than its
    answer.
    """
    try:
        data = json.loads(body_bytes)
        if isinstance(data, dict):
            data[ROUTE_FIELD] = report
            return json.dumps(data).encode("utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[route_report:inject_route_report] {e}")
        traceback.print_exc()
    return body_bytes


def route_chunk(report: dict, model_id: str) -> bytes:
    """Encode *report* as one leading ``chat.completion.chunk`` SSE frame.

    A stream has no body to inject into, and rewriting the upstream's own chunks
    would mean parsing and re-serializing every one of them, forfeiting the
    byte-for-byte relay the cycling path depends on. So the provenance rides a
    single synthetic frame ahead of the real ones.

    The frame carries an empty ``choices`` array, which is already the shape
    clients see from the final usage chunk that ``stream_options.include_usage``
    produces, so it needs no special handling on the receiving end. Returns
    ``b""`` on any failure, which simply omits the frame.
    """
    try:
        return sse_data({
            "id": f"chatcmpl-route-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [],
            ROUTE_FIELD: report,
        })
    except Exception as e:  # noqa: BLE001
        print(f"[route_report:route_chunk] {e}")
        traceback.print_exc()
        return b""
