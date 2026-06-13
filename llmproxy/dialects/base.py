"""Dialect translation layer — interfaces, registry, and SSE helpers.

llmproxy's internal "canonical" representation is the OpenAI chat/completions
schema: requests look like ``{"model", "messages", "stream", ...}`` and
responses like ``{"choices": [{"message"/"delta", "finish_reason"}], "usage"}``.

Two adapter axes sit at the edges of the proxy:

* **Inbound** — what the *client* speaks. Selected by the route
  (``/v1/chat/completions`` → ``openai``; ``/v1/messages`` → ``anthropic``).
  Converts the client request to canonical form and renders the canonical
  response back into the client's dialect.
* **Outbound** — what the *upstream provider* speaks. Selected by the provider
  config field ``"protocol"`` (``openai`` default, ``anthropic``, ``gemini``).
  Builds the provider-native request and parses the native response back to
  canonical form.

The common case (openai inbound + openai upstream) uses identity adapters, so
behavior is byte-for-byte identical to the original passthrough; translation
cost is paid only when a dialect actually differs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def iter_sse_events(byte_iter: Iterable[bytes]) -> Iterator[tuple[str | None, str]]:
    """Parse a raw SSE byte stream into ``(event, data)`` tuples.

    Buffers across chunk boundaries and joins multi-line ``data:`` fields with
    newlines, per the SSE spec. ``event`` is ``None`` when the event carried no
    explicit ``event:`` field (the OpenAI/Gemini ``data:``-only style).
    """
    buf = ""
    event: str | None = None
    data_lines: list[str] = []
    for chunk in byte_iter:
        if not chunk:
            continue
        buf += chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")
            if line == "":
                if data_lines:
                    yield event, "\n".join(data_lines)
                event = None
                data_lines = []
                continue
            if line.startswith(":"):
                continue  # comment / heartbeat
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip(" "))
    if data_lines:
        yield event, "\n".join(data_lines)


def iter_canonical_json_chunks(byte_iter: Iterable[bytes]) -> Iterator[dict | None]:
    """Parse an OpenAI-style ``data: {json}`` SSE stream into chunk dicts.

    Yields parsed chunk dicts; yields ``None`` for the terminal ``[DONE]``
    sentinel. Non-JSON data lines are skipped.
    """
    for _event, data in iter_sse_events(byte_iter):
        if data == "[DONE]":
            yield None
            continue
        try:
            yield json.loads(data)
        except (ValueError, TypeError):
            continue


def sse_data(obj) -> bytes:
    """Encode an object as a single ``data: {json}`` SSE frame."""
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n\n"


def sse_event(event: str, obj) -> bytes:
    """Encode a named SSE frame (``event: <type>`` + ``data: {json}``)."""
    return (
        f"event: {event}\n".encode()
        + b"data: " + json.dumps(obj, separators=(",", ":")).encode("utf-8")
        + b"\n\n"
    )


# ---------------------------------------------------------------------------
# Adapter interfaces
# ---------------------------------------------------------------------------

class OutboundAdapter:
    """Translate canonical OpenAI ⇄ a provider's native protocol."""

    name: str = "openai"
    is_identity: bool = False

    def build_request(
        self,
        endpoint: str,
        base_url: str,
        provider_cfg: dict,
        payload: dict,
        *,
        stream: bool,
        forwarded_headers: dict,
    ) -> tuple[str, dict, dict]:
        """Return ``(url, headers, body)`` for the upstream POST."""
        raise NotImplementedError

    def translate_response(self, content: bytes) -> bytes:
        """Convert a native non-streaming response body to canonical OpenAI."""
        return content

    def parse_stream(self, byte_iter: Iterable[bytes]) -> Iterator[dict | None]:
        """Parse a native SSE byte stream into canonical OpenAI chunk dicts."""
        return iter_canonical_json_chunks(byte_iter)


class InboundAdapter:
    """Translate a client dialect ⇄ canonical OpenAI."""

    name: str = "openai"
    is_identity: bool = False

    def to_canonical_request(self, body: dict) -> dict:
        """Convert a client request body to a canonical OpenAI payload."""
        return body

    def render_response(self, canonical: bytes) -> bytes:
        """Render a canonical OpenAI response body into the client dialect."""
        return canonical

    def render_stream(self, chunks: Iterator[dict | None]) -> Iterator[bytes]:
        """Render canonical OpenAI chunk dicts into client-dialect SSE bytes."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_OUTBOUND: dict[str, OutboundAdapter] = {}
_INBOUND: dict[str, InboundAdapter] = {}


def register_outbound(adapter: OutboundAdapter) -> None:
    _OUTBOUND[adapter.name] = adapter


def register_inbound(adapter: InboundAdapter) -> None:
    _INBOUND[adapter.name] = adapter


def get_outbound(protocol: str | None) -> OutboundAdapter:
    """Return the outbound adapter for a provider ``protocol`` (default openai)."""
    return _OUTBOUND.get((protocol or "openai").lower(), _OUTBOUND["openai"])


def get_inbound(dialect: str | None) -> InboundAdapter:
    """Return the inbound adapter for a client ``dialect`` (default openai)."""
    return _INBOUND.get((dialect or "openai").lower(), _INBOUND["openai"])
