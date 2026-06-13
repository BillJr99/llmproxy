"""Identity adapters — OpenAI is the canonical dialect.

The outbound adapter reproduces the original passthrough exactly (``{base_url}/
{endpoint}`` with ``Authorization: Bearer``). The inbound adapter is a no-op for
requests/responses but still provides a real ``render_stream`` so that an
OpenAI-speaking client can sit in front of a *non*-OpenAI upstream (the
canonical chunks are re-encoded as standard ``data:`` SSE frames).
"""

from __future__ import annotations

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


register_outbound(OpenAIOutbound())
register_inbound(OpenAIInbound())
