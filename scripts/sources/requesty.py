"""Requesty source — pricing-aware /v1/models detection.

Requesty's `/v1/models` endpoint (OpenAI-compatible) returns pricing
metadata per model as flat top-level fields — `input_price`, `output_price`,
`cached_price` — not nested under a `pricing` sub-object. Models with zero
input/output cost are genuinely free (Requesty publishes a small set of
$0-cost models from upstream providers that host them for free, e.g. some
Nemotron and Poolside variants). Requires ``REQUESTY_API_KEY`` in the
environment; without it, the source yields nothing.

Model ids are already namespaced by upstream provider (e.g.
"anthropic/claude-opus-4-7"); we prefix with our own provider key for the
fully-qualified id, consistent with every other source.

Prices are plain per-token dollar amounts (confirmed against a live
response), consistent with OpenRouter's convention rather than Together's
per-million-token convention.
"""

from __future__ import annotations

import os

import requests

from .base import Evidence, Source

REQUESTY_URL = "https://router.requesty.ai/v1/models"
TIMEOUT = (5, 15)


class RequestySource(Source):
    name = "requesty"

    def __init__(self, url: str = REQUESTY_URL):
        self.url = url

    def fetch(self) -> list[Evidence]:
        api_key = os.environ.get("REQUESTY_API_KEY")
        if not api_key:
            return []
        resp = requests.get(
            self.url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(models, list):
            return []

        out: list[Evidence] = []
        for model in models:
            if not isinstance(model, dict):
                continue
            mid = model.get("id")
            if not mid:
                continue
            in_cost = _to_float(model.get("input_price"))
            out_cost = _to_float(model.get("output_price"))
            is_free = (in_cost == 0.0 and out_cost == 0.0)
            out.append(Evidence(
                provider="requesty",
                model_id=f"requesty/{mid}",
                is_free=is_free,
                source=self.name,
                confidence="high",
                url=self.url,
                pricing=_pricing(in_cost, out_cost),
                notes=f"input_price={model.get('input_price')!r} output_price={model.get('output_price')!r}",
            ))
        return out


def _pricing(in_cost: float, out_cost: float) -> dict | None:
    """Per-token pricing for a PAID model, or None when free/unknown."""
    in_ok = in_cost != float("inf")
    out_ok = out_cost != float("inf")
    if not in_ok and not out_ok:
        return None
    in_cost = in_cost if in_ok else 0.0
    out_cost = out_cost if out_ok else 0.0
    if in_cost == 0.0 and out_cost == 0.0:
        return None
    return {"input_cost_per_token": in_cost, "output_cost_per_token": out_cost}


def _to_float(v) -> float:
    if v is None:
        return float("inf")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")
