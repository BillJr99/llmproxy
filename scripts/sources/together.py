"""Together AI source — pricing-aware /v1/models detection.

Together AI's `/v1/models` endpoint returns pricing metadata per model.
Models where the input and output prices are both zero are genuinely free
(Together has a small free tier of community-hosted models). Requires
``TOGETHER_API_KEY`` in the environment; without it, the source yields nothing.

The response shape (abridged) is::

    [
      {
        "id": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "type": "chat",
        "pricing": {"input": 0, "output": 0, "hourly": 0, ...},
        ...
      },
      ...
    ]

Pricing fields are floats expressed per 1M tokens; "input"/"output"/"base"
== 0 indicates the model is offered at no cost. Confidence is `high` —
this is the provider's own pricing endpoint.
"""

from __future__ import annotations

import os

import requests

from .base import Evidence, Source

TOGETHER_URL = "https://api.together.xyz/v1/models"
TIMEOUT = (5, 15)


class TogetherSource(Source):
    name = "together"

    def __init__(self, url: str = TOGETHER_URL):
        self.url = url

    def fetch(self) -> list[Evidence]:
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            return []
        resp = requests.get(
            self.url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # Together returns a plain list at the top level (not {"data": [...]}).
        if isinstance(data, dict):
            data = data.get("data") or data.get("models") or []

        out: list[Evidence] = []
        for model in data:
            if not isinstance(model, dict):
                continue
            mid = model.get("id") or model.get("name")
            if not mid:
                continue
            # Only chat-capable models — skip embeddings, image, audio.
            model_type = (model.get("type") or "").lower()
            if model_type and model_type not in ("chat", "language", "completion"):
                continue
            pricing = model.get("pricing") or {}
            in_cost = _to_float(pricing.get("input"))
            out_cost = _to_float(pricing.get("output"))
            base_cost = _to_float(pricing.get("base", 0))
            is_free = (in_cost == 0.0 and out_cost == 0.0 and base_cost == 0.0)
            out.append(Evidence(
                provider="together",
                model_id=f"together/{mid}",
                is_free=is_free,
                source=self.name,
                confidence="high",
                url=self.url,
                notes=f"input={pricing.get('input')!r} output={pricing.get('output')!r}",
            ))
        return out


def _to_float(v) -> float:
    if v is None:
        return float("inf")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")
