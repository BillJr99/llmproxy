"""Fireworks AI source — free-serverless detection via /v1/models.

Fireworks lists all available models at `/inference/v1/models`. Free models
are marked with ``"is_free": true`` or have ``"serverless_billing": "free"``
in their entry. The OpenAI-compatible base URL is at
``https://api.fireworks.ai/inference/v1``; the /models endpoint there
returns the standard OpenAI shape plus Fireworks-specific extras.

Requires ``FIREWORKS_API_KEY`` in the environment; without it, the source
yields nothing. Confidence is `high` — it's the provider's own metadata.
"""

from __future__ import annotations

import os

import requests

from .base import Evidence, Source

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/models"
TIMEOUT = (5, 15)


class FireworksSource(Source):
    name = "fireworks"

    def __init__(self, url: str = FIREWORKS_URL):
        self.url = url

    def fetch(self) -> list[Evidence]:
        api_key = os.environ.get("FIREWORKS_API_KEY")
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
            mid = model.get("id") or model.get("name")
            if not mid:
                continue
            # Fireworks marks free tier with one of several flags. Treat any
            # of them as a positive signal.
            flag_is_free = bool(model.get("is_free"))
            billing = (model.get("serverless_billing") or "").lower()
            pricing = model.get("pricing") or {}
            in_cost = _to_float(pricing.get("input"))
            out_cost = _to_float(pricing.get("output"))

            is_free = (
                flag_is_free
                or billing == "free"
                or (in_cost == 0.0 and out_cost == 0.0)
            )
            out.append(Evidence(
                provider="fireworks",
                model_id=f"fireworks/{mid}",
                is_free=is_free,
                source=self.name,
                confidence="high",
                url=self.url,
                notes=(
                    f"is_free={flag_is_free} billing={billing!r} "
                    f"input={pricing.get('input')!r}"
                ),
            ))
        return out


def _to_float(v) -> float:
    if v is None:
        return float("inf")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")
