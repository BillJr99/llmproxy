"""OpenRouter source — high-confidence detection of $0-priced models.

OpenRouter's /api/v1/models endpoint returns pricing per model. Models with
`pricing.prompt == "0"` and `pricing.completion == "0"` are free at the
gateway level. They typically also have ":free" in their model id.
"""

from __future__ import annotations

import requests

from .base import Evidence, Source

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
TIMEOUT = (5, 10)


class OpenRouterSource(Source):
    name = "openrouter"

    def __init__(self, url: str = OPENROUTER_URL):
        self.url = url

    def fetch(self) -> list[Evidence]:
        resp = requests.get(self.url, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        out: list[Evidence] = []
        for model in data.get("data", []):
            mid = model.get("id")
            if not mid:
                continue
            pricing = model.get("pricing") or {}
            prompt_price = _to_float(pricing.get("prompt"))
            completion_price = _to_float(pricing.get("completion"))
            is_free = (prompt_price == 0.0 and completion_price == 0.0)
            out.append(Evidence(
                provider="openrouter",
                model_id=f"openrouter/{mid}",
                is_free=is_free,
                source=self.name,
                confidence="high",
                url=self.url,
                capabilities=_capabilities(model),
                pricing=_pricing(prompt_price, completion_price),
                notes=f"prompt={pricing.get('prompt')!r} completion={pricing.get('completion')!r}",
            ))
        return out


def _capabilities(model: dict) -> list[str]:
    """Map an OpenRouter model entry to llmproxy capability tags.

    Derives from `supported_parameters` (tools / reasoning / structured-output)
    and `architecture.input_modalities` (image -> vision).
    """
    supported = model.get("supported_parameters") or []
    if not isinstance(supported, list):
        supported = []
    modalities = (model.get("architecture") or {}).get("input_modalities") or []
    if not isinstance(modalities, list):
        modalities = []
    caps: list[str] = []
    if "tools" in supported:
        caps.append("tools")
    if "image" in modalities:
        caps.append("vision")
    if "reasoning" in supported:
        caps.append("reasoning")
    if "structured_outputs" in supported or "response_format" in supported:
        caps.append("json")
    return caps


def _pricing(prompt_price: float, completion_price: float) -> dict | None:
    """Build a pricing record for a PAID model, or None when free/unknown.

    Zero-priced models are captured by believed_free, not the pricing block, so
    they contribute no pricing opinion. ``inf`` means OpenRouter omitted/garbled
    the price — also no opinion.
    """
    in_ok = prompt_price != float("inf")
    out_ok = completion_price != float("inf")
    if not in_ok and not out_ok:
        return None
    in_cost = prompt_price if in_ok else 0.0
    out_cost = completion_price if out_ok else 0.0
    if in_cost == 0.0 and out_cost == 0.0:
        return None
    return {"input_cost_per_token": in_cost, "output_cost_per_token": out_cost}


def _to_float(v) -> float:
    """Best-effort float coercion. OpenRouter encodes prices as strings."""
    if v is None:
        return float("inf")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")
