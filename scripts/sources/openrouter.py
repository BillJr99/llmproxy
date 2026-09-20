"""OpenRouter source — high-confidence detection of $0-priced models.

OpenRouter's /api/v1/models endpoint returns pricing per model. Models with
`pricing.prompt == "0"` and `pricing.completion == "0"` are free at the
gateway level. They typically also have ":free" in their model id.
"""

from __future__ import annotations

import requests

from llmproxy.providers import capabilities_from_listing  # type: ignore

from .base import Evidence, Source

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
TIMEOUT = (5, 10)


class OpenRouterSource(Source):
    name = "openrouter"
    # /api/v1/models is the gateway's complete catalog, not a free-tier subset,
    # so a model missing from the response has genuinely been withdrawn. This is
    # what lets short-lived cloaked models age out of believed_free again.
    enumerates_catalog = True

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

    Thin adapter over `llmproxy.providers.capabilities_from_listing`, the one
    implementation shared with the route-cache rebuild and the routing-metadata
    refresh — deriving the tags here as well is how the copies drifted before.

    The shared helper returns a set; `Evidence.capabilities` is documented as a
    `list[str]`, so the tags are sorted on the way out. Alphabetical order is
    arbitrary but deterministic, which is what keeps the serialized sidecar
    byte-stable across re-scrapes.
    """
    return sorted(capabilities_from_listing(model))


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
