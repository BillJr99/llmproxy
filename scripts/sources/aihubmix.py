"""AIHubMix source — pricing-aware catalog detection, no API key required.

AIHubMix publishes its catalog with prices at ``https://aihubmix.com/api/v1/models``,
which answers an unauthenticated GET, so like the xKiro source this one also
produces evidence inside the ``Update providers.json`` workflow (no secrets).
That endpoint is the catalog only: inference is served from
``https://aihubmix.com/v1``, and the ids here are the same ids that API takes.

The response shape (abridged) is::

    {"data": [
      {
        "model_id": "coding-glm-5.4-free",
        "types": "llm",                       # comma-separated: llm, embedding, ...
        "pricing": {"input": 0, "output": 0}, # USD per 1M tokens
        "context_length": 1048576,
        "tool_call": true,
        "input_modalities": "text,image"
      },
      ...
    ]}

A model is free only when the catalog reports a price and EVERY reported price
is zero: input, output, any cache price and every context tier. A model whose
id says "free" but carries any price above zero is paid — AIHubMix re-serves
several legacy ``:free`` ids from other gateways at a real per-token price — so
it gets a negative and per-token pricing, which keeps it out of /free at
request time even though its id matches the "free" rule. A model with no
pricing block gets no free/paid opinion at all (``is_free=None``).

Prices are USD per 1M tokens (gpt-4o-mini lists 0.15 / 0.6) and are converted
to the per-token convention providers.json uses.

Like the xKiro source this emits a high-confidence negative for any id in
AIHubMix's ``believed_free`` that the live catalog no longer lists, so a stale
entry does not survive forever.
"""

from __future__ import annotations

import requests

from llmproxy import USER_AGENT
from llmproxy.providers import load_data  # type: ignore

from .base import Evidence, Source

AIHUBMIX_URL = "https://aihubmix.com/api/v1/models"
TIMEOUT = (5, 30)

PROVIDER = "aihubmix"

_PER_MILLION = 1_000_000.0

# Entries in ``types`` that mark a chat/completion model. Image, video, audio,
# embedding, rerank and moderation models must never reach believed_free.
_CHAT_TYPES = frozenset({"llm"})


class AihubmixSource(Source):
    name = "aihubmix"

    def __init__(self, url: str = AIHUBMIX_URL):
        self.url = url

    def fetch(self) -> list[Evidence]:
        # A hard failure must propagate so the CLI records "no evidence"
        # rather than an empty catalog, which would read as "every model was
        # removed".
        resp = requests.get(self.url, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(models, list):
            return []

        out: list[Evidence] = []
        seen: set[str] = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            mid = model.get("model_id") or model.get("id")
            if not mid:
                continue
            types = {t.strip().lower() for t in str(model.get("types") or "").split(",")}
            if not types & _CHAT_TYPES:
                continue

            qualified = f"{PROVIDER}/{mid}"
            seen.add(qualified)

            pricing = model.get("pricing") if isinstance(model.get("pricing"), dict) else {}
            prices = _all_prices(pricing)
            if not prices:
                is_free = None
                note = "no pricing reported"
            else:
                is_free = all(p == 0.0 for p in prices)
                note = f"input={pricing.get('input')!r} output={pricing.get('output')!r}"
                if not is_free and "free" in str(mid).lower():
                    note += " (id says free but a price above zero is reported — treated as paid)"

            out.append(Evidence(
                provider=PROVIDER,
                model_id=qualified,
                is_free=is_free,
                source=self.name,
                confidence="high",
                url=self.url,
                capabilities=_capabilities(model),
                pricing=None if is_free is not False else _pricing(pricing),
                notes=note,
            ))

        out.extend(self._stale_negatives(seen))
        return out

    def _stale_negatives(self, seen: set[str]) -> list[Evidence]:
        try:
            believed_free = load_data()["providers"][PROVIDER].get("believed_free", [])
        except (KeyError, TypeError):
            return []
        return [
            Evidence(provider=PROVIDER, model_id=qualified, is_free=False,
                     source=self.name, confidence="high", url=self.url,
                     notes="catalog no longer lists this model")
            for qualified in sorted(set(believed_free) - seen)
        ]


def _to_price(v) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _all_prices(pricing: dict | None) -> list[float]:
    """Every numeric price in the block, context tiers included.

    Empty when nothing numeric is reported, which means "unknown", not "free".
    A block with input/output missing is also unknown: a model is only free
    when both are reported and everything reported is zero.
    """
    if not pricing:
        return []
    if _to_price(pricing.get("input")) is None or _to_price(pricing.get("output")) is None:
        return []
    out = []
    for key, val in pricing.items():
        if key == "tiers":
            for tier in val if isinstance(val, list) else []:
                if isinstance(tier, dict):
                    out.extend(p for k, v in tier.items() if k != "tier"
                               for p in [_to_price(v)] if p is not None)
            continue
        p = _to_price(val)
        if p is not None:
            out.append(p)
    return out


def _pricing(pricing: dict | None) -> dict | None:
    """Per-token input/output pricing for a paid model."""
    if not pricing:
        return None
    in_m, out_m = _to_price(pricing.get("input")), _to_price(pricing.get("output"))
    if in_m is None or out_m is None:
        return None
    return {"input_cost_per_token": in_m / _PER_MILLION,
            "output_cost_per_token": out_m / _PER_MILLION}


def _capabilities(model: dict) -> list[str] | None:
    """tools / vision from the catalog; None when it gives no opinion."""
    caps = []
    known = False
    if isinstance(model.get("tool_call"), bool):
        known = True
        if model["tool_call"]:
            caps.append("tools")
    modalities = model.get("input_modalities")
    if isinstance(modalities, str) and modalities:
        known = True
        if "image" in modalities.lower():
            caps.append("vision")
    return caps if known else None
