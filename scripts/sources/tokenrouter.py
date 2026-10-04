"""TokenRouter source — public pricing feed, no API key required.

TokenRouter (https://api.tokenrouter.com/v1) is a deployment of the open-source
``new-api`` gateway. Its ``/v1/models`` route is key-gated, but the catalog
behind the public pricing page (https://www.tokenrouter.com/models) is served
unauthenticated as JSON from ``GET https://api.tokenrouter.com/api/pricing``,
so this source needs no credential and therefore produces evidence inside the
``Update providers.json`` workflow, which runs with no secrets configured.

The response shape (abridged) is::

    {"success": true,
     "group_ratio": {"default": 1, "vip": 1},
     "data": [
       {"model_name": "anthropic/claude-sonnet-4", "tags": "Text",
        "quota_type": 0,               # 0 = per token, 1 = fixed price per call
        "model_ratio": 1.5,            # input price, in new-api ratio units
        "completion_ratio": 5,         # output price as a multiple of input
        "cache_ratio": 0.1, "create_cache_ratio": 1.25,
        "enable_groups": ["default"],
        "supported_endpoint_types": ["anthropic-compatible"]},
       ...
     ],
     "tiered_pricing": {...}, ...}

Prices are not quoted in dollars but as new-api ratios, where a model_ratio of
1 is $2 per 1M input tokens. The conversion is therefore::

    input  $/1M = model_ratio * 2 * group_ratio["default"]
    output $/1M = input $/1M * completion_ratio

That convention is confirmed by the payload itself: its ``tiered_pricing`` block
quotes both a ratio and a dollar ``pricePerUnit`` for the same tier, e.g.
anthropic/claude-sonnet-4 at ratio 1.5 and $3/1M. For tiered models the
row-level ratios equal the first (smallest-context) tier, which is the price
recorded here. A missing or non-positive default group ratio yields no pricing
opinion rather than a figure that could be wrong by an arbitrary factor.

Only per-token rows (``quota_type == 0``) served on a text-generation endpoint
are considered; image, video, embedding and audio-only rows are skipped.
Freeness requires a zero model_ratio AND the OpenAI chat endpoint, because the
OpenAI surface is the only one llmproxy routes to; a zero-priced model reachable
only through another dialect is recorded as not free.

Model ids are already namespaced by upstream vendor (e.g. "openai/gpt-5.5");
we prefix with our own provider key, giving "tokenrouter/openai/gpt-5.5".

Like the xKiro source, this one emits *negative* evidence for any id in
TokenRouter's ``believed_free`` that the live feed no longer lists free, because
aggregate() only removes a model by absence for the source named "api".
"""

from __future__ import annotations

import requests

from llmproxy import USER_AGENT
from llmproxy.providers import load_data  # type: ignore

from .base import Evidence, Source

TOKENROUTER_PRICING_URL = "https://api.tokenrouter.com/api/pricing"
TIMEOUT = (5, 15)

PROVIDER = "tokenrouter"

# new-api's base unit: a model_ratio of 1 is $2 per 1M tokens (it is defined as
# $0.002 per 1K tokens). providers.json stores per-token prices.
_DOLLARS_PER_MILLION_PER_RATIO = 2.0
_PER_MILLION = 1_000_000.0

# Endpoint types that carry text generation. Anything else (embeddings,
# image-generation, video-*, audio-chat, system-one) is never priced here.
_TEXT_ENDPOINTS = frozenset({
    "openai", "openai-response", "anthropic", "anthropic-compatible", "gemini",
})
# The surface llmproxy actually routes to.
_ROUTABLE_ENDPOINT = "openai"


class TokenRouterSource(Source):
    name = "tokenrouter"

    def __init__(self, url: str = TOKENROUTER_PRICING_URL):
        self.url = url

    def fetch(self) -> list[Evidence]:
        # No auth: the pricing feed is public. A hard failure must propagate so
        # the CLI records succeeded=False ("no evidence") rather than an empty
        # catalog, which would read as "every model was removed".
        resp = requests.get(self.url, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("unexpected TokenRouter pricing payload shape")

        group_ratio = _default_group_ratio(data.get("group_ratio"))

        out: list[Evidence] = []
        free_seen: set[str] = set()

        for row in data["data"]:
            if not isinstance(row, dict):
                continue
            mid = row.get("model_name")
            if not isinstance(mid, str) or not mid:
                continue
            if row.get("quota_type") != 0:
                continue
            endpoints = row.get("supported_endpoint_types")
            endpoints = set(endpoints) if isinstance(endpoints, list) else set()
            if not endpoints & _TEXT_ENDPOINTS:
                continue

            qualified = f"{PROVIDER}/{mid}"
            model_ratio = _to_float(row.get("model_ratio"))
            completion_ratio = _to_float(row.get("completion_ratio"))
            zero_priced = model_ratio == 0.0
            routable = _ROUTABLE_ENDPOINT in endpoints

            is_free = zero_priced and routable
            if is_free:
                free_seen.add(qualified)

            note = (f"model_ratio={row.get('model_ratio')!r} "
                    f"completion_ratio={row.get('completion_ratio')!r} "
                    f"endpoints={sorted(endpoints)}")
            if zero_priced and not routable:
                note += " (zero-priced but not on the OpenAI endpoint — treated as not free)"

            out.append(Evidence(
                provider=PROVIDER,
                model_id=qualified,
                is_free=is_free,
                source=self.name,
                confidence="high",
                url=self.url,
                pricing=_pricing(model_ratio, completion_ratio, group_ratio),
                notes=note,
            ))

        out.extend(self._stale_negatives(free_seen))
        return out

    def _stale_negatives(self, free_seen: set[str]) -> list[Evidence]:
        """High-confidence negatives for believed_free ids no longer listed free.

        Only reached after a successful fetch, so an unreachable host can never
        produce removals. Ids still listed but now paid already got an is_free
        False record above; repeating it here is harmless.
        """
        try:
            believed_free = load_data()["providers"][PROVIDER].get("believed_free", [])
        except (KeyError, TypeError):
            return []
        return [
            Evidence(
                provider=PROVIDER,
                model_id=qualified,
                is_free=False,
                source=self.name,
                confidence="high",
                url=self.url,
                notes="pricing feed no longer lists this model as free",
            )
            for qualified in sorted(set(believed_free) - free_seen)
        ]


def _default_group_ratio(raw) -> float | None:
    """The price multiplier for the "default" user group, or None if unusable."""
    if not isinstance(raw, dict):
        return None
    ratio = _to_float(raw.get("default"))
    if ratio == float("inf") or ratio <= 0.0:
        return None
    return ratio


def _pricing(model_ratio: float, completion_ratio: float,
             group_ratio: float | None) -> dict | None:
    """Per-token pricing for a PAID model, or None when free/unknown."""
    if group_ratio is None or model_ratio == float("inf") or model_ratio < 0.0:
        return None
    if completion_ratio == float("inf") or completion_ratio < 0.0:
        return None
    in_per_million = model_ratio * _DOLLARS_PER_MILLION_PER_RATIO * group_ratio
    out_per_million = in_per_million * completion_ratio
    if in_per_million == 0.0 and out_per_million == 0.0:
        return None
    return {
        "input_cost_per_token": round(in_per_million / _PER_MILLION, 15),
        "output_cost_per_token": round(out_per_million / _PER_MILLION, 15),
    }


def _to_float(v) -> float:
    if v is None or isinstance(v, bool):
        return float("inf")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")
