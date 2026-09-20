"""xKiro source — pricing-aware /v1/models detection, no API key required.

xKiro's `/v1/models` endpoint is public: it answers an unauthenticated GET with
the full catalog, so unlike the Requesty/Together sources this one needs no
credential and therefore actually produces evidence inside the
``Update providers.json`` workflow, which runs with no secrets configured.

The response shape (abridged) is::

    {"object": "list", "data": [
      {
        "id": "mistralai/ministral-8b",
        "modality": "chat",
        "access_tier": "free",                        # free | paid | premium
        "pricing": {"currency": "USD", "unit": "per_1m_tokens",
                    "input": 0, "output": 0, "cache_read": 0},
        "capabilities": {"vision": true, "tools": true, "reasoning": false},
        "context_length": 256000, "max_output_tokens": 65536
      },
      ...
    ]}

Freeness is asserted only when the two independent signals agree: xKiro marks
the model ``access_tier: "free"`` AND its input and output prices are both zero.
A disagreement resolves to *not free* and is recorded in ``notes`` — a false
positive here bills the user for real tokens, whereas a false negative costs
nothing but a missing entry.

Model ids are already namespaced by upstream vendor (e.g. "mistralai/ministral-8b");
we prefix with our own provider key for the fully-qualified id, consistent with
every other source, giving "xkiro/mistralai/ministral-8b".

Prices are per 1M tokens (the payload says so explicitly via ``pricing.unit``),
so they are converted to the per-token convention that providers.json uses.

This source also emits *negative* evidence for any id currently in xKiro's
``believed_free`` that the live catalog no longer lists. That is deliberate:
aggregate() only removes a model by absence when the evidence came from the
source literally named "api" (see update_free_models.aggregate), so without an
explicit high-confidence negative a stale entry would survive forever. The
pattern mirrors scripts/sources/endpoint_probe.py.
"""

from __future__ import annotations

import requests

from llmproxy import USER_AGENT
from llmproxy.providers import load_data  # type: ignore

from .base import Evidence, Source

XKIRO_URL = "https://api.xkiro.com/v1/models"
TIMEOUT = (5, 15)

PROVIDER = "xkiro"

# xKiro expresses prices per 1M tokens; providers.json pricing is per token.
_PER_MILLION = 1_000_000.0
_EXPECTED_PRICING_UNIT = "per_1m_tokens"

# Modalities that represent a chat/completion model. Anything else (embeddings,
# image, audio, rerank) must never reach believed_free.
_CHAT_MODALITIES = frozenset({"chat", "language", "completion", "text"})

# xKiro capability flag -> the vocabulary providers.json uses.
_CAPABILITY_FLAGS = (("tools", "tools"), ("vision", "vision"), ("reasoning", "reasoning"))


class XkiroSource(Source):
    name = "xkiro"

    def __init__(self, url: str = XKIRO_URL):
        self.url = url

    def fetch(self) -> list[Evidence]:
        # No auth: the catalog is public. A hard failure must propagate so the
        # CLI records succeeded=False ("no evidence") rather than an empty
        # catalog, which would read as "every model was removed".
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
            mid = model.get("id")
            if not mid:
                continue

            modality = (model.get("modality") or "").lower()
            if modality and modality not in _CHAT_MODALITIES:
                continue

            qualified = f"{PROVIDER}/{mid}"
            seen.add(qualified)

            pricing = model.get("pricing") if isinstance(model.get("pricing"), dict) else {}
            in_cost = _to_float(pricing.get("input"))
            out_cost = _to_float(pricing.get("output"))
            zero_priced = (in_cost == 0.0 and out_cost == 0.0)
            tier = (model.get("access_tier") or "").lower()
            tier_free = (tier == "free")

            is_free = tier_free and zero_priced
            note = f"access_tier={tier!r} input={pricing.get('input')!r} output={pricing.get('output')!r}"
            if tier_free != zero_priced:
                note += " (signals disagree — treated as not free)"

            out.append(Evidence(
                provider=PROVIDER,
                model_id=qualified,
                is_free=is_free,
                source=self.name,
                confidence="high",
                url=self.url,
                capabilities=_capabilities(model.get("capabilities")),
                pricing=_pricing(in_cost, out_cost, pricing.get("unit")),
                notes=note,
            ))

        out.extend(self._stale_negatives(seen))
        return out

    def _stale_negatives(self, seen: set[str]) -> list[Evidence]:
        """High-confidence negatives for believed_free ids the catalog dropped.

        Only reached after a successful fetch, so an unreachable host can never
        produce removals.
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
                notes="catalog no longer lists this model",
            )
            for qualified in sorted(set(believed_free) - seen)
        ]


def _capabilities(raw) -> list[str] | None:
    """Map xKiro's capability flags onto the providers.json vocabulary.

    None means "no opinion" — don't touch existing capability metadata.
    """
    if not isinstance(raw, dict):
        return None
    return [name for flag, name in _CAPABILITY_FLAGS if raw.get(flag) is True]


def _pricing(in_per_million: float, out_per_million: float, unit) -> dict | None:
    """Per-token pricing for a PAID model, or None when free/unknown.

    Inputs are xKiro's per-1M-token prices; convert to per token. A payload that
    ever stops saying per_1m_tokens yields no pricing opinion rather than a
    figure that could be wrong by a factor of a million.
    """
    if unit is not None and str(unit).lower() != _EXPECTED_PRICING_UNIT:
        return None
    in_ok = in_per_million != float("inf")
    out_ok = out_per_million != float("inf")
    if not in_ok and not out_ok:
        return None
    in_cost = (in_per_million / _PER_MILLION) if in_ok else 0.0
    out_cost = (out_per_million / _PER_MILLION) if out_ok else 0.0
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
