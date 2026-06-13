"""LiteLLM cost-map source — cross-provider zero-price detection.

LiteLLM publishes a public model_prices_and_context_window.json on GitHub
containing pricing for ~2,000 model/provider combos. Models with
`input_cost_per_token == 0` and `output_cost_per_token == 0` are free at
the listed provider.

Entries look like::

    "groq/llama-3.1-8b-instant": {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "litellm_provider": "groq",
        "mode": "chat"
    },
    "openrouter/google/gemini-2.0-flash-exp:free": {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "litellm_provider": "openrouter",
        ...
    }

We emit one Evidence per `<provider>/<model>` key where the cost is zero,
mapping LiteLLM's `litellm_provider` to our provider keys via PROVIDER_ALIASES.
No auth required; failures fall through to "no evidence".

Confidence is `medium` — LiteLLM is community-maintained and lags upstream
changes by days. It's broader than any single docs scraper, so it's a useful
complement, but not authoritative.
"""

from __future__ import annotations

from collections.abc import Iterable

import requests

from .base import Evidence, Source

LITELLM_COST_MAP_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
TIMEOUT = (5, 15)

# LiteLLM uses different provider keys for some providers — map to ours.
# Conservative: anything we can't map is dropped (better to miss than to
# attribute models to the wrong provider).
PROVIDER_ALIASES: dict[str, str] = {
    "groq": "groq",
    "cerebras": "cerebras",
    "openrouter": "openrouter",
    "gemini": "google",
    "google": "google",
    "google_ai_studio": "google",
    "vertex_ai-language-models": "google",
    "mistral": "mistral",
    "codestral": "mistral",
    "cohere": "cohere",
    "cohere_chat": "cohere",
    "deepseek": "deepseek",
    "xai": "xai",
    "ai21": "ai21",
    "sambanova": "sambanova",
    "huggingface": "huggingface",
    "github": "github",
    "cloudflare": "cloudflare-workers",
    "together_ai": "together",
    "together": "together",
    "fireworks_ai": "fireworks",
    "fireworks_ai-chat-models": "fireworks",
    "fireworks": "fireworks",
    "nvidia_nim": "nvidia",
    "nvidia": "nvidia",
    "ollama": "ollama-cloud",
    "moonshot": "moonshot",
}


class LiteLLMCostMapSource(Source):
    name = "litellm_cost_map"

    def __init__(self, url: str = LITELLM_COST_MAP_URL):
        self.url = url

    def fetch(self) -> list[Evidence]:
        resp = requests.get(self.url, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return list(self._parse(data))

    def _parse(self, data: dict) -> Iterable[Evidence]:
        for key, meta in data.items():
            if not isinstance(meta, dict):
                continue
            litellm_provider = meta.get("litellm_provider")
            provider_key = PROVIDER_ALIASES.get(litellm_provider or "")
            if not provider_key:
                continue
            # Skip non-chat models — we only care about LLM chat completions.
            mode = meta.get("mode")
            if mode and mode not in ("chat", "completion", "responses"):
                continue
            input_cost = _to_float(meta.get("input_cost_per_token"))
            output_cost = _to_float(meta.get("output_cost_per_token"))
            is_free = (input_cost == 0.0 and output_cost == 0.0)
            if not is_free:
                continue
            # Extract bare model id — strip "<litellm_provider>/" prefix if present.
            bare = key
            if "/" in key and key.split("/", 1)[0] == litellm_provider:
                bare = key.split("/", 1)[1]
            yield Evidence(
                provider=provider_key,
                model_id=f"{provider_key}/{bare}",
                is_free=True,
                source=self.name,
                confidence="medium",
                url=self.url,
                notes=f"litellm_key={key!r}",
            )


def _to_float(v) -> float:
    if v is None:
        return float("inf")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")


# ---------------------------------------------------------------------------
# Pricing snapshot (per-token USD costs for runtime cost accounting)
# ---------------------------------------------------------------------------
# The same litellm map that flags zero-priced models also carries the actual
# per-token prices for paid ones. We snapshot those into providers.json so the
# proxy can price tokens offline (server.py / usage.py compute_cost), preferring
# a provider-reported usage.cost when present and falling back to this map.


def build_pricing_map(data: dict) -> dict[str, dict]:
    """Build ``provider/model`` → {input_cost_per_token, output_cost_per_token}.

    Keys are lowercased and mapped to llmproxy provider keys via PROVIDER_ALIASES.
    Only chat-style models with at least one non-zero price are included (zero
    cost is already captured by believed_free, and omitting it keeps the
    snapshot compact).
    """
    out: dict[str, dict] = {}
    for key, meta in data.items():
        if not isinstance(meta, dict):
            continue
        litellm_provider = meta.get("litellm_provider")
        provider_key = PROVIDER_ALIASES.get(litellm_provider or "")
        if not provider_key:
            continue
        mode = meta.get("mode")
        if mode and mode not in ("chat", "completion", "responses"):
            continue
        input_cost = meta.get("input_cost_per_token")
        output_cost = meta.get("output_cost_per_token")
        in_f = _to_float(input_cost)
        out_f = _to_float(output_cost)
        if in_f == float("inf") and out_f == float("inf"):
            continue
        in_val = 0.0 if in_f == float("inf") else in_f
        out_val = 0.0 if out_f == float("inf") else out_f
        if in_val == 0.0 and out_val == 0.0:
            continue
        bare = key
        if "/" in key and key.split("/", 1)[0] == litellm_provider:
            bare = key.split("/", 1)[1]
        out[f"{provider_key}/{bare}".lower()] = {
            "input_cost_per_token": in_val,
            "output_cost_per_token": out_val,
        }
    return out


def fetch_pricing_map(url: str = LITELLM_COST_MAP_URL) -> dict[str, dict]:
    """Download the litellm cost map and return the pricing snapshot."""
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return build_pricing_map(resp.json())
