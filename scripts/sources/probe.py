"""probe.py — active cost probe for believed-free models.

Unlike every other source (which reads docs / pricing pages / /models
catalogs), this source sends a *real*, minimal chat completion to each model in
``believed_free`` and inspects the returned ``usage`` block. Any model that
reports a non-zero cost — either a provider-supplied ``usage.cost`` (OpenRouter,
Vercel) or a computed cost from the bundled pricing map — is emitted as
high-confidence ``is_free=False`` so the aggregator can flag/remove it.

This source spends real quota (and possibly real money), so it is OFF by
default: it only runs when the user sets ``probe_cost: true`` in config.json or
passes ``--probe`` on the command line, and it skips any provider that has no
configured API key.
"""

from __future__ import annotations

import requests

from llmproxy.config import (
    get_provider,
    load_config,
    provider_api_key,
    provider_base_url,
)
from llmproxy.providers import load_data
from llmproxy.usage import compute_cost, extract_usage, load_pricing_map

from .base import Evidence, Source

TIMEOUT = (5, 30)
# A deliberately tiny request — one token out — to minimize spend per probe.
_PROBE_BODY = {
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 1,
    "stream": False,
}


class ProbeSource(Source):
    name = "probe"

    def __init__(
        self,
        config_path: str | None = None,
        max_models: int | None = None,
        provider_filter: str | None = None,
    ) -> None:
        self.config_path = config_path
        self.max_models = max_models
        self.provider_filter = provider_filter

    def _believed_free_models(self, sidecar: dict) -> list[str]:
        """Collect qualified ``provider/model`` ids from the sidecar believed_free."""
        out: list[str] = []
        for prov in sidecar.get("providers", {}).values():
            out.extend(prov.get("believed_free", []))
        return out

    def fetch(self) -> list[Evidence]:
        config = load_config(self.config_path, force_reload=True)
        sidecar = load_data()
        pricing = load_pricing_map(sidecar)

        models = self._believed_free_models(sidecar)
        evidence: list[Evidence] = []
        probed = 0
        for qualified in models:
            if "/" not in qualified:
                continue
            provider_name, upstream_model = qualified.split("/", 1)
            if self.provider_filter and provider_name != self.provider_filter:
                continue
            provider_cfg = get_provider(config, provider_name)
            if not provider_cfg:
                continue  # provider not configured locally — no key to probe with
            base_url = provider_base_url(provider_cfg)
            api_key = provider_api_key(provider_cfg)
            if not api_key:
                continue  # no credentials → skip rather than spend an anonymous call
            if self.max_models is not None and probed >= self.max_models:
                break
            probed += 1

            usage = self._probe(base_url, api_key, upstream_model)
            if usage is None:
                continue  # unreachable / no usage → no opinion (fail-soft)
            cost, source = compute_cost(provider_name, upstream_model, usage, pricing)
            if cost > 0:
                evidence.append(Evidence(
                    provider=provider_name,
                    model_id=qualified,
                    is_free=False,
                    source=self.name,
                    confidence="high",
                    url=base_url,
                    notes=f"probe observed cost={cost:.8f} (source={source})",
                ))
        return evidence

    def _probe(self, base_url: str, api_key: str, model: str) -> dict | None:
        """Send one minimal completion and return its usage block, or None."""
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={**_PROBE_BODY, "model": model},
                timeout=TIMEOUT,
            )
            if resp.status_code >= 400:
                return None
            return extract_usage(resp.content)
        except Exception:
            return None
