"""endpoint_probe.py — discovers :free-suffixed models via GET /v1/models.

Unlike cost_probe.py (which sends real chat completions to cost-verify models),
this source only makes authenticated GET requests to each configured provider's
/v1/models endpoint. It emits:

  * positive Evidence for model IDs ending in ':free' that are returned by the
    endpoint but are not yet in believed_free — new free models to consider adding.
  * negative Evidence for model IDs ending in ':free' that *are* in believed_free
    but are no longer returned by the endpoint — models that may have been removed.

Non-:free believed_free models are ignored by this source; their presence/absence
in the /v1/models listing is handled by other sources (e.g. api_models).

This source makes authenticated network calls but does not spend quota, so it
is controlled by the existing sync_on_startup / update_on_startup startup flags
rather than requiring its own 'enabled' toggle.
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

from .base import Evidence, Source

DEFAULT_TIMEOUT = 10


class EndpointProbeSource(Source):
    name = "endpoint_probe"

    def __init__(
        self,
        config_path: str | None = None,
        provider_filter: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.config_path = config_path
        self.provider_filter = provider_filter
        self.timeout = timeout

    def fetch(self) -> list[Evidence]:
        config = load_config(self.config_path, force_reload=True)
        sidecar = load_data()

        # Build set of believed_free :free-suffixed models per provider.
        believed_free_per_provider: dict[str, set[str]] = {}
        for _prov_name, prov_data in sidecar.get("providers", {}).items():
            for qualified in prov_data.get("believed_free", []):
                if "/" not in qualified:
                    continue
                p, m = qualified.split("/", 1)
                if m.endswith(":free"):
                    believed_free_per_provider.setdefault(p, set()).add(m)

        evidence: list[Evidence] = []
        for provider_name, provider_cfg in config.get("providers", {}).items():
            if self.provider_filter and provider_name != self.provider_filter:
                continue
            if not get_provider(config, provider_name):
                continue
            api_key = provider_api_key(provider_cfg)
            if not api_key:
                continue
            base_url = provider_base_url(provider_cfg)
            if not base_url:
                continue

            endpoint_ids = self._fetch_model_ids(base_url, api_key)
            if endpoint_ids is None:
                continue  # network failure — emit no opinion (fail-soft)

            endpoint_free = {m for m in endpoint_ids if m.endswith(":free")}
            known_free = believed_free_per_provider.get(provider_name, set())
            models_url = f"{base_url}/models"

            # Positive: new :free models not yet in believed_free.
            for model_id in sorted(endpoint_free - known_free):
                evidence.append(Evidence(
                    provider=provider_name,
                    model_id=f"{provider_name}/{model_id}",
                    is_free=True,
                    source=self.name,
                    confidence="high",
                    url=models_url,
                    notes="endpoint lists model with :free suffix; not yet in believed_free",
                ))

            # Negative: believed_free :free models absent from endpoint.
            for model_id in sorted(known_free - endpoint_free):
                evidence.append(Evidence(
                    provider=provider_name,
                    model_id=f"{provider_name}/{model_id}",
                    is_free=False,
                    source=self.name,
                    confidence="high",
                    url=models_url,
                    notes="endpoint no longer lists this :free model",
                ))

        return evidence

    def _fetch_model_ids(self, base_url: str, api_key: str) -> list[str] | None:
        """GET {base_url}/models and return the list of model id strings, or None on error."""
        try:
            resp = requests.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                return None
            data = resp.json()
            models = data.get("data") if isinstance(data, dict) else data
            if not isinstance(models, list):
                return None
            return [
                m.get("id", "")
                for m in models
                if isinstance(m, dict) and m.get("id")
            ]
        except Exception:  # noqa: BLE001 — any failure means no opinion
            return None
