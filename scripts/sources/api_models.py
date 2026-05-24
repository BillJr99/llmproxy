"""Provider /v1/models source — confirms a model still exists upstream.

For each provider that has its API key exposed in the environment (via the
``<PROVIDER>_API_KEY`` env var, e.g. GROQ_API_KEY), we call the OpenAI-
compatible /models endpoint. The result is medium confidence: existence
proves the model is still offered, but not whether it's free.

Aggregation logic in update_free_models.py uses this signal mainly to
detect *removals* — if a model in believed_free is missing from a
successful /models fetch, that's strong evidence of a removal.
"""

from __future__ import annotations

import os

import requests

from llmproxy.free_models import load_data  # type: ignore

from .base import Evidence, Source

TIMEOUT = (5, 10)


def _env_key(provider_key: str) -> str:
    """Convert 'cloudflare-workers' -> 'CLOUDFLARE_WORKERS_API_KEY'."""
    return provider_key.upper().replace("-", "_") + "_API_KEY"


class ApiModelsSource(Source):
    name = "api"

    def __init__(self, providers: dict[str, str] | None = None):
        """providers maps provider_key -> base_url. If None, derived from
        the sidecar at module load."""
        if providers is None:
            data = load_data()
            providers = {
                k: v["base_url"]
                for k, v in data["providers"].items()
                if "{account_id}" not in v.get("base_url", "")  # skip URL-templated providers without env vars
                and "{gateway_id}" not in v.get("base_url", "")
            }
        self.providers = providers

    def fetch(self) -> list[Evidence]:
        out: list[Evidence] = []
        for provider_key, base_url in self.providers.items():
            env_var = _env_key(provider_key)
            api_key = os.environ.get(env_var)
            if not api_key:
                continue
            try:
                models = self._fetch_models(base_url, api_key)
            except Exception as exc:  # noqa: BLE001 — per-provider failure isolated
                print(f"  [api:{provider_key}] /models fetch failed: {exc}")
                continue
            for mid in models:
                out.append(Evidence(
                    provider=provider_key,
                    model_id=f"{provider_key}/{mid}",
                    is_free=None,  # presence doesn't imply free
                    source=self.name,
                    confidence="medium",
                    url=f"{base_url.rstrip('/')}/models",
                ))
        return out

    @staticmethod
    def _fetch_models(base_url: str, api_key: str) -> list[str]:
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return [m.get("id", "") for m in resp.json().get("data", []) if m.get("id")]
