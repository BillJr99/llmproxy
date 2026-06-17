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

Probes run with bounded *per-provider* concurrency: requests to different
providers overlap, but each provider is capped (``concurrency``, default 3) so
we stay under its requests-per-minute ceiling. Free tiers are gated on request
count, not payload size, so the tiny prompt does not exempt us from rate limits;
a throttled probe returns no usage and would silently suppress cost detection.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

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
DEFAULT_CONCURRENCY = 3


def _make_progress(total: int, desc: str):
    """Return a tqdm progress bar, or None if tqdm is unavailable.

    tqdm is an optional, CLI-only convenience — the scraper must work without
    it, so a missing import degrades to no progress bar rather than an error.
    """
    if total <= 0:
        return None
    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001 — tqdm is optional
        return None
    return tqdm(total=total, desc=desc, unit="model")
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
        concurrency: int | None = None,
    ) -> None:
        self.config_path = config_path
        self.max_models = max_models
        self.provider_filter = provider_filter
        self.concurrency = max(1, concurrency or DEFAULT_CONCURRENCY)

    def _believed_free_models(self, sidecar: dict) -> list[str]:
        """Collect qualified ``provider/model`` ids from the sidecar believed_free."""
        out: list[str] = []
        for prov in sidecar.get("providers", {}).values():
            out.extend(prov.get("believed_free", []))
        return out

    def _candidates(self, config: dict, sidecar: dict) -> list[tuple[str, str, str, str, str]]:
        """Build the probe-able set as ``(provider, upstream_model, qualified, base_url, key)``.

        Filtering (provider filter, configured-provider, present API key) and the
        ``max_models`` budget are applied here, sequentially, so the parallel
        phase has a fixed, already-bounded work list and a stable progress total.
        """
        out: list[tuple[str, str, str, str, str]] = []
        for qualified in self._believed_free_models(sidecar):
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
            out.append((provider_name, upstream_model, qualified, base_url, api_key))
            if self.max_models is not None and len(out) >= self.max_models:
                break
        return out

    def fetch(self) -> list[Evidence]:
        config = load_config(self.config_path, force_reload=True)
        sidecar = load_data()
        pricing = load_pricing_map(sidecar)

        candidates = self._candidates(config, sidecar)
        if not candidates:
            return []

        evidence: list[Evidence] = []
        evidence_lock = threading.Lock()
        # One semaphore per provider caps concurrent in-flight requests to that
        # provider; distinct providers still overlap. This keeps us under
        # per-provider RPM limits without serializing the whole probe.
        sems: dict[str, threading.Semaphore] = {}
        for provider_name, *_ in candidates:
            sems.setdefault(provider_name, threading.Semaphore(self.concurrency))

        progress = _make_progress(len(candidates), "probing believed_free")

        def _work(item: tuple[str, str, str, str, str]) -> None:
            provider_name, upstream_model, qualified, base_url, api_key = item
            try:
                with sems[provider_name]:
                    usage = self._probe(base_url, api_key, upstream_model)
                if usage is None:
                    return  # unreachable / no usage → no opinion (fail-soft)
                cost, source = compute_cost(provider_name, upstream_model, usage, pricing)
                if cost > 0:
                    ev = Evidence(
                        provider=provider_name,
                        model_id=qualified,
                        is_free=False,
                        source=self.name,
                        confidence="high",
                        url=base_url,
                        notes=f"probe observed cost={cost:.8f} (source={source})",
                    )
                    with evidence_lock:
                        evidence.append(ev)
            finally:
                if progress is not None:
                    progress.update(1)

        # Total workers are bounded by the per-provider cap times the provider
        # count so no provider ever exceeds its semaphore, and by the candidate
        # count so we never spin up idle threads.
        max_workers = min(len(candidates), self.concurrency * len(sems))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                list(ex.map(_work, candidates))
        finally:
            if progress is not None:
                progress.close()
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
