"""Cloudflare Workers AI free-tier model catalog scraper.

Source: https://developers.cloudflare.com/workers-ai/models/

Cloudflare's Workers AI catalog lists every model in machine-readable JSON
embedded in the page's data attributes / script blocks, and as anchor
elements pointing at `/workers-ai/models/<model-name>/`. The Workers Free
plan includes AI inference up to a daily neuron budget — all listed models
are available to free-plan users at point of access, gated only by neuron
quota (handled by the runtime, not by per-model pricing).

We extract every Cloudflare-style model identifier (``@cf/<vendor>/<name>``)
from the page and emit it at high confidence as is_free=True. Confidence is
high because the catalog page IS the source of truth for the Workers AI
model list.
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://developers.cloudflare.com/workers-ai/models/"

# Cloudflare model IDs always start with "@cf/" or "@hf/".
_MODEL_RE = re.compile(r"@(?:cf|hf)/[A-Za-z0-9_.-]+/[A-Za-z0-9_.\-]+")


class CloudflareWorkersDocs(DocsScraperBase):
    name = "cloudflare-workers-docs"
    url = URL
    provider_key = "cloudflare-workers"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        text = soup.get_text(" ", strip=True).lower()

        # Sanity guard — page must look like the Workers AI model catalog.
        if "workers ai" not in text and "@cf/" not in html:
            return []

        out: list[Evidence] = []
        seen: set[str] = set()
        for raw in _MODEL_RE.findall(html):
            mid = raw.strip()
            if mid in seen:
                continue
            seen.add(mid)
            out.append(_evidence(
                provider="cloudflare-workers",
                model=mid,
                source=self.name,
                url=self.url,
                is_free=True,
                notes="Workers Free plan includes daily neuron budget for all listed models",
            ))
        return out
