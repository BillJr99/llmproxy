"""Mistral free-tier scraper.

Historical source: https://docs.mistral.ai/deployment/laplateforme/tier/
(404 as of mid-2025 — Mistral removed the dedicated free-experiments page.)

Mistral's La Plateforme no longer documents a free API tier in static HTML;
rate limits and free-tier access are managed through the AI Studio console.
The scraper now tries the models-overview page as a fallback, looks for any
"free" or "experiment" context near a Mistral model name, and emits those.
If no usable page is found it returns an empty list so the aggregator treats
this as "no evidence" (never a removal signal).
"""

from __future__ import annotations

import re

import requests

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

# Try URLs in order; use the first that returns 200.
_CANDIDATE_URLS = [
    "https://docs.mistral.ai/deployment/laplateforme/tier/",
    "https://docs.mistral.ai/getting-started/models/models_overview/",
]

MODEL_RE = re.compile(r"\b(mistral|magistral|codestral|pixtral|ministral|devstral)-[\w.-]+", re.IGNORECASE)
_TIMEOUT = (5, 10)


class MistralDocs(DocsScraperBase):
    name = "mistral-docs"
    url = _CANDIDATE_URLS[0]   # for display; actual URL is resolved at fetch time
    provider_key = "mistral"

    def fetch(self) -> list[Evidence]:
        """Override fetch to try multiple URL candidates."""
        html = None
        resolved_url = self.url
        for url in _CANDIDATE_URLS:
            try:
                resp = requests.get(url, timeout=_TIMEOUT, headers={
                    "User-Agent": "llmproxy-update-free-models/1.0 (+https://github.com/billjr99/llmproxy)",
                })
                if resp.status_code == 200:
                    html = resp.text
                    resolved_url = url
                    break
            except Exception:  # noqa: BLE001 — try next candidate
                continue
        if html is None:
            # All URLs failed — emit no evidence (treated as failed source).
            raise RuntimeError("No reachable Mistral docs URL found; all candidates returned non-200.")
        self._resolved_url = resolved_url
        return list(self.parse(html))

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        text = soup.get_text(" ", strip=True)
        url = getattr(self, "_resolved_url", self.url)

        # Only emit free evidence if the page contains "free" near a model name.
        if "free" not in text.lower() and "experiment" not in text.lower():
            return []

        out: list[Evidence] = []
        seen: set[str] = set()

        # Find windows where "free"/"experiment" appears near a model name.
        for kw_match in re.finditer(r"free|experiment", text, re.IGNORECASE):
            window = text[max(0, kw_match.start() - 120): kw_match.end() + 120]
            for m in MODEL_RE.finditer(window):
                full = m.group(0).lower()
                if full in seen:
                    continue
                seen.add(full)
                out.append(_evidence(
                    provider="mistral",
                    model=full,
                    source=self.name,
                    url=url,
                    is_free=True,
                ))
        return out
