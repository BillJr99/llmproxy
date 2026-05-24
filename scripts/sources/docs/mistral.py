"""Mistral free-tier ("La Plateforme — Free Experiments" tier) scraper.

Source: https://docs.mistral.ai/deployment/laplateforme/tier/
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://docs.mistral.ai/deployment/laplateforme/tier/"
MODEL_RE = re.compile(r"\b(mistral|magistral|codestral|pixtral|ministral)-[\w.-]+", re.IGNORECASE)


class MistralDocs(DocsScraperBase):
    name = "mistral-docs"
    url = URL
    provider_key = "mistral"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        text = soup.get_text(" ", strip=True)
        # Only fire if the page actually discusses a free experiment / free tier.
        if "free" not in text.lower():
            return []
        out: list[Evidence] = []
        seen: set[str] = set()
        for m in MODEL_RE.finditer(text):
            full = m.group(0).lower()
            if full in seen:
                continue
            seen.add(full)
            out.append(_evidence(
                provider="mistral",
                model=full,
                source=self.name,
                url=self.url,
                is_free=True,
            ))
        return out
