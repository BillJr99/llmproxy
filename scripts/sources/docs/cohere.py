"""Cohere trial-key (free) docs scraper.

Source: https://docs.cohere.com/docs/rate-limits

Cohere offers a trial-key tier that gives free access to a small set of
models (notably command-r-plus-08-2024 at low rpm/rpd).
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://docs.cohere.com/docs/rate-limits"
MODEL_RE = re.compile(r"\bcommand[\w.-]+", re.IGNORECASE)


class CohereDocs(DocsScraperBase):
    name = "cohere-docs"
    url = URL
    provider_key = "cohere"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        out: list[Evidence] = []
        seen: set[str] = set()
        for table in soup.find_all("table"):
            text = table.get_text(" ", strip=True).lower()
            if "trial" not in text and "free" not in text:
                continue
            for model in MODEL_RE.findall(text):
                mid = model.lower()
                if mid in seen:
                    continue
                seen.add(mid)
                out.append(_evidence(
                    provider="cohere",
                    model=mid,
                    source=self.name,
                    url=self.url,
                    is_free=True,
                ))
        return out
