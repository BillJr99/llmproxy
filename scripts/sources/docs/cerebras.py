"""Cerebras free-tier docs scraper.

Source: https://inference-docs.cerebras.ai/support/rate-limits
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://inference-docs.cerebras.ai/support/rate-limits"

# Model IDs on this page are written as "qwen-3-coder-480b" / "gpt-oss-120b".
MODEL_RE = re.compile(
    r"\b(?:qwen[\w.-]*|llama[\w.-]*|gpt-oss[\w.-]*|deepseek[\w.-]*)\b",
    re.IGNORECASE,
)


class CerebrasDocs(DocsScraperBase):
    name = "cerebras-docs"
    url = URL
    provider_key = "cerebras"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        out: list[Evidence] = []
        seen: set[str] = set()
        for table in soup.find_all("table"):
            joined_table = table.get_text(" ", strip=True).lower()
            if "free" not in joined_table:
                # Cerebras "Free Tier" appears as a header — only emit when present.
                continue
            for row in table.find_all("tr"):
                row_text = row.get_text(" ", strip=True)
                for model in MODEL_RE.findall(row_text):
                    m = model.lower()
                    if m in seen:
                        continue
                    seen.add(m)
                    out.append(_evidence(
                        provider="cerebras",
                        model=m,
                        source=self.name,
                        url=self.url,
                        is_free=True,
                    ))
        return out
