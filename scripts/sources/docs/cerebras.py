"""Cerebras free-tier docs scraper.

Source: https://inference-docs.cerebras.ai/support/rate-limits

The Cerebras rate-limits page lists free-tier (no-billing) limits in the
FIRST table on the page — there is no explicit "Free" heading in the table
text itself, so we can't filter by keyword. We simply treat every model found
in the first data table as a free-tier model.

Model IDs on this page look like:  gpt-oss-120b, llama3.1-8b,
qwen-3-235b-a22b-instruct-2507, zai-glm-4.7, qwen-3-coder-480b, etc.
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://inference-docs.cerebras.ai/support/rate-limits"

MODEL_RE = re.compile(
    r"\b(?:"
    r"qwen[\w.-]*"          # qwen-3-coder-480b, qwen3-235b, ...
    r"|llama[\w.-]*"        # llama3.1-8b, llama-4-maverick-17b-128e-instruct, ...
    r"|gpt-oss[\w.-]*"      # gpt-oss-120b
    r"|deepseek[\w.-]*"     # deepseek-r1, ...
    r"|zai-glm[\w.-]*"      # zai-glm-4.7
    r")\b",
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

        # The free-tier table is always the FIRST <table> on the page.
        # Cerebras removed the explicit "Free" label from table headers (as of
        # 2025-Q2), so we can't rely on keyword detection — just take table[0].
        tables = soup.find_all("table")
        if not tables:
            return out
        free_table = tables[0]

        for row in free_table.find_all("tr"):
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
