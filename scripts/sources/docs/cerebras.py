"""Cerebras free-tier docs scraper.

Source: https://inference-docs.cerebras.ai/support/rate-limits

Page structure (as of mid-2025):
  Under the "Limits by Tier" h2, two consecutive tables appear with no
  separating text. The first table has lower limits (RPM=5, TPM=30K) and
  corresponds to the free tier; the second has higher limits (RPM=1K+) for
  paid tiers. Neither table contains the word "free" — we identify the free
  tier by taking the first table under the "Limits by Tier" heading.
"""

from __future__ import annotations

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://inference-docs.cerebras.ai/support/rate-limits"


class CerebrasDocs(DocsScraperBase):
    name = "cerebras-docs"
    url = URL
    provider_key = "cerebras"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        out: list[Evidence] = []
        seen: set[str] = set()

        # Find the "Limits by Tier" heading, then take the first table that
        # follows it — that is the free-tier table (lowest limits).
        tier_heading = None
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            if "limits by tier" in h.get_text(strip=True).lower():
                tier_heading = h
                break

        free_table = None
        if tier_heading is not None:
            # Walk forward siblings of the heading's parent until we hit a table.
            for elem in tier_heading.next_elements:
                name = getattr(elem, "name", None)
                if name == "table":
                    free_table = elem
                    break
                # Stop if we reach a new major section heading.
                if name in ("h1", "h2") and elem is not tier_heading:
                    break

        if free_table is None:
            # Fallback: take the first table on the page that has a model-like
            # first column (heuristic for when the heading structure changes).
            for table in soup.find_all("table"):
                rows = table.find_all("tr")
                if len(rows) > 1:
                    first_cell = rows[1].find(["td", "th"])
                    if first_cell and any(
                        kw in first_cell.get_text(strip=True).lower()
                        for kw in ("llama", "qwen", "gpt-oss", "deepseek", "glm", "zai")
                    ):
                        free_table = table
                        break

        if free_table is None:
            return out

        for row in free_table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if not cells or not cells[0]:
                continue
            model = cells[0].strip().lower()
            # Skip header rows
            if model in ("model", "model id", ""):
                continue
            if model in seen:
                continue
            seen.add(model)
            out.append(_evidence(
                provider="cerebras",
                model=model,
                source=self.name,
                url=self.url,
                is_free=True,
            ))
        return out
