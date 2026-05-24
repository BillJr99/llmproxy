"""Cohere trial-key (free) docs scraper.

Source: https://docs.cohere.com/docs/rate-limits

Cohere offers a trial-key tier that gives free access to a small set of
models. The page uses the word "Trial" (not "free") in the tier table, and
model display names may include spaces and special chars (e.g. "Command A+",
"Command A Reasoning"). We extract model names row-by-row from the first
cell of each row in trial/free tables, then normalize to an API-compatible ID.
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://docs.cohere.com/docs/rate-limits"

# Matches display names like: "command-r-plus-08-2024", "Command A+",
# "Command A Reasoning", "command r", "embed-english-v3.0"
_COMMAND_RE = re.compile(r"\bcommand\b", re.IGNORECASE)
_EMBED_RE = re.compile(r"\bembed[\w.-]*", re.IGNORECASE)


def _normalize_model_name(display: str) -> str | None:
    """Convert a Cohere display name to a likely API model ID.

    Examples:
      "Command A+"            -> "command-a-plus"
      "Command A Reasoning"   -> "command-a-reasoning"
      "Command R+"            -> "command-r-plus"
      "Command R"             -> "command-r"
      "command-r-plus-08-2024"-> "command-r-plus-08-2024"  (already normalized)
      "embed-english-v3.0"    -> "embed-english-v3.0"
    """
    s = display.strip()
    if not s:
        return None
    # Already looks like an API ID (hyphen-separated, no spaces except "+" suffix).
    if " " not in s and s == s.lower():
        return s.lower()
    # Normalize display name: lowercase, spaces→hyphens, "+" → "plus"
    normalized = s.lower()
    normalized = normalized.replace("+", "-plus")
    normalized = re.sub(r"[\s]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized


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

            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if not cells:
                    continue
                cell_text = cells[0].get_text(strip=True)
                if not (_COMMAND_RE.search(cell_text) or _EMBED_RE.search(cell_text)):
                    continue
                model_id = _normalize_model_name(cell_text)
                if not model_id or model_id in seen:
                    continue
                seen.add(model_id)
                out.append(_evidence(
                    provider="cohere",
                    model=model_id,
                    source=self.name,
                    url=self.url,
                    is_free=True,
                ))
        return out
