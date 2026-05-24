"""Cohere trial-key (free) docs scraper.

Source: https://docs.cohere.com/docs/rate-limits

Cohere offers a trial-key tier that gives free access to Command models.
The rate-limits page has a table with columns "Model", "Trial rate limit",
"Production rate limit". Model display names changed in 2025 from
hyphenated slugs (command-r-plus-08-2024) to space-separated names
(Command A+, Command R+, Command A, Command R, …).

We take the first cell of every data row in a table that contains a
"Trial rate limit" column, slugify the display name, and emit it.
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://docs.cohere.com/docs/rate-limits"

# Identifies tables that list trial (free) limits.
_TRIAL_RE = re.compile(r"trial\s+rate\s+limit", re.IGNORECASE)
# Matches any model display name starting with "Command" (case-insensitive).
_COMMAND_RE = re.compile(r"^command\b", re.IGNORECASE)


def _slugify(display: str) -> str:
    """'Command A+' -> 'command-a-plus', 'Command R+' -> 'command-r-plus'."""
    s = display.strip().lower()
    s = s.replace("+", "-plus").replace("_", "-")
    s = re.sub(r"[^a-z0-9-]", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


class CohereDocs(DocsScraperBase):
    name = "cohere-docs"
    url = URL
    provider_key = "cohere"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        out: list[Evidence] = []
        seen: set[str] = set()

        for table in soup.find_all("table"):
            header_text = table.get_text(" ", strip=True)
            if not _TRIAL_RE.search(header_text):
                continue

            for row in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if not cells:
                    continue
                display = cells[0].strip()
                if not _COMMAND_RE.match(display):
                    continue
                slug = _slugify(display)
                if not slug or slug in seen:
                    continue
                seen.add(slug)
                out.append(_evidence(
                    provider="cohere",
                    model=slug,
                    source=self.name,
                    url=self.url,
                    is_free=True,
                    notes=f"display_name={display!r}",
                ))
        return out
