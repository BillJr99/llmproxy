"""Google AI Studio (Gemini) free-tier docs scraper.

Source: https://ai.google.dev/gemini-api/docs/rate-limits

The page contains a table with one row per tier. We extract the "Free Tier"
row's model column and the per-model rpm/rpd/tpm values.

NB: Google often changes the rendered HTML; the scraper is intentionally
loose — it walks every table cell looking for "gemini-…" identifiers
rather than depending on positional column indexing.
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://ai.google.dev/gemini-api/docs/rate-limits"
MODEL_RE = re.compile(r"gemini-[\w.-]+", re.IGNORECASE)


class GoogleDocs(DocsScraperBase):
    name = "google-docs"
    url = URL
    provider_key = "google"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        out: list[Evidence] = []
        seen: set[str] = set()

        # Strategy: find every table on the page; for each, check if the
        # nearest preceding heading (h1-h4) indicates the free tier.
        # Google's docs have changed over time:
        #   - Old layout: heading "Free Tier" / "Paid Tier" before each table.
        #   - New layout: heading "Tier 1 Rate Limits" (Tier 1 = free, no billing).
        # We accept any heading that contains "free" OR "tier 1".
        for table in soup.find_all("table"):
            heading = _nearest_preceding_heading(table)
            heading_text = (heading.get_text(" ", strip=True).lower() if heading else "")
            if "free" not in heading_text and "tier 1" not in heading_text:
                continue

            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                joined = " ".join(c.get_text(" ", strip=True) for c in cells)
                for model in MODEL_RE.findall(joined):
                    model = model.lower()
                    if model in seen:
                        continue
                    seen.add(model)
                    rpm, rpd, tpm = _extract_limits(joined)
                    out.append(_evidence(
                        provider="google",
                        model=model,
                        source=self.name,
                        url=self.url,
                        is_free=True,
                        limits={
                            "requests_per_minute": rpm,
                            "requests_per_day": rpd,
                            "tokens_per_minute": tpm,
                            "tokens_per_day": None,
                        } if (rpm or rpd or tpm) else None,
                    ))
        return out


def _nearest_preceding_heading(tag):
    """Walk backwards through siblings (and up) to find the nearest h1-h4."""
    for sib in tag.previous_elements:
        name = getattr(sib, "name", None)
        if name in ("h1", "h2", "h3", "h4"):
            return sib
    return None


_RPM_RE = re.compile(r"(\d[\d,]*)\s*(?:RPM|requests?\s*/\s*min)", re.IGNORECASE)
_RPD_RE = re.compile(r"(\d[\d,]*)\s*(?:RPD|requests?\s*/\s*day)", re.IGNORECASE)
_TPM_RE = re.compile(r"(\d[\d,]*)\s*(?:TPM|tokens?\s*/\s*min)", re.IGNORECASE)


def _extract_limits(text: str) -> tuple[int | None, int | None, int | None]:
    def _grab(rx):
        m = rx.search(text)
        return int(m.group(1).replace(",", "")) if m else None
    return _grab(_RPM_RE), _grab(_RPD_RE), _grab(_TPM_RE)
