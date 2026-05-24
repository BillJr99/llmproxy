"""Groq free-tier docs scraper.

Source: https://console.groq.com/docs/rate-limits

Page structure (as of mid-2025):
  table[0] — 2-row header: row 0 spans "Free Plan Limits / Developer Plan
             Limits"; row 1 is the column header (MODEL ID, RPM, RPD, …).
  table[1] — one data row per model (model id in plain <td>, not <code>).
  table[2] — rate-limit response headers reference (not model data).

Every model listed on this page is part of the free plan, so all rows are
emitted as is_free=True. Limit values use K/M suffixes (e.g. "7K", "1.2K").
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://console.groq.com/docs/rate-limits"

# Matches the first table's first cell when it announces free-plan limits.
_FREE_PLAN_RE = re.compile(r"free\s+plan", re.IGNORECASE)


class GroqDocs(DocsScraperBase):
    name = "groq-docs"
    url = URL
    provider_key = "groq"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        out: list[Evidence] = []
        seen: set[str] = set()

        tables = soup.find_all("table")

        # Find the header table that announces "Free Plan Limits" and then
        # take the immediately following sibling table as the model data.
        # Fall back to heuristic (first table with plain model-id rows).
        model_table = None
        for i, table in enumerate(tables):
            if _FREE_PLAN_RE.search(table.get_text()):
                # The next table in the list is the model data table.
                if i + 1 < len(tables):
                    model_table = tables[i + 1]
                break

        if model_table is None:
            return out

        col_headers: list[str] = []
        for row in model_table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if not cells:
                continue
            # Detect header row (MODEL ID / RPM / RPD …)
            if not col_headers and cells[0].upper() in ("MODEL ID", "MODEL", "ID"):
                col_headers = [c.upper() for c in cells]
                continue
            if not cells[0] or cells[0] in seen:
                continue
            model = cells[0].strip()
            seen.add(model)

            # Map column positions; fall back to positional defaults.
            def _col(name: str, default: int, cells=cells, headers=col_headers) -> str | None:
                idx = next((i for i, h in enumerate(headers) if name in h), default)
                return cells[idx] if idx < len(cells) else None

            rpm = _parse_k(_col("RPM", 1))
            rpd = _parse_k(_col("RPD", 2))
            tpm = _parse_k(_col("TPM", 3))
            tpd = _parse_k(_col("TPD", 4))

            out.append(_evidence(
                provider="groq",
                model=model,
                source=self.name,
                url=self.url,
                is_free=True,
                limits={
                    "requests_per_minute": rpm,
                    "requests_per_day": rpd,
                    "tokens_per_minute": tpm,
                    "tokens_per_day": tpd,
                } if any([rpm, rpd, tpm, tpd]) else None,
            ))
        return out


_KM_RE = re.compile(r"^([\d.]+)\s*([KkMm]?)$")


def _parse_k(s: str | None) -> int | None:
    """Parse a limit cell like '7K', '1.2K', '500K', '1M', '30', '-'."""
    if not s or s.strip() in ("-", "", "N/A"):
        return None
    m = _KM_RE.match(s.strip().replace(",", ""))
    if not m:
        return None
    val = float(m.group(1))
    suffix = m.group(2).upper()
    if suffix == "K":
        val *= 1_000
    elif suffix == "M":
        val *= 1_000_000
    return int(val)
