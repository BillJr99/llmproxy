"""Groq free-tier docs scraper.

Source: https://console.groq.com/docs/rate-limits

Groq publishes one rate-limit table per model on its docs page. We extract
the model id (the page uses inline <code>…</code> for ids) and the rpm/rpd
columns. Free tier is the default — there's no separate "Free Tier" label,
so every model on this page is treated as free.
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://console.groq.com/docs/rate-limits"


class GroqDocs(DocsScraperBase):
    name = "groq-docs"
    url = URL
    provider_key = "groq"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        out: list[Evidence] = []
        seen: set[str] = set()

        # Walk tables; the model-id column uses <code>...</code> tags.
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                code_tags = row.find_all("code")
                if not code_tags:
                    continue
                model = code_tags[0].get_text(strip=True)
                if not model or "/" in model or model in seen:
                    continue
                seen.add(model)
                joined = row.get_text(" ", strip=True)
                rpm, rpd, tpm, tpd = _extract_limits(joined)
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


_RPM_RE = re.compile(r"(\d[\d,]*)\s*(?:RPM|req\.?/min)", re.IGNORECASE)
_RPD_RE = re.compile(r"(\d[\d,]*)\s*(?:RPD|req\.?/day)", re.IGNORECASE)
_TPM_RE = re.compile(r"(\d[\d,]*)\s*(?:TPM|tok\.?/min)", re.IGNORECASE)
_TPD_RE = re.compile(r"(\d[\d,]*)\s*(?:TPD|tok\.?/day)", re.IGNORECASE)


def _extract_limits(text: str):
    def _grab(rx):
        m = rx.search(text)
        return int(m.group(1).replace(",", "")) if m else None
    return _grab(_RPM_RE), _grab(_RPD_RE), _grab(_TPM_RE), _grab(_TPD_RE)
