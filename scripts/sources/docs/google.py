"""Google AI Studio (Gemini) free-tier docs scraper.

Source: https://ai.google.dev/gemini-api/docs/rate-limits

As of mid-2025 Google moved per-model rate limits out of this page and into
AI Studio (dynamic, not scrapeable). The page still contains:

  • A "Usage tiers" table whose first data row identifies the "Free" tier.
  • Model ID strings scattered throughout the page text (in code blocks,
    table cells, and anchor targets) that identify which models exist.

Strategy:
  1. Confirm the "Free" usage tier is present in the usage-tiers table.
     If not found, emit nothing (conservative — a missing tier row means
     the page structure changed too much to trust our output).
  2. Collect every gemini-* model ID string from the page text.
  3. Emit them all as is_free=True with no rate-limit figures (since the
     page no longer publishes per-model free limits).

This is intentionally broad — the OpenRouter source provides higher-
confidence pricing data; this source's job is only to surface model IDs
that should be considered free for the Google native provider.
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://ai.google.dev/gemini-api/docs/rate-limits"

# Gemini model IDs in the page source, e.g. gemini-2.5-flash-lite.
# Exclude image/asset filenames and non-model strings.
_MODEL_RE = re.compile(r"\bgemini-[\d]+[\.\d]*-[\w.-]+", re.IGNORECASE)

# Suffixes that indicate non-model strings (UI components, images, etc.)
_NOISE_SUFFIXES = {".png", ".svg", ".jpg", "-hovercard", "-button", "-logo",
                   "-switcher", "-table", "-preview-tts", "-native-audio",
                   "-image-preview", "-live-preview", "-flash-tts"}


def _clean_model(raw: str) -> str | None:
    """Normalise a raw regex match; return None to discard."""
    s = raw.lower()
    if any(s.endswith(sfx) for sfx in _NOISE_SUFFIXES):
        return None
    if any(sfx in s for sfx in ("-hovercard", "-logo-", "-api-", "robotics")):
        return None
    return s


class GoogleDocs(DocsScraperBase):
    name = "google-docs"
    url = URL
    provider_key = "google"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        out: list[Evidence] = []

        # Step 1 — confirm "Free" tier row exists in a usage-tiers table.
        free_tier_confirmed = False
        for table in soup.find_all("table"):
            text = table.get_text(" ", strip=True)
            if "usage tier" in text.lower() or "free" in text.lower():
                for row in table.find_all("tr"):
                    cells = [td.get_text(strip=True).lower() for td in row.find_all(["td", "th"])]
                    if cells and cells[0] == "free":
                        free_tier_confirmed = True
                        break
            if free_tier_confirmed:
                break

        if not free_tier_confirmed:
            return out

        # Step 2 — collect model IDs from the full page text.
        seen: set[str] = set()
        for raw in _MODEL_RE.findall(html):
            mid = _clean_model(raw)
            if mid and mid not in seen:
                seen.add(mid)
                out.append(_evidence(
                    provider="google",
                    model=mid,
                    source=self.name,
                    url=self.url,
                    is_free=True,
                    notes="rate-limits page no longer publishes per-model free limits; existence inferred",
                ))
        return out
