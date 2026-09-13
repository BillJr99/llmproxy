"""Token Harbor docs scraper — free-tier models from the public catalog page.

Token Harbor's JSON catalog at ``/v1/models`` requires a Universal Key (an
unauthenticated GET answers 401), so it cannot be the CI path. The public
catalog page at https://tokenharbor.ai/models can be: it is server-rendered and
ships its data inside the Next.js flight payload, where the free tier lives in a
dedicated ``freeRows`` array, separate from the segment-paginated paid list.

Each row looks like::

    {"surface": "deepseek-v4-flash:free", "label": "DeepSeek V4 Flash",
     "family": "deepseek", "tier": "value", "priceIn": 0, "priceOut": 0,
     "isFree": true, "promo": false, "limited": false, ...}

Token Harbor's own documentation states the convention these ids follow: "model
IDs ending in :free never charge your balance". A row is emitted only when all
three signals agree — the ``:free`` suffix, ``isFree``, and zero in/out price —
so a paid row that drifts into the free section is never published as free.

Because this parses a rendered page rather than a documented contract, a shape
change must degrade to "no evidence", never to an exception or a bogus empty
catalog. Docs scrapers emit positives only, so returning [] causes no removals.
"""

from __future__ import annotations

import codecs
import json
import re

from ..base import Evidence
from .base import DocsScraperBase, _evidence

# Next.js streams the flight payload as a series of self.__next_f.push([1,"..."])
# calls whose payloads concatenate into one string.
_FLIGHT_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[1,\s*"((?:[^"\\]|\\.)*)"\]\)')
_FREE_ROWS_KEY = '"freeRows"'


class TokenHarborDocs(DocsScraperBase):
    name = "tokenharbor"
    url = "https://tokenharbor.ai/models"
    provider_key = "tokenharbor"

    def parse(self, html: str) -> list[Evidence]:
        rows = _free_rows(html)
        out: list[Evidence] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            surface = row.get("surface")
            if not isinstance(surface, str) or not surface.endswith(":free"):
                continue
            if row.get("isFree") is not True:
                continue
            if _num(row.get("priceIn")) != 0.0 or _num(row.get("priceOut")) != 0.0:
                continue
            out.append(_evidence(
                self.provider_key,
                surface,
                source=self.name,
                url=self.url,
                is_free=True,
                notes=f"freeRows entry on the public catalog page (label={row.get('label')!r})",
            ))
        return out


def _free_rows(html: str) -> list:
    """Recover the freeRows array from the page's flight payload.

    Returns [] on any structural surprise — a missing key, an unbalanced
    bracket, or malformed JSON — so a redesign of the page degrades to
    "no opinion" rather than raising.
    """
    try:
        chunks = _FLIGHT_CHUNK_RE.findall(html)
        payload = "".join(codecs.decode(c, "unicode_escape") for c in chunks) if chunks else html
        idx = payload.find(_FREE_ROWS_KEY)
        if idx == -1:
            return []
        start = payload.find("[", idx)
        if start == -1:
            return []
        end = _matching_bracket(payload, start)
        if end == -1:
            return []
        rows = json.loads(payload[start:end])
    except Exception:  # noqa: BLE001 — any parse surprise means "no evidence"
        return []
    return rows if isinstance(rows, list) else []


def _matching_bracket(s: str, start: int) -> int:
    """Index just past the ']' matching s[start], or -1. String-aware."""
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")
