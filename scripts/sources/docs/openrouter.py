"""Alternative OpenRouter docs scraper.

The primary OpenRouter source (scripts.sources.openrouter.OpenRouterSource)
uses the /v1/models JSON API. This module is a fallback that scrapes
openrouter.ai/models?max_price=0 if the JSON API ever becomes unavailable.

Currently a thin stub — not registered in DOCS_SOURCES by default — kept so
the JSON source has a documented backup.
"""

from __future__ import annotations

from ..base import Evidence
from .base import DocsScraperBase

URL = "https://openrouter.ai/models?max_price=0"


class OpenRouterFreeFilter(DocsScraperBase):
    name = "openrouter-docs"
    url = URL
    provider_key = "openrouter"

    def parse(self, html: str) -> list[Evidence]:
        # The page is JS-rendered, so HTML scraping returns little useful data
        # without a headless browser. Prefer the /v1/models JSON API. This
        # method exists as a documented hook for future implementation; for
        # now it returns no evidence so we don't accidentally signal removals.
        return []
