"""Per-provider docs scrapers.

Each scraper reads a single provider's docs / pricing / rate-limits page and
emits Evidence at "high" confidence. Adding a new scraper:

  1. Drop a module at scripts/sources/docs/<provider>.py exporting a class
     deriving from DocsScraperBase.
  2. Add the class to DOCS_SOURCES below.

If a scraper raises, the aggregator at __init__.py:_DocsAggregator catches
the exception and treats it as "no evidence" from that source. Failures
must NEVER be interpreted as removals.
"""

from __future__ import annotations

from .base import DocsScraperBase
from .cerebras import CerebrasDocs
from .cohere import CohereDocs
from .google import GoogleDocs
from .groq import GroqDocs
from .mistral import (
    MistralDocs as MistralDocs,  # disabled — URL 404 as of 2025-05; re-enable when fixed
)
from .openrouter import OpenRouterFreeFilter as OpenRouterFreeFilter  # alt form

DOCS_SOURCES: list[type[DocsScraperBase]] = [
    GoogleDocs,
    GroqDocs,
    CerebrasDocs,
    # MistralDocs — disabled: https://docs.mistral.ai/deployment/laplateforme/tier/ returns 404.
    # The Mistral La Plateforme free tier still exists but there is no known replacement URL.
    # Believed-free list is maintained manually in llmproxy/free_models.json until a stable
    # docs URL is available.
    CohereDocs,
]
