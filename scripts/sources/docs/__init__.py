"""Per-provider docs scrapers.

Each scraper reads a single provider's docs / pricing / rate-limits page and
emits Evidence. Adding a new scraper:

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
from .cloudflare import CloudflareWorkersDocs
from .cohere import CohereDocs
from .github_models import GitHubModelsDocs
from .google import GoogleDocs
from .groq import GroqDocs
from .huggingface import HuggingFaceDocs
from .mistral import MistralDocs
from .openrouter import OpenRouterFreeFilter as OpenRouterFreeFilter  # alt form
from .sambanova import SambaNovaDocs

DOCS_SOURCES: list[type[DocsScraperBase]] = [
    GoogleDocs,
    GroqDocs,
    CerebrasDocs,
    MistralDocs,
    CohereDocs,
    SambaNovaDocs,
    GitHubModelsDocs,
    CloudflareWorkersDocs,
    HuggingFaceDocs,
]
