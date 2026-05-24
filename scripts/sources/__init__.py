"""Source plugins for the free-models scraper.

Each source emits Evidence records (see scripts.sources.base.Evidence).
Sources are registered in ALL_SOURCES so the CLI can enumerate them.
"""

from .api_models import ApiModelsSource
from .base import Evidence, Source
from .community import CommunitySource
from .openrouter import OpenRouterSource

ALL_SOURCES: dict[str, type[Source]] = {
    "openrouter": OpenRouterSource,
    "community": CommunitySource,
    "api": ApiModelsSource,
}

# Docs scrapers are registered separately so we can list them under --source docs.
from .docs import DOCS_SOURCES  # noqa: E402

ALL_SOURCES["docs"] = lambda: _DocsAggregator()  # type: ignore[assignment]


class _DocsAggregator(Source):
    """Runs every provider-specific docs scraper and concatenates evidence."""

    name = "docs"

    def fetch(self) -> list[Evidence]:
        out: list[Evidence] = []
        for cls in DOCS_SOURCES:
            try:
                out.extend(cls().fetch())
            except Exception as exc:  # noqa: BLE001 — any failure is non-fatal per source
                print(f"  [docs:{cls.name}] failed: {exc}")
        return out


__all__ = ["ALL_SOURCES", "Evidence", "Source"]
