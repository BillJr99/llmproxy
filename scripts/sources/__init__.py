"""Source plugins for the free-models scraper.

Each source emits Evidence records (see scripts.sources.base.Evidence).
Sources are registered in ALL_SOURCES so the CLI can enumerate them.
"""

from .api_models import ApiModelsSource
from .base import Evidence, Source
from .community import CommunitySource
from .cost_probe import CostProbeSource
from .endpoint_probe import EndpointProbeSource
from .fireworks import FireworksSource
from .litellm_cost_map import LiteLLMCostMapSource
from .openrouter import OpenRouterSource
from .requesty import RequestySource
from .together import TogetherSource

ALL_SOURCES: dict[str, type[Source]] = {
    "openrouter": OpenRouterSource,
    "community": CommunitySource,
    "api": ApiModelsSource,
    "litellm_cost_map": LiteLLMCostMapSource,
    "together": TogetherSource,
    "fireworks": FireworksSource,
    "requesty": RequestySource,
    # Active cost probe. Excluded from the default source set because it sends
    # real requests; opt in via free_tier.cost_probe.enabled or --cost-probe.
    "cost_probe": CostProbeSource,
    # Endpoint discovery probe. Excluded from the default source set because it
    # makes authenticated GET /models requests; opt in via sync_on_startup or
    # update_on_startup, or pass --source endpoint_probe.
    "endpoint_probe": EndpointProbeSource,
}

# Sources that must NOT run unless explicitly opted into (they spend real quota
# or make authenticated requests beyond the normal scrape set).
OPT_IN_SOURCES: frozenset[str] = frozenset({"cost_probe", "endpoint_probe"})

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


__all__ = ["ALL_SOURCES", "OPT_IN_SOURCES", "Evidence", "Source"]
