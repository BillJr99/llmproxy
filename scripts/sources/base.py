"""Source ABC and Evidence record for the free-models scraper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

Confidence = Literal["high", "medium", "low"]


@dataclass
class Evidence:
    """One piece of evidence about a specific (provider, model) pair.

    Aggregation logic in scripts.update_free_models combines many Evidence
    records across sources into add/remove/limits-changed decisions.
    """

    provider: str
    model_id: str  # Fully qualified "<provider>/<id>"
    is_free: bool | None
    source: str
    confidence: Confidence
    url: str
    limits: dict | None = None
    reasoning: str | None = None
    # Capabilities the model supports, e.g. ["tools", "vision", "reasoning", "json"].
    # None = the source has no opinion (don't touch existing capability metadata).
    capabilities: list[str] | None = None
    notes: str = ""


@dataclass
class SourceResult:
    """Output of a Source.fetch() call.

    `succeeded` is set False when the source could not produce any data —
    network failure, bad upstream response, etc. The aggregator must treat
    a failed source as "no evidence" rather than "every model is absent".
    """

    source: str
    succeeded: bool
    evidence: list[Evidence] = field(default_factory=list)
    error: str | None = None


class Source(ABC):
    """Base class for all evidence sources."""

    name: str = ""

    @abstractmethod
    def fetch(self) -> list[Evidence]:
        """Return a list of Evidence records.

        Implementations should raise on hard failure; the CLI wraps each
        call in try/except and converts exceptions into a SourceResult with
        succeeded=False.
        """
