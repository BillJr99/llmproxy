"""Shared helpers for per-provider docs scrapers."""

from __future__ import annotations

import requests

from ..base import Evidence, Source

TIMEOUT = (5, 10)


class DocsScraperBase(Source):
    """Base for one-page provider docs scrapers.

    Subclasses implement parse() taking the page HTML and returning
    Evidence. The base handles HTTP and timeout policy.
    """

    name: str = ""
    url: str = ""
    provider_key: str = ""  # e.g. "google", "groq"

    def fetch(self) -> list[Evidence]:
        resp = requests.get(self.url, timeout=TIMEOUT, headers={
            "User-Agent": "llmproxy-update-free-models/1.0 (+https://github.com/billjr99/llmproxy)",
        })
        resp.raise_for_status()
        return list(self.parse(resp.text))

    def parse(self, html: str) -> list[Evidence]:  # noqa: D401 — overridden by subclasses
        raise NotImplementedError


def _bs(html: str):
    """Lazy bs4 import so the production llmproxy package doesn't need it."""
    from bs4 import BeautifulSoup  # type: ignore
    return BeautifulSoup(html, "html.parser")


def _evidence(provider: str, model: str, *, source: str, url: str,
              is_free: bool | None = True,
              limits: dict | None = None,
              notes: str = "") -> Evidence:
    return Evidence(
        provider=provider,
        model_id=f"{provider}/{model}",
        is_free=is_free,
        source=source,
        confidence="high",
        url=url,
        limits=limits,
        notes=notes,
    )
