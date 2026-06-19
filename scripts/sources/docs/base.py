"""Shared helpers for per-provider docs scrapers."""

from __future__ import annotations

import time

import requests

from ..base import Evidence, Source

TIMEOUT = (5, 10)
_HEADERS = {
    "User-Agent": "llmproxy-update-free-models/1.0 (+https://github.com/billjr99/llmproxy)",
}
_429_RETRY_DELAYS = (5.0, 15.0, 30.0)  # seconds between successive 429 retries


class DocsScraperBase(Source):
    """Base for one-page provider docs scrapers.

    Subclasses implement parse() taking the page HTML and returning
    Evidence. The base handles HTTP and timeout policy.
    """

    name: str = ""
    url: str = ""
    provider_key: str = ""  # e.g. "google", "groq"

    def fetch(self) -> list[Evidence]:
        delays = iter(_429_RETRY_DELAYS)
        while True:
            resp = requests.get(self.url, timeout=TIMEOUT, headers=_HEADERS)
            if resp.status_code == 429:
                wait = next(delays, None)
                if wait is None:
                    return []  # retries exhausted — skip silently rather than raise
                ra = resp.headers.get("Retry-After", "")
                actual_wait = float(ra) if ra.isdigit() else wait
                time.sleep(min(actual_wait, 60.0))
                continue
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
