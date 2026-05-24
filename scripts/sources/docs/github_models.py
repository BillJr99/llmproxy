"""GitHub Models free-tier docs scraper.

Source: https://docs.github.com/en/github-models/prototyping-with-ai-models

GitHub publishes a per-tier rate-limit table on its prototyping docs page
listing models accessible via the GitHub Models endpoint (Meta Llama,
GPT-4o, Mistral, Phi, …). The table identifies "Low" and "High" rate-limit
tiers; both are free at point of use (gated by GitHub PAT scopes only).

We collect every model ID that appears in any rate-limits table row and
emit it at high confidence as is_free=True. We do not attempt to extract
per-tier RPM/RPD figures (the table layout is complex and tier mapping is
opaque); the OpenRouter and LiteLLM sources provide better rate data.
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://docs.github.com/en/github-models/prototyping-with-ai-models"

# Match the "<vendor>/<model>" style IDs used in the GitHub Models catalog.
# Examples:
#   openai/gpt-4o, openai/gpt-4o-mini
#   meta/Meta-Llama-3.1-70B-Instruct
#   mistral-ai/Mistral-Nemo
#   microsoft/Phi-3.5-mini-instruct
_VENDOR_PREFIXES = (
    "openai", "meta", "mistral-ai", "mistral", "microsoft",
    "ai21-labs", "cohere", "deepseek", "xai", "core42",
)
_MODEL_ID_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _VENDOR_PREFIXES) + r")/[A-Za-z0-9][\w.\-]*",
    re.IGNORECASE,
)


class GitHubModelsDocs(DocsScraperBase):
    name = "github-models-docs"
    url = URL
    provider_key = "github"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        text = soup.get_text(" ", strip=True)

        # Guard: page must look like the rate-limits doc (mentions both "rate"
        # and one of the tier markers). Otherwise emit nothing — the catalog
        # changed and we don't want to attribute random vendor/model strings.
        low = text.lower()
        if "rate limit" not in low or not any(
            kw in low for kw in ("free", "low tier", "high tier", "github models")
        ):
            return []

        out: list[Evidence] = []
        seen: set[str] = set()
        # Search the raw HTML so anchor hrefs and code blocks are both covered.
        for raw in _MODEL_ID_RE.findall(html):
            mid = raw.strip()
            if mid.lower() in seen:
                continue
            seen.add(mid.lower())
            out.append(_evidence(
                provider="github",
                model=mid,
                source=self.name,
                url=self.url,
                is_free=True,
                notes="surfaced from GitHub Models prototyping docs",
            ))
        return out
