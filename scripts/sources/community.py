"""Community-list source — github.com/tashfeenahmed/freellmapi.

This is the upstream cited in llmproxy/providers.json's _sources field
(and originally at setup_wizard.py:219 before the refactor). It is a
community-maintained list of free-tier LLM APIs; confidence is "low"
because entries lag provider changes and aren't always normalized.

The repo exposes its data as a README markdown table. Rather than HTML-
scrape the rendered page (fragile), we fetch the raw README from the
default branch and parse the table.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import requests

from .base import Evidence, Source

# Raw README on GitHub. Pinned to default branch; failures fall through to
# "no evidence" so a 404 or rename doesn't trigger removals.
COMMUNITY_README_URL = (
    "https://raw.githubusercontent.com/tashfeenahmed/freellmapi/main/README.md"
)
TIMEOUT = (5, 10)


# Map README provider labels to our provider keys. Conservative — anything
# we can't confidently map gets dropped (better to miss adds than to wrongly
# attribute models to the wrong provider).
PROVIDER_ALIASES: dict[str, str] = {
    "google": "google",
    "google ai studio": "google",
    "google gemini": "google",
    "groq": "groq",
    "cerebras": "cerebras",
    "github models": "github",
    "github": "github",
    "sambanova": "sambanova",
    "sambanova cloud": "sambanova",
    "mistral": "mistral",
    "mistral ai": "mistral",
    "cloudflare workers ai": "cloudflare-workers",
    "cloudflare workers": "cloudflare-workers",
    "cohere": "cohere",
    "zhipu": "zhipu",
    "zhipu ai": "zhipu",
    "z.ai": "z-ai",
    "z-ai": "z-ai",
    "moonshot": "moonshot",
    "moonshot ai": "moonshot",
    "minimax": "minimax",
    "huggingface": "huggingface",
    "hugging face": "huggingface",
    "nous": "nous",
    "nous research": "nous",
    "nvidia": "nvidia",
    "nvidia nim": "nvidia",
    "xai": "xai",
    "x.ai": "xai",
    "grok": "xai",
    "openrouter": "openrouter",
    "vercel": "vercel",
    "venice": "venice",
    "venice ai": "venice",
    "deepseek": "deepseek",
    "ollama cloud": "ollama-cloud",
    "opencode zen": "opencode-zen",
    "opencode-zen": "opencode-zen",
}


class CommunitySource(Source):
    name = "community"

    def __init__(self, url: str = COMMUNITY_README_URL):
        self.url = url

    def fetch(self) -> list[Evidence]:
        resp = requests.get(self.url, timeout=TIMEOUT)
        resp.raise_for_status()
        return list(self._parse(resp.text))

    def _parse(self, readme: str) -> Iterable[Evidence]:
        """Yield Evidence for each row of a markdown table that looks like
        | provider | model | ... |.

        We only emit positive (is_free=True) evidence — the community list
        is by definition a list of *free* offerings.
        """
        for row in _iter_table_rows(readme):
            if len(row) < 2:
                continue
            provider_label = row[0].strip().lower()
            model_field = row[1].strip()
            provider_key = PROVIDER_ALIASES.get(provider_label)
            if not provider_key:
                continue
            # Model cell can be "model-a, model-b" or markdown-linked.
            for raw in re.split(r"[,/]\s*", model_field):
                model = _strip_markdown(raw).strip()
                if not model:
                    continue
                yield Evidence(
                    provider=provider_key,
                    model_id=f"{provider_key}/{model}",
                    is_free=True,
                    source=self.name,
                    confidence="low",
                    url=self.url,
                )


_ROW_RE = re.compile(r"^\|(.+)\|\s*$")


def _iter_table_rows(md: str) -> Iterable[list[str]]:
    """Yield rows from any markdown table in *md*, skipping header/separator."""
    in_table = False
    for line in md.splitlines():
        m = _ROW_RE.match(line.rstrip())
        if not m:
            in_table = False
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        # Separator rows are all dashes — skip them and use as table marker.
        if all(set(_c) <= set("-: ") for _c in cells if _c):
            in_table = True
            continue
        if in_table:
            yield cells


def _strip_markdown(s: str) -> str:
    """Strip markdown link syntax: '[text](url)' -> 'text'; '`code`' -> 'code'."""
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = s.replace("`", "")
    return s
