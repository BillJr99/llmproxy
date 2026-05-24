"""Hugging Face Inference Providers free-tier docs scraper.

Source: https://huggingface.co/docs/inference-providers/

The Inference Providers docs describe which third-party serverless backends
HF routes inference through (Together, Fireworks, Cerebras, etc.). Many are
free at HF's free tier — capped by request count rather than dollars.

The page lists models as ``<vendor>/<model>`` identifiers (the same shape
the HF router accepts via ``https://router.huggingface.co/v1/chat/completions``).
We extract them and emit at medium confidence — "free" status depends on
which provider HF routes through, and HF doesn't break that down per-model
in the static docs. Confidence drops to medium accordingly, but the IDs
themselves are reliable.
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs

URL = "https://huggingface.co/docs/inference-providers/"

# HF model IDs are "<vendor>/<repo>" with optional ":<tag>".
# Vendors include: meta-llama, mistralai, Qwen, deepseek-ai, google,
# microsoft, openai-community, accounts/fireworks, etc.
_MODEL_RE = re.compile(
    r"\b(?:meta-llama|mistralai|Qwen|deepseek-ai|google|microsoft|"
    r"openai-community|accounts/fireworks/models|HuggingFaceH4|"
    r"NousResearch|tiiuae|allenai|CohereForAI|01-ai)/[A-Za-z0-9._\-]+",
    re.IGNORECASE,
)


class HuggingFaceDocs(DocsScraperBase):
    name = "huggingface-docs"
    url = URL
    provider_key = "huggingface"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        text = soup.get_text(" ", strip=True).lower()

        # Sanity guard — page must mention "inference providers" and at least
        # one signal that this is the catalog/overview, not an unrelated docs
        # page.
        if "inference provider" not in text:
            return []

        out: list[Evidence] = []
        seen: set[str] = set()
        for raw in _MODEL_RE.findall(html):
            mid = raw.strip()
            if mid in seen:
                continue
            seen.add(mid)
            # Emit at medium confidence (see module docstring).
            out.append(Evidence(
                provider="huggingface",
                model_id=f"huggingface/{mid}",
                is_free=True,
                source=self.name,
                confidence="medium",
                url=self.url,
                notes="HF Inference Providers docs — free up to per-account rate cap",
            ))
        return out
