"""SambaNova Cloud free-tier docs scraper.

Source: https://cloud.sambanova.ai/apis

SambaNova publishes a rate-limit table for their free tier with columns
Model, RPM, RPD, TPM, TPD. We walk every table on the page and emit
evidence for any row whose first cell looks like a SambaNova model id
(e.g. ``Meta-Llama-3.3-70B-Instruct``, ``DeepSeek-R1``, ``Qwen3-32B``).

Confidence is `high` — this is the provider's own published table.
"""

from __future__ import annotations

import re

from ..base import Evidence
from .base import DocsScraperBase, _bs, _evidence

URL = "https://cloud.sambanova.ai/apis"

# SambaNova model IDs are CamelCase / hyphenated and may include "-Instruct",
# "-Chat", "-Base" suffixes. We accept any first-cell value containing one
# of these model-family roots.
_MODEL_ROOTS = (
    "llama", "deepseek", "qwen", "mistral", "mixtral",
    "tulu", "gemma", "phi", "command",
)
_NUM_RE = re.compile(r"^[\d.,]+\s*([KkMm])?$")


class SambaNovaDocs(DocsScraperBase):
    name = "sambanova-docs"
    url = URL
    provider_key = "sambanova"

    def parse(self, html: str) -> list[Evidence]:
        soup = _bs(html)
        out: list[Evidence] = []
        seen: set[str] = set()

        for table in soup.find_all("table"):
            col_headers: list[str] = []
            for row in table.find_all("tr"):
                cells = [td.get_text(" ", strip=True) for td in row.find_all(["td", "th"])]
                if not cells:
                    continue
                # Capture column header row.
                upper = [c.upper() for c in cells]
                if not col_headers and any(
                    h in (" ".join(upper)) for h in ("RPM", "TPM", "RPD")
                ):
                    col_headers = upper
                    continue
                model = cells[0].strip()
                if not _looks_like_model(model) or model in seen:
                    continue
                seen.add(model)

                def _col(name: str, default_idx: int, cells=cells, headers=col_headers) -> str | None:
                    idx = next(
                        (i for i, h in enumerate(headers) if name in h),
                        default_idx,
                    )
                    return cells[idx] if 0 <= idx < len(cells) else None

                rpm = _parse_num(_col("RPM", 1))
                rpd = _parse_num(_col("RPD", 2))
                tpm = _parse_num(_col("TPM", 3))
                tpd = _parse_num(_col("TPD", 4))

                out.append(_evidence(
                    provider="sambanova",
                    model=model,
                    source=self.name,
                    url=self.url,
                    is_free=True,
                    limits={
                        "requests_per_minute": rpm,
                        "requests_per_day": rpd,
                        "tokens_per_minute": tpm,
                        "tokens_per_day": tpd,
                    } if any([rpm, rpd, tpm, tpd]) else None,
                ))
        return out


def _looks_like_model(s: str) -> bool:
    low = s.lower()
    return any(root in low for root in _MODEL_ROOTS)


def _parse_num(s: str | None) -> int | None:
    """Parse '20', '14.4K', '1M', '-', '' → int|None."""
    if not s:
        return None
    s = s.strip().replace(",", "")
    if s in ("-", "", "N/A", "—"):
        return None
    m = _NUM_RE.match(s)
    if not m:
        return None
    suffix = (m.group(1) or "").upper()
    body = s.rstrip("KkMm")
    try:
        val = float(body)
    except ValueError:
        return None
    if suffix == "K":
        val *= 1_000
    elif suffix == "M":
        val *= 1_000_000
    return int(val)
