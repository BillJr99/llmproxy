"""providers.py — Loader for the providers.json sidecar.

providers.json is the single source of truth for every supported provider:
  * Provider templates (display name, base_url, key/account/gateway requirements)
  * Per-provider believed_free lists
  * Per-provider model_reasoning (exploratory / standard / deep) tags
  * Per-provider free_limits (rpm / rpd / tpm / tpd)

Providers are listed regardless of whether they offer a free tier; the
believed_free / model_reasoning / free_limits fields simply carry the
free-tier metadata for those that do.

Both setup_wizard.py and config.example.json derive from this file. The
scraper at scripts/update_free_models.py keeps the free-tier fields current.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "providers.json"

# Fields that belong to the provider-template (wizard menu) view.
_TEMPLATE_FIELDS = frozenset({
    "display",
    "base_url",
    "key_required",
    "key_hint",
    "account_id_required",
    "account_id_label",
    "account_id_hint",
    "gateway_id_required",
    "gateway_id_label",
    "gateway_id_hint",
})

# Fields that belong to the free-tier metadata view.
_FREE_INFO_FIELDS = ("believed_free", "model_reasoning", "free_limits")

VALID_REASONING_LEVELS = frozenset({"exploratory", "standard", "deep"})
FREE_LIMIT_KEYS = ("requests_per_minute", "requests_per_day",
                   "tokens_per_minute", "tokens_per_day")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data(path: Path | None = None) -> dict:
    """Load and return the raw providers.json contents."""
    p = Path(path) if path else DATA_PATH
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _cached_data() -> dict:
    return load_data()


def get_provider_templates(data: dict | None = None) -> list[dict]:
    """Return the provider templates in display order.

    Shape matches the legacy PROVIDER_TEMPLATES list-of-dicts: each entry has
    a 'key' field plus whichever template fields are present for that provider.
    """
    d = data if data is not None else _cached_data()
    order = d.get("provider_order") or list(d["providers"].keys())
    out: list[dict] = []
    for key in order:
        prov = d["providers"].get(key)
        if prov is None:
            continue
        tmpl = {"key": key}
        for field, val in prov.items():
            if field in _TEMPLATE_FIELDS:
                tmpl[field] = val
        out.append(tmpl)
    return out


def get_provider_free_info(data: dict | None = None) -> dict[str, dict]:
    """Return the per-provider free-tier metadata.

    Shape matches the legacy PROVIDER_FREE_INFO dict: provider_key →
    {believed_free, model_reasoning, free_limits}.
    """
    d = data if data is not None else _cached_data()
    return {
        key: {
            "believed_free": list(prov.get("believed_free", [])),
            "model_reasoning": dict(prov.get("model_reasoning", {})),
            "free_limits": {k: dict(v) for k, v in prov.get("free_limits", {}).items()},
        }
        for key, prov in d["providers"].items()
    }


# ---------------------------------------------------------------------------
# Reasoning-level inference (moved from setup_wizard for reuse)
# ---------------------------------------------------------------------------

_DEEP_KEYWORDS = (
    "qwq", "deepseek-r1", "deepseek-r2", "magistral",
    ":r1", "-r1", "o1-", "o3-", "reasoning",
)

_STANDARD_KEYWORDS = ("large", "medium", "mixtral", "70", "72", "32")


def infer_reasoning_level(model_id: str) -> str:
    """Infer exploratory / standard / deep from a model name.

    Used when a local provider (Ollama, OpenWebUI) reports a model that lacks
    an explicit reasoning tag, and by the scraper when adding newly-discovered
    models to model_reasoning.
    """
    s = model_id.lower()

    if any(p in s for p in _DEEP_KEYWORDS):
        return "deep"

    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", s)
    if m:
        params = float(m.group(1))
        if params >= 100:
            return "deep"
        if params >= 15:
            return "standard"
        return "exploratory"

    if any(p in s for p in _STANDARD_KEYWORDS):
        return "standard"

    return "exploratory"
