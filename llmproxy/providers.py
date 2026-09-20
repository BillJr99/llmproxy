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
    # Optional model-discovery overrides for providers whose /models endpoint
    # is non-standard (different path, different id field, or mixed tasks).
    "models_url",
    "models_id_field",
    "models_keep_task",
    # Upstream dialect for non-OpenAI-compatible providers (anthropic/gemini).
    "protocol",
})

# Fields that belong to the free-tier metadata view.
_FREE_INFO_FIELDS = ("believed_free", "model_reasoning", "model_capabilities", "free_limits")

# The reasoning tiers, ordered weakest to strongest. This tuple is the single
# source of truth: routing rank is its index (server._quality_key), and the
# virtual-endpoint name sets are comprehensions over it, so order is load
# bearing and a new tier belongs at the end. server.py and setup_wizard.py
# import it rather than restating it; tests/test_reasoning_levels.py asserts
# that every consumer agrees.
#
# flagship is an OVERLAY on top of the other three rather than a fourth
# exclusive level: a flagship model keeps its own deep/standard tag in
# model_reasoning and additionally appears in the flagship membership set, so
# promoting a model does not empty it out of llmproxy/deep.
REASONING_LEVELS: tuple[str, ...] = ("exploratory", "standard", "deep", "flagship")

# The tier membership is computed, not hand-tagged, so it is not offered in the
# wizard's manual level picker and never inferred from a model name.
OVERLAY_REASONING_LEVELS: frozenset[str] = frozenset({"flagship"})

# Levels a user may set by hand on a model (admin API, wizard, config).
VALID_REASONING_LEVELS = frozenset(REASONING_LEVELS) - OVERLAY_REASONING_LEVELS
FREE_LIMIT_KEYS = ("requests_per_minute", "requests_per_day",
                   "tokens_per_minute", "tokens_per_day")
# Per-token USD prices recorded in the top-level providers.json "pricing" block.
PRICING_KEYS = ("input_cost_per_token", "output_cost_per_token")


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
    {believed_free, model_reasoning, model_capabilities, free_limits}.
    """
    d = data if data is not None else _cached_data()
    return {
        key: {
            "believed_free": list(prov.get("believed_free", [])),
            "model_reasoning": dict(prov.get("model_reasoning", {})),
            "model_capabilities": {k: list(v) for k, v in prov.get("model_capabilities", {}).items()},
            "free_limits": {k: dict(v) for k, v in prov.get("free_limits", {}).items()},
        }
        for key, prov in d["providers"].items()
    }


def get_pricing_map(data: dict | None = None) -> dict[str, tuple[float, float]]:
    """Return the top-level ``pricing`` block as model_key → (input, output) per token.

    The pricing block is written by the scraper (scripts/sources/litellm_cost_map.py)
    and keyed by ``provider/model`` (lowercased). Missing/malformed entries are
    skipped so a hand-edited providers.json can never raise here.
    """
    d = data if data is not None else _cached_data()
    raw = d.get("pricing")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for key, val in raw.items():
        if not isinstance(key, str) or not isinstance(val, dict):
            continue
        try:
            in_cost = float(val.get("input_cost_per_token", 0) or 0)
            out_cost = float(val.get("output_cost_per_token", 0) or 0)
        except (TypeError, ValueError):
            continue
        out[key.lower()] = (in_cost, out_cost)
    return out


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


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

# How strongly a learned routing fact is believed, weakest to strongest. The
# sidecar carries one of these beside each fact as ``<fact>_source``, because it
# holds machine-learned and hand-set values together and a refresh must be able
# to tell them apart. Without it the inference pass would overwrite the very
# corrections a user made to fix what inference got wrong.
#
#   inferred  derived from this model's own name
#   family    unanimous across the models sharing its family
#   observed  a provider listing or the OpenRouter catalog said so
#   curated   set by hand, in the admin UI or migrated from a config.json
#
# Order is load bearing: ``fact_rank`` is the index, and a refresh may replace a
# fact of equal or weaker rank and never a stronger one.
FACT_SOURCES: tuple[str, ...] = ("inferred", "family", "observed", "curated")

# What a fact with no recorded provenance counts as. Sidecars written before
# provenance existed carry none, and treating those as the weakest grade would
# let the first refresh overwrite hand-migrated data — exactly the loss this
# field exists to prevent. They are therefore read as "observed".
DEFAULT_FACT_SOURCE = "observed"


def fact_rank(source: str | None) -> int:
    """Rank a provenance grade. Unknown or missing reads as DEFAULT_FACT_SOURCE."""
    if not isinstance(source, str) or source not in FACT_SOURCES:
        source = DEFAULT_FACT_SOURCE
    return FACT_SOURCES.index(source)


# ---------------------------------------------------------------------------
# Family derivation
# ---------------------------------------------------------------------------

# Tokens that mark a size or a tuning rather than a generation, so they end the
# family rather than extending it.
_FAMILY_VERSION_RE = re.compile(r"^[a-z]\d+$")     # "m2", "r1" — a lettered series
_FAMILY_SPLIT_RE = re.compile(r"[-_.]+")

# At most this many leading alphabetic tokens form the base ("gpt"+"oss",
# "command"+"a"), and at most this many numeric tokens form the generation
# ("5"+"3" for glm-5.3). Both caps stop a long descriptive id from collapsing
# into a family of one.
_FAMILY_MAX_BASE_TOKENS = 2
_FAMILY_MAX_VERSION_TOKENS = 2


def family_key(model_id: str, *, generation: bool = True) -> str | None:
    """Group a model with its siblings: ``z-ai/glm-5.3-flash`` -> ``glm5``/``glm``.

    Two granularities. With *generation* the key carries the major version, so
    ``llama-2-7b`` and ``llama-4-scout`` land in different families and cannot
    lend each other capabilities they do not share. Without it they share
    ``llama``, which is coarser but covers models whose generation is too sparse
    to say anything on its own.

    Derived from the RAW id, never from ``normalize_model_id``'s output. That
    function strips separators, so ``llama-2-7b`` becomes ``llama27b`` and any
    version read afterwards is a number that was never a version: the family
    would come out ``llama27``. Same hazard as ``infer_reasoning_level``.

    Returns None when the id yields nothing usable to group on.
    """
    if not isinstance(model_id, str):
        return None
    s = model_id.lower().strip().split("/")[-1]
    s = s.split(":", 1)[0]
    tokens = [t for t in _FAMILY_SPLIT_RE.split(s) if t]
    if not tokens:
        return None

    # The first token may already embed its generation ("qwen3"), so peel it.
    head = tokens[0]
    m = re.match(r"^([a-z]+)(\d*)$", head)
    if not m or len(m.group(1)) < 2:
        return None
    base, version = m.group(1), m.group(2)

    base_tokens = 1
    version_tokens = 1 if version else 0
    for tok in tokens[1:]:
        if not version and tok.isalpha() and base_tokens < _FAMILY_MAX_BASE_TOKENS:
            base += tok                       # "gpt"+"oss", "command"+"a"
            base_tokens += 1
            continue
        if tok.isdigit() and version_tokens < _FAMILY_MAX_VERSION_TOKENS:
            version += tok                    # "5"+"3" for glm-5.3
            version_tokens += 1
            continue
        if not version and _FAMILY_VERSION_RE.match(tok):
            version += tok                    # "m2" for minimax-m2.7
            version_tokens += 1
            continue
        break

    if not generation:
        return base or None
    return (base + version) or None


# ---------------------------------------------------------------------------
# Capability derivation
# ---------------------------------------------------------------------------

def capabilities_from_listing(model: dict) -> set[str]:
    """Derive llmproxy capability tags from one provider's model listing entry.

    Reads the two fields OpenAI-compatible gateways actually publish:
    ``supported_parameters`` (tools / reasoning / structured output) and
    ``architecture.input_modalities`` (image -> vision). Both are optional, and
    a bare OpenAI-shaped object carries neither.

    Returns an EMPTY set for an entry that says nothing, which callers must
    treat as "unknown" rather than "incapable" — guessing from the model's name
    would manufacture facts, and the three-valued capability lookup in the
    router depends on absence being distinguishable from denial.

    One implementation shared by the scraper, the route-cache rebuild and the
    routing-metadata refresh, so the three cannot drift apart.
    """
    if not isinstance(model, dict):
        return set()
    supported = model.get("supported_parameters")
    if not isinstance(supported, list):
        supported = []
    arch = model.get("architecture")
    modalities = (arch or {}).get("input_modalities") if isinstance(arch, dict) else None
    if not isinstance(modalities, list):
        modalities = []

    caps: set[str] = set()
    if "tools" in supported:
        caps.add("tools")
    if "reasoning" in supported:
        caps.add("reasoning")
    if "structured_outputs" in supported or "response_format" in supported:
        caps.add("json")
    if "image" in modalities:
        caps.add("vision")
    return caps
