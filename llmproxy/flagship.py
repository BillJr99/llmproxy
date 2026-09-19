"""Selection of the `flagship` tier — the computed top tier above `deep`.

Flagship holds the models near the current state of the art, so
``llmproxy/flagship`` and ``flagship/free`` route only to models capable of
driving an agent loop. Membership is never hardcoded: it depends on which
providers a deployment has configured and what each of them currently serves,
so it is recomputed locally and cached in ``flagship_models.json``.

The rules, in the order they apply:

1. **Benchmarks rank.** Sources may use unrelated scales, so each is
   rank-normalised to a percentile over the models it covers and the
   percentiles are combined. Raw scores are never averaged, so adding a source
   on a different scale cannot swamp the existing one.
2. **Spec gates veto.** Tool-calling and a context floor. These barely
   discriminate on their own, so they are a veto rather than a selector: a
   model that cannot call tools cannot drive an agent loop whatever it scores.
3. **The bar floats.** Starting from ``start_percentile``, the bar is lowered
   until at least ``min_flagship_free_models`` DISTINCT free models qualify.
   Cross-provider duplicates count once for that floor, because three providers
   serving the same weights is one model's worth of capability — but each
   provider's instance is still its own routing target, since each has its own
   quota, rate limits and outage profile.
4. **Pins and excludes win.** A pin bypasses both the bar and the veto, which
   is the only way to admit a provider no benchmark covers. An exclude is
   applied last and beats everything.

Free status is per-provider throughout: the same weights may be free on one
provider and paid on another, so ``is_free`` is a property of the candidate,
not of the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

# Suffixes that denote a billing or routing variant of the SAME underlying
# model, not a different one. Collapsed before ranking so a model does not
# occupy several slots — roughly half of a raw catalog listing is ``:batch``.
_VARIANT_SUFFIXES = ("batch", "free", "extended", "nitro", "floor")
_VARIANT_RE = re.compile(r"[:\-](?:{})$".format("|".join(_VARIANT_SUFFIXES)))

# Prefixes some providers wrap around an otherwise-standard model path.
_PATH_PREFIXES = ("@cf/", "accounts/fireworks/models/")

# Qualifiers that describe the tuning rather than the model family, dropped so
# the same weights served under slightly different names still join.
_TRAILING_QUALIFIERS = re.compile(
    r"-(?:instruct|it|chat|versatile|preview|latest|hf)\b"
)


def normalize_model_id(model_id: str) -> str:
    """Collapse a provider-qualified id to a key that joins across providers.

    ``cloudflare-workers/@cf/zai-org/glm-5.3`` and ``openrouter/z-ai/glm-5.3``
    both reduce to ``glm53``, so a score learned for one provider's listing
    applies to the same weights elsewhere.

    This is deliberately lossy and therefore heuristic. A false join would
    admit a model on another's reputation, so callers should treat a joined
    score as weaker evidence than a native one.
    """
    s = model_id.lower().strip()
    for prefix in _PATH_PREFIXES:
        s = s.replace(prefix, "")
    s = _VARIANT_RE.sub("", s)
    s = s.split("/")[-1]
    s = _VARIANT_RE.sub("", s)
    s = _TRAILING_QUALIFIERS.sub("", s)
    return re.sub(r"[^a-z0-9]", "", s)


@dataclass
class Candidate:
    """One routing target: a specific model on a specific provider."""

    provider: str
    upstream_id: str
    is_free: bool = False
    context_length: int | None = None
    supports_tools: bool | None = None
    # source name -> raw score, on that source's own scale.
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def qualified(self) -> str:
        return f"{self.provider}/{self.upstream_id}"

    @property
    def model_key(self) -> str:
        return normalize_model_id(self.upstream_id)


def _percentiles(values: dict[str, float]) -> dict[str, float]:
    """Rank-normalise ``{key: raw}`` to ``{key: percentile in [0, 1]}``.

    Ties share the lower percentile so equal scores rank equally. A single
    value is defined as 1.0: it is the whole field, therefore the top of it.
    """
    if not values:
        return {}
    ordered = sorted(values.values())
    n = len(ordered)
    if n == 1:
        return dict.fromkeys(values, 1.0)
    out: dict[str, float] = {}
    for key, raw in values.items():
        below = sum(1 for v in ordered if v < raw)
        out[key] = below / (n - 1)
    return out


def combine_scores(candidates: list[Candidate]) -> dict[str, float]:
    """Combined percentile per distinct model, across all sources.

    Each source is ranked over the models it actually covers, then a model's
    per-source percentiles are combined with a median. Using the median rather
    than a mean keeps one outlying source from dominating, and a model missing
    from a source is simply ranked on the sources that do cover it rather than
    being penalised for the gap.
    """
    by_source: dict[str, dict[str, float]] = {}
    for cand in candidates:
        for source, raw in cand.scores.items():
            if raw is None:
                continue
            # Several providers may carry a score for the same weights; keep
            # the best, since a lower one usually means a stale listing.
            prev = by_source.setdefault(source, {}).get(cand.model_key)
            if prev is None or raw > prev:
                by_source[source][cand.model_key] = raw

    ranked = {source: _percentiles(vals) for source, vals in by_source.items()}
    combined: dict[str, float] = {}
    for cand in candidates:
        pcts = [r[cand.model_key] for r in ranked.values() if cand.model_key in r]
        if pcts:
            combined[cand.model_key] = median(pcts)
    return combined


def passes_spec_gate(cand: Candidate, min_context: int, require_tools: bool) -> bool:
    """Whether *cand* can plausibly drive an agent loop.

    Unknown capability data fails the gate. Admitting a model we cannot verify
    would quietly fill the tier with whatever a provider happens not to
    document; a pin is the deliberate way to override that.
    """
    if require_tools and not cand.supports_tools:
        return False
    if min_context and (cand.context_length or 0) < min_context:
        return False
    return True


@dataclass
class Selection:
    """Result of a selection run, ready to cache."""

    members: list[str]
    bar: float | None
    distinct_models: list[str]
    free_models: list[str]
    pinned: list[str]
    unverified_pins: list[str]


def select_flagship(candidates: list[Candidate], tier_cfg: dict) -> Selection:
    """Choose the flagship membership for this deployment.

    Returns qualified ``provider/model`` ids, one per routing target, so a model
    served by several providers contributes several members even though it
    counts once toward ``min_flagship_free_models``.
    """
    min_context = int(tier_cfg.get("min_context") or 0)
    require_tools = bool(tier_cfg.get("require_tools", True))
    min_free = int(tier_cfg.get("min_flagship_free_models") or 0)
    start_percentile = float(tier_cfg.get("start_percentile") or 0.0)
    max_models = tier_cfg.get("max_models")
    pin = {p.lower() for p in (tier_cfg.get("pin") or []) if isinstance(p, str)}
    exclude = {e.lower() for e in (tier_cfg.get("exclude") or []) if isinstance(e, str)}

    combined = combine_scores(candidates)

    # Eligible = passes the veto and has a score. Pins are handled separately
    # precisely because they need neither.
    eligible = [
        c for c in candidates
        if c.model_key in combined and passes_spec_gate(c, min_context, require_tools)
    ]

    # Distinct models, strongest first. The bar moves over this list, so the
    # unit of the floor is the model rather than the routing target.
    ranked_models = sorted(
        {c.model_key for c in eligible},
        key=lambda k: combined[k],
        reverse=True,
    )
    free_keys = {c.model_key for c in eligible if c.is_free}

    # Walk down until the start percentile is satisfied AND the free floor is
    # met. There is no lower bound: the floor is honoured as far as the pool
    # allows, and if the pool runs out the tier is simply smaller.
    admitted: list[str] = []
    free_count = 0
    for key in ranked_models:
        above_start = combined[key] >= start_percentile
        need_more_free = free_count < min_free
        if not above_start and not need_more_free:
            break
        admitted.append(key)
        if key in free_keys:
            free_count += 1

    if max_models:
        admitted = admitted[: int(max_models)]

    admitted_set = set(admitted)
    bar = combined[admitted[-1]] if admitted else None

    members = {
        c.qualified.lower() for c in eligible if c.model_key in admitted_set
    }

    # Pins bypass the bar and the veto. A pinned id that no candidate matches
    # is still honoured — the model may be reachable even though nothing we
    # scraped describes it — but it is reported so the caller can warn.
    known = {c.qualified.lower() for c in candidates}
    verified_pins = {p for p in pin if p in known}
    members |= pin
    members -= exclude

    return Selection(
        members=sorted(members),
        bar=bar,
        distinct_models=admitted,
        free_models=[k for k in admitted if k in free_keys],
        pinned=sorted(pin),
        unverified_pins=sorted(pin - verified_pins),
    )


# ---------------------------------------------------------------------------
# Score sources
# ---------------------------------------------------------------------------
#
# Only sources whose terms permit this use, and which work without credentials
# a user would have to go and obtain, are wired up:
#
#   * LLM Stats forbids redistribution on every tier ("Technical access is not
#     a redistribution license").
#   * BenchLM states no licence at all, which is an absence of any grant
#     rather than a restrictive one.
#   * Epoch AI publishes under CC-BY, but its `epochai` client is an Airtable
#     ORM: it reads AIRTABLE_PERSONAL_ACCESS_TOKEN and AIRTABLE_BASE_ID at
#     import time and raises without them, and its data model is benchmark
#     *runs* rather than a leaderboard. Installing it does not make the source
#     work, it just fails quietly, so it is not a dependency. Epoch's public
#     CC-BY CSV export would be the way to add it, keyed the same way and
#     credited per the licence.
#
# Nothing fetched here is committed: scores live only in the local
# flagship_models.json cache.

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_FETCH_TIMEOUT = (5, 15)


def fetch_openrouter_profiles(url: str = OPENROUTER_MODELS_URL) -> dict[str, dict]:
    """Benchmark scores and capability specs, keyed by join key.

    OpenRouter's catalog embeds Artificial Analysis indices under
    ``benchmarks.artificial_analysis``, so the scores arrive with the model
    listing we already fetch each sweep: no second scraper, no extra API key,
    and ids that match by construction. ``agentic_index`` is used because it is
    the closest published measure of what this tier is for — driving a tool
    loop — where chat-arena style ratings measure conversational preference.

    Returns ``{model_key: {"scores": {...}, "context_length": int|None,
    "supports_tools": bool}}``. Specs come from the same entry so a provider
    that publishes no capability data of its own can still be gated on the
    same weights served elsewhere.
    """
    import requests

    resp = requests.get(url, timeout=_FETCH_TIMEOUT)
    resp.raise_for_status()
    out: dict[str, dict] = {}
    for model in resp.json().get("data", []):
        mid = model.get("id")
        if not mid:
            continue
        key = normalize_model_id(mid)
        bench = (model.get("benchmarks") or {}).get("artificial_analysis") or {}
        score = bench.get("agentic_index")
        supported = model.get("supported_parameters") or []
        profile = out.setdefault(key, {"scores": {}, "context_length": None,
                                       "supports_tools": False})
        if score is not None:
            prev = profile["scores"].get("openrouter_aa")
            if prev is None or score > prev:
                profile["scores"]["openrouter_aa"] = float(score)
        ctx = model.get("context_length")
        if ctx and (profile["context_length"] or 0) < ctx:
            profile["context_length"] = ctx
        if "tools" in supported:
            profile["supports_tools"] = True
    return out


_SOURCE_FETCHERS = {
    "openrouter_aa": fetch_openrouter_profiles,
}


def fetch_profiles(sources: list[str] | None = None) -> dict[str, dict]:
    """Merge every enabled source into one ``{model_key: profile}`` map.

    A source that fails is skipped rather than failing the refresh, so one
    unreachable leaderboard degrades the ranking instead of emptying the tier.
    Returns the merged map and leaves the caller to log per-source outcomes.
    """
    merged: dict[str, dict] = {}
    for name in (sources or list(_SOURCE_FETCHERS)):
        fetcher = _SOURCE_FETCHERS.get(name)
        if fetcher is None:
            continue
        try:
            profiles = fetcher()
        except Exception:  # noqa: BLE001 — see docstring
            continue
        for key, profile in profiles.items():
            slot = merged.setdefault(key, {"scores": {}, "context_length": None,
                                           "supports_tools": False})
            slot["scores"].update(profile.get("scores") or {})
            ctx = profile.get("context_length")
            if ctx and (slot["context_length"] or 0) < ctx:
                slot["context_length"] = ctx
            if profile.get("supports_tools"):
                slot["supports_tools"] = True
    return merged
