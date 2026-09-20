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
   on a different scale cannot swamp the existing one. The combined percentile
   is *kept*, not only used for admission: it is what orders the tier at
   request time, so a flagship pool is walked strongest-first.
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

from . import USER_AGENT

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


def has_variant_suffix(model_id: str) -> bool:
    """Whether *model_id* names a billing or routing variant of another model.

    True for ``z-ai/glm-5.2:free``, ``deepseek/deepseek-v4:batch`` and the
    ``-free`` spelling some gateways use; False for a plain id.

    The check mirrors the double application inside ``normalize_model_id``: the
    suffix is tested against the raw id and again against the post-``/`` tail,
    because a vendor path may carry the variant on either side.

    Callers use this to decide when NOT to join a fact across the normalized
    key. A provider that lists both ``z-ai/glm-5.2`` and ``z-ai/glm-5.2:free``
    is discriminating between two routing targets rather than being terse about
    one, so what it says about the variant is about the variant alone.
    """
    if not model_id:
        return False
    s = model_id.lower().strip()
    for prefix in _PATH_PREFIXES:
        s = s.replace(prefix, "")
    return bool(_VARIANT_RE.search(s) or _VARIANT_RE.search(s.split("/")[-1]))


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


def passes_spec_gate(cand: Candidate, min_context: int, require_tools: bool = False) -> bool:
    """Whether *cand* clears the tier's hard minimums.

    The context floor is the substantive check: a window too small to hold an
    agentic conversation makes a model unusable for this tier whatever it
    scores. Unknown context fails it, since admitting a model we cannot verify
    would quietly fill the tier with whatever a provider happens not to
    document; a pin is the deliberate way to override that.

    ``require_tools`` is an opt-in capability veto and is **off by default**. It
    used to default on, and that was the wrong layer. A request that needs tools
    is already prevented from selecting a model that cannot serve it, per
    request, by the server's capability gate — so vetoing at MEMBERSHIP time
    was both redundant and harmful: it made the floating bar hunt further down
    the ranking for free models that happened to clear the veto, filling the
    tier with weaker models on the strength of a capability tag rather than a
    benchmark score. Turn it on when you want the tier itself restricted to
    tool-callers; leave it off to have membership decided on merit and
    capability decided per request.
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
    # Lowercased qualified "provider/model" -> {"combined": float, "model_key": str}.
    # One entry per member we could score, which is what lets the router walk the
    # tier strongest-first instead of in the arbitrary order of `members`.
    scores: dict[str, dict] = field(default_factory=dict)
    # Lowercased qualified id -> the percentile a pin assigned it. Reported so
    # the refresh can say which placements displaced a measured score, which is
    # otherwise invisible: the override is silent by design.
    pinned_placements: dict[str, float] = field(default_factory=dict)
    # Normalised model key -> combined percentile, for EVERY scored model this
    # deployment can see, admitted or not. A score belongs to the weights rather
    # than to the provider serving them, so a routing target that appeared after
    # the last refresh — or a pinned provider no leaderboard covers by name —
    # can still join its score through this map.
    model_scores: dict[str, float] = field(default_factory=dict)


# A pin may carry a placement as well as a name. Percentiles live in [0, 1]
# internally, matching `start_percentile`, but "98" is the natural thing to
# write, so both are accepted under one unambiguous rule: a value <= 1 is a
# fraction, a value > 1 is a percentage. There is no boundary case where the
# reading is in doubt.
_PIN_PERCENTILE_MAX_FRACTION = 1.0


def parse_pins(raw) -> dict[str, float | None]:
    """``flagship_tier.pin`` as ``{lowercased name: percentile or None}``.

    Two entry shapes, freely mixed in one list::

        "pin": ["atria-asi/Atria-Dawn-Preview",
                {"name": "Atria-Dawn-Preview", "percentile": 98}]

    A bare string pins without saying where to place it, which is the original
    behaviour: nothing scores it, so it sorts after every scored candidate. That
    is safe but it is rarely what the author wanted — you pin a model because
    you want it used, and it ends up last in the failover queue.

    A percentile says where it belongs. It is honoured even when a benchmark
    also scores the model, because a pin is an explicit instruction rather than
    a hint; that is what makes it possible to demote a model you distrust as
    well as promote one you hold credits on. The caller logs when an override
    actually displaces a measured score.

    Malformed entries degrade to a plain pin rather than raising: a typo in a
    percentile should cost placement, not the tier.
    """
    out: dict[str, float | None] = {}
    for entry in raw or []:
        if isinstance(entry, str):
            out.setdefault(entry.lower(), None)
            continue
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model") or entry.get("id")
        if not isinstance(name, str) or not name.strip():
            continue
        out[name.strip().lower()] = coerce_pin_percentile(entry.get("percentile"))
    return out


def coerce_pin_percentile(raw) -> float | None:
    """A pin's ``percentile`` as a [0, 1] fraction, or None when unusable.

    ``bool`` is rejected explicitly: it is an ``int`` subclass, so ``True``
    would otherwise read as the 100th percentile.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value > _PIN_PERCENTILE_MAX_FRACTION:
        value /= 100.0
    if value < 0.0:
        return None
    # A percentile above the top of the scale is a typo (`percentile: 980`),
    # and clamping keeps it meaning "first" rather than silently sorting it
    # somewhere arbitrary.
    return min(value, 1.0)


def select_flagship(candidates: list[Candidate], tier_cfg: dict) -> Selection:
    """Choose the flagship membership for this deployment.

    Returns qualified ``provider/model`` ids, one per routing target, so a model
    served by several providers contributes several members even though it
    counts once toward ``min_flagship_free_models``. The combined percentile
    that admitted each one is returned alongside, keyed both per routing target
    and per model, so the router can order the tier without recomputing it.

    ``pin`` accepts a qualified id or a bare upstream id; the latter admits
    every provider serving that model. See the pin block below for how the two
    are told apart, and why the normalized key is not a third option.
    """
    min_context = int(tier_cfg.get("min_context") or 0)
    require_tools = bool(tier_cfg.get("require_tools", False))
    min_free = int(tier_cfg.get("min_flagship_free_models") or 0)
    start_percentile = float(tier_cfg.get("start_percentile") or 0.0)
    max_models = tier_cfg.get("max_models")
    pin_percentiles = parse_pins(tier_cfg.get("pin"))
    pin = set(pin_percentiles)
    exclude = {e.lower() for e in (tier_cfg.get("exclude") or []) if isinstance(e, str)}

    combined = combine_scores(candidates)

    # Eligible = not excluded, passes the veto, and has a score. Pins are
    # handled separately precisely because they need neither.
    #
    # Excluding here rather than only at the end is what makes
    # `min_flagship_free_models` a statement about the tier instead of about
    # the walk. The walk below stops as soon as that many distinct free models
    # have been admitted, so an excluded target counted along the way left the
    # tier one member short of the floor it had just promised, silently,
    # because the exclusion only took effect afterwards.
    #
    # It also fixes `free_keys`: a model whose only free routing target is
    # excluded no longer counts as free, which is precisely the case a
    # `cost_observed_free_tier` correction creates.
    eligible = [
        c for c in candidates
        if c.qualified.lower() not in exclude
        and c.model_key in combined and passes_spec_gate(c, min_context, require_tools)
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
    #
    # A pin may be written either way, matching `believed_free`, the key it is
    # normally paired with:
    #
    #   "gmi/google/gemini-3.7-flash"  -> that one routing target
    #   "gemini-3.7-flash"             -> every provider serving that model
    #
    # `/` cannot tell the two apart — upstream ids routinely contain one
    # (`gmi` serves `google/gemini-3.8-flash`, `openrouter` serves
    # `qwen/qwen3.8-27b:free`) — so resolution is by precedence instead:
    # an exact qualified match first, the bare upstream id second. The more
    # specific reading wins, deterministically.
    #
    # The NORMALIZED key is deliberately not matched. That join is heuristic,
    # and a wrong one would pin a model the user never named — the same failure
    # the spec gate was fixed to stop making. A pin is an explicit instruction
    # and stays literal.
    known = {c.qualified.lower() for c in candidates}
    by_upstream: dict[str, set[str]] = {}
    for c in candidates:
        by_upstream.setdefault(c.upstream_id.lower(), set()).add(c.qualified.lower())

    pinned_targets: set[str] = set()
    verified_pins: set[str] = set()
    for p in pin:
        if p in known:
            pinned_targets.add(p)
            verified_pins.add(p)
        elif p in by_upstream:
            pinned_targets |= by_upstream[p]
            verified_pins.add(p)
        else:
            # Unresolvable against anything this deployment can see. Honoured
            # anyway, per the note above, and reported to the caller.
            pinned_targets.add(p)

    members |= pinned_targets
    # Applied last, so an exclude still beats a pin — including beating one arm
    # of an expanded bare pin, which is how you pin a model everywhere except
    # on the one provider whose copy of it is broken.
    members -= exclude

    # Built from `members` rather than from `eligible`, so a pin that a source
    # happens to cover is still rankable even though it bypassed the bar. A pin
    # nothing scores, and which names no percentile, simply has no entry, and
    # the router sorts it last.
    scores = {
        c.qualified.lower(): {"combined": combined[c.model_key],
                              "model_key": c.model_key}
        for c in candidates
        if c.qualified.lower() in members and c.model_key in combined
    }

    # A pin that names a percentile places itself. Written last so it wins over
    # a measured score: a pin is an explicit instruction, which is what lets it
    # demote a model you distrust as well as promote one nothing has scored.
    # `pinned_placements` reports every override so the caller can say which
    # ones displaced real evidence.
    key_of = {c.qualified.lower(): c.model_key for c in candidates}
    pinned_placements: dict[str, float] = {}
    for name, percentile in pin_percentiles.items():
        if percentile is None:
            continue
        # One pin may name a bare model and expand to several routing targets;
        # each inherits the placement.
        targets = by_upstream.get(name, {name}) if name not in known else {name}
        for target in targets:
            if target not in members:
                continue
            pinned_placements[target] = percentile
            scores[target] = {"combined": percentile,
                              "model_key": key_of.get(target, target),
                              "pinned": True}

    # Reported from the FINAL member set, after pins and excludes, rather than
    # from `admitted`. What is written to flagship_models.json — and read back
    # by the floor check — has to describe the tier that will actually be
    # served, not the state of the walk halfway through it. `bar` still reports
    # where the walk stopped, which is a different and still useful fact.
    member_key_of: dict[str, str] = {}
    member_free_keys: set[str] = set()
    for c in candidates:
        q = c.qualified.lower()
        if q not in members:
            continue
        member_key_of[q] = c.model_key
        if c.is_free:
            member_free_keys.add(c.model_key)

    # Strongest first, unscored last, so the order still means something to a
    # reader of the cache file. A pin nothing scored has no percentile and
    # sorts to the end rather than to the front.
    final_models = sorted(
        set(member_key_of.values()),
        key=lambda k: (k in combined, combined.get(k, 0.0)),
        reverse=True,
    )

    return Selection(
        members=sorted(members),
        bar=bar,
        distinct_models=final_models,
        free_models=[k for k in final_models if k in member_free_keys],
        pinned=sorted(pin),
        pinned_placements=pinned_placements,
        unverified_pins=sorted(pin - verified_pins),
        scores=scores,
        model_scores=dict(combined),
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
    "supports_tools": bool, "capabilities": set[str], "model_id": str,
    "by_id": {exact_id: {"supports_tools": bool, "context_length": int|None}}}}``.
    The merged specs come from every entry sharing the key, so a provider that
    publishes no capability data of its own can still be gated on the same
    weights served elsewhere.

    ``by_id`` records what the catalog said about each EXACT id, unmerged. Both
    are needed and neither replaces the other. The merged view is the right
    answer for a provider the catalog does not list at all, which is the
    cross-provider carry-across the tier depends on. It is the wrong answer for
    a billing variant the catalog lists separately: ``z-ai/glm-5.2:free``
    normalizes onto ``z-ai/glm-5.2``, so without the exact-id view it inherits
    tool support and a 1M context window from its paid sibling and is admitted
    to a tier it cannot serve, failing at request time with an upstream 404.

    ``capabilities`` is the broad base the routing-metadata refresh joins under
    its normalized key. It is derived with the same
    ``providers.capabilities_from_listing`` the route-cache rebuild and the
    scraper use, so the three cannot disagree about what a listing means, and it
    is UNIONED across every catalog entry sharing a key — one gateway spelling
    publishing a thinner ``supported_parameters`` must not retract what another
    asserted. ``model_id`` keeps one raw id per key, because every inference
    downstream has to run on the raw id rather than the normalized one.
    """
    import requests

    from .providers import capabilities_from_listing

    # A third-party catalog behind a CDN will refuse a bare library
    # default as readily as a provider will.
    resp = requests.get(url, timeout=_FETCH_TIMEOUT,
                        headers={"User-Agent": USER_AGENT})
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
                                       "supports_tools": False,
                                       "capabilities": set(), "model_id": mid,
                                       "by_id": {}})
        profile["capabilities"] |= capabilities_from_listing(model)
        ctx_raw = model.get("context_length")
        # This entry's own claim, from this entry alone: no latch, no max-wins.
        # A variant that says less than its sibling is asserting a difference.
        profile["by_id"][mid.lower()] = {
            "supports_tools": "tools" in supported,
            "context_length": ctx_raw,
        }
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
                                           "supports_tools": False,
                                           "capabilities": set(), "model_id": None,
                                           "by_id": {}})
            slot["scores"].update(profile.get("scores") or {})
            slot["capabilities"] |= set(profile.get("capabilities") or ())
            if not slot["model_id"]:
                slot["model_id"] = profile.get("model_id")
            ctx = profile.get("context_length")
            if ctx and (slot["context_length"] or 0) < ctx:
                slot["context_length"] = ctx
            if profile.get("supports_tools"):
                slot["supports_tools"] = True
            # Two sources describing the SAME exact id is the terse-listing
            # case, not the variant case — one of them may simply publish less
            # about a model both cover. So the module's monotone semantics
            # apply within an id: tool support latches on, context takes the
            # larger. Across DIFFERENT ids nothing is shared, which is the
            # whole point of keeping this map beside the merged fields.
            for exact_id, spec in (profile.get("by_id") or {}).items():
                entry = slot["by_id"].setdefault(
                    exact_id, {"supports_tools": False, "context_length": None}
                )
                if spec.get("supports_tools"):
                    entry["supports_tools"] = True
                ctx = spec.get("context_length")
                if ctx and (entry["context_length"] or 0) < ctx:
                    entry["context_length"] = ctx
    return merged
