"""
server.py — OpenAI-compatible proxy server for llmproxy.

Implements the following OpenAI API endpoints:
  GET  /v1/models                  Aggregate models from all providers
  GET  /v1/models/<model_id>       Single model metadata lookup
  POST /v1/chat/completions        Proxy chat completions (streaming + non-streaming)
  POST /v1/completions             Proxy legacy completions (chat/completions fallback)
  POST /v1/embeddings              Proxy embeddings
  GET  /v1/usage                   Token + cost accounting report
  POST /v1/usage/reset             Clear usage counters (admin-gated)
  GET  /health                     Health check

Model naming convention
-----------------------
GET /v1/models advertises every model in the canonical display form:
    <provider_name>__<upstream_model_id>

For example, an "ollama" provider serving "qwen2.5vl:3b" is shown as
"ollama__qwen2.5vl:3b".  Spaces in either side are replaced with "_".  The
"__" (double underscore) is the unambiguous provider separator; a single "/"
may still appear *inside* the upstream model portion (e.g.
"openrouter__deepseek/deepseek-chat-v3").  This keeps the advertised id free of
a leading "provider/…" segment, which matters for clients that group their
model picker by the text before the first "/" (e.g. opencode) — a leading slash
would collapse every model under one provider group.

Upstream ids that contain multiple slashes are flattened so the display id
carries at most one "/": all but the last slash become "_".  For example an
"openrouter" provider serving "meta-llama/llama-3/instruct" is shown as
"openrouter__meta-llama_llama-3/instruct".  Routing always uses the original
(un-flattened) upstream id when forwarding upstream.

Virtual models (the reserved "llmproxy" namespace) are the exception: they are
advertised in the "llmproxy/<name>" slash form, with any "/" inside <name>
encoded as "__" (e.g. "llmproxy/deep__free", "llmproxy/loadbalanced").  This puts
every virtual under one "llmproxy" picker group with a distinct label per entry
instead of collapsing them.  Each virtual also carries a human-readable,
slash-free ``name`` (e.g. "[llmproxy] Deep — Free") for UIs that display the
``name`` field.

The following input forms are also accepted on every proxied endpoint:
    <provider_name>/<upstream_model_id>     (slash form; interior "/" as "__")
    <upstream_model_id>__<provider_name>    (PR #27 legacy display form)
    <upstream_model_id> (<provider_name>)   (pre-PR #27 legacy display form)
For virtual models the canonical "llmproxy__<name>" form, the legacy three-part
"llmproxy/<name>/<dimension>" form, and an all-"__" spelling are also accepted.

The server strips the provider prefix/suffix before forwarding each request
to the appropriate upstream base URL.
"""

import contextlib
import datetime
import hashlib
import io
import itertools
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import (
    Flask,
    Response,
    g,
    has_request_context,
    jsonify,
    make_response,
    request,
    stream_with_context,
)

from . import USER_AGENT, __version__
from . import fusion as _fusion
from . import route_report as _route_report
from .config import (
    DEFAULT_FREE_TIER_CONFIG,
    RESERVED_PROVIDER_NAMES,
    account_bound_cfg,
    flagship_tier_cfg,
    get_config_path,
    get_provider,
    get_routing_metadata_path,
    load_config,
    load_flagship_state,
    load_routing_metadata,
    model_is_allowed,
    parse_model_string,
    provider_account_id,
    provider_account_strategy,
    provider_accounts,
    provider_api_key,
    provider_base_url,
    resolve_env_refs,
    routing_metadata_cfg,
    save_config,
    save_flagship_state,
    save_routing_metadata,
)
from .dialects import get_inbound, get_outbound
from .dialects.responses import UnknownPreviousResponse
from .providers import (
    OVERLAY_REASONING_LEVELS,
    REASONING_LEVELS,
    capabilities_from_listing,
    get_provider_free_info,
)
from .signals import (
    SOURCE_NEUTRAL,
    extract_tool_signals,
    score_signals,
    tier_adjustment,
)

try:
    import fcntl  # POSIX advisory file locking
except ImportError:  # pragma: no cover - non-POSIX (e.g. Windows)
    fcntl = None
from .usage import (
    ModelUsage,
    compute_cost,
    extract_usage,
    load_pricing_map,
    parse_stream_usage,
)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)
logger = logging.getLogger("llmproxy.server")

# The per-request audit channel, deliberately separate from `logger`.
#
# It writes bare JSON — one object per line, no level, no timestamp prefix — to
# **stdout**, while everything else llmproxy logs goes to stderr. That split is
# the point: `docker logs` and every collector can tee the two apart, so a
# machine-readable record stream never has to be grepped out of human log
# chatter, and turning the audit on does not change the operational log at all.
#
# ``propagate = False`` keeps these lines out of the root handler that
# ``logging.basicConfig`` installs, which would otherwise print each record a
# second time on stderr with a level prefix, and ``_request_log_mode`` gates
# emission rather than the log level, so the record stream is independent of
# ``server.log_level``.
request_logger = logging.getLogger("llmproxy.requests")
request_logger.propagate = False
if not request_logger.handlers:
    _request_log_handler = logging.StreamHandler(sys.stdout)
    _request_log_handler.setFormatter(logging.Formatter("%(message)s"))
    request_logger.addHandler(_request_log_handler)
    # NOTSET so the channel inherits the root level that logging.basicConfig
    # sets from server.log_level. Pinning it to INFO here would swallow the
    # body-carrying records even with log_level: DEBUG, since propagate is off
    # and this logger's own level would be the one that decides.
    request_logger.setLevel(logging.NOTSET)


class _StripApiPrefix:
    """Mirror every API route under an ``/api`` prefix.

    Many clients (OpenRouter-, Open WebUI-, and Ollama-style) assume the API
    lives under ``/api`` or ``/api/v1`` and probe e.g. ``/api/v1/models`` before
    falling back. Rather than duplicating ``@app.route`` decorators, this WSGI
    shim strips a leading ``/api`` from PATH_INFO so ``/api/v1/...`` and
    ``/api/v1beta/...`` reach the same handlers as ``/v1/...``. The original
    ``/v1`` surface is unchanged.

    ``/api/admin`` is intentionally NOT aliased: the admin UI/API stays reachable
    only at its canonical ``/admin`` path to keep that surface area small.
    """

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == "/api" or path.startswith("/api/"):
            stripped = path[4:] or "/"
            if not stripped.startswith("/admin"):
                environ["PATH_INFO"] = stripped
        return self.wsgi_app(environ, start_response)


app.wsgi_app = _StripApiPrefix(app.wsgi_app)

# Canonical tier order lives in llmproxy/providers.py; see REASONING_LEVELS
# there for why it is ordered and why flagship is an overlay. Aliased here
# because the virtual-model name sets below are comprehensions over it.
_REASONING_LEVELS: tuple[str, ...] = REASONING_LEVELS
# Tiers whose membership is computed rather than read from model_reasoning.
_OVERLAY_REASONING_LEVELS: frozenset[str] = OVERLAY_REASONING_LEVELS
# The strongest tier a prompt-size heuristic may target on its own. Overlay
# tiers are opt-in by name only, never reached by the router drifting upward.
_MAX_INFERRED_LEVEL_INDEX: int = max(
    i for i, lvl in enumerate(_REASONING_LEVELS) if lvl not in OVERLAY_REASONING_LEVELS
)
# Capabilities that get their own capability-selecting virtual endpoints
# (llmproxy__tools, llmproxy__vision, and their /free variants).
_CAPABILITY_VIRTUALS: tuple[str, ...] = ("tools", "vision")
# Dimensions a single provider can be sliced into via per-provider virtual models
# (llmproxy__<provider>/<dimension>).  The bare "" form (llmproxy__<provider>)
# cycles through ALL of that provider's models and is handled separately.
_PER_PROVIDER_DIMENSIONS: tuple[str, ...] = (*_REASONING_LEVELS, *_CAPABILITY_VIRTUALS, "free")
# Per-candidate timeout for virtual-model cycling so a slow upstream doesn't block all failover.
_VIRTUAL_CANDIDATE_TIMEOUT: int = 60
# Extra attempts on the *same* candidate when it returns a transient failure
# (HTTP 429 / 5xx, timeout, connection error). To keep cost and latency low,
# these same-candidate retries are only spent on the LAST candidate — while
# alternatives remain, a transient failure fails over to the next candidate
# immediately (no backoff), since trying a different (often free/local) model
# beats waiting on a rate-limited one. See _candidate_max_attempts.
_VIRTUAL_MAX_RETRIES: int = 1
# Backoff (seconds) between those same-candidate retries.
_VIRTUAL_RETRY_BACKOFF: float = 0.5
# Auto-budget escalation: when a candidate answers 200 but the completion is
# empty *because* it was truncated on ``max_tokens`` (a reasoning model that
# spent the whole budget thinking), the same candidate is retried with a larger
# budget before failing over. The budget is multiplied by _BUDGET_BUMP_FACTOR
# each retry, capped at _BUDGET_BUMP_CEILING, for at most _BUDGET_BUMP_MAX_RETRIES
# rounds — bounded so a pathological model can't drive unbounded cost/latency.
# Factor 16 rather than a gentler ramp because the models that hit this are
# reasoning models: one starved at 8 tokens is starved at 32 and at 128 too, so
# the intermediate rungs are pure waste — three full round trips to learn what
# the first one already implied. Sixteen reaches the ceiling in at most three
# bumps from any starting budget (8 -> 128 -> 2048 -> 4096; from 256, one), so
# the fourth retry is margin for an unusually small budget rather than a step
# that normally fires: _bumped_budget returns None at the ceiling and the loop
# exits without spending a call.
#
# The ceiling stays 4096 deliberately. Nothing in llmproxy knows any model's
# real maximum output, so a higher ceiling risks asking for more than the model
# allows — and a provider answers that with a 400, which turns a recoverable
# empty completion into a hard failure. Raising it would need per-model output
# caps first.
_BUDGET_BUMP_FACTOR: int = 16
_BUDGET_BUMP_CEILING: int = 65535
_BUDGET_BUMP_MAX_RETRIES: int = 4
# Stable per-process timestamp used as the OpenAI-standard ``created`` fallback
# for models whose upstream listing omits it (and for synthetic virtual models).
_SERVER_EPOCH: int = int(time.time())
# Bytes of the streamed SSE response kept buffered so the final `usage` chunk can
# be parsed for token/cost accounting without buffering the whole stream.
_STREAM_TAIL_BYTES: int = 16384
# Virtual models use the "llmproxy__" prefix (same double-underscore as the
# provider display form) so strict clients accept them and they sort together.
# The legacy "llmproxy/" prefix is kept in the membership set so pinned client
# configs continue to resolve; only the new form is advertised in /v1/models.
_NEW_VIRTUAL_MODELS: frozenset[str] = frozenset({
    "llmproxy__free", "llmproxy__local", "llmproxy__loadbalanced",
    *(f"llmproxy__{lvl}" for lvl in _REASONING_LEVELS),
    *(f"llmproxy__{lvl}/free" for lvl in _REASONING_LEVELS),
    *(f"llmproxy__{lvl}/local" for lvl in _REASONING_LEVELS),
    *(f"llmproxy__{cap}" for cap in _CAPABILITY_VIRTUALS),
    *(f"llmproxy__{cap}/free" for cap in _CAPABILITY_VIRTUALS),
})
_LEGACY_VIRTUAL_MODELS: frozenset[str] = frozenset({
    "llmproxy/free", "llmproxy/local", "llmproxy/loadbalanced",
    *(f"llmproxy/{lvl}" for lvl in _REASONING_LEVELS),
    *(f"llmproxy/{lvl}/free" for lvl in _REASONING_LEVELS),
    *(f"llmproxy/{lvl}/local" for lvl in _REASONING_LEVELS),
    *(f"llmproxy/{cap}" for cap in _CAPABILITY_VIRTUALS),
    *(f"llmproxy/{cap}/free" for cap in _CAPABILITY_VIRTUALS),
})
_VIRTUAL_MODELS: frozenset[str] = _NEW_VIRTUAL_MODELS | _LEGACY_VIRTUAL_MODELS
# Fusion (multi-model deliberation) virtual models. These do NOT cycle/failover
# like the sets above; they fan out to a panel, judge, and synthesize, so they
# are dispatched on a separate path (_proxy_fusion) before the cycling logic.
# They are members of _VIRTUAL_MODELS (so the cache-bypass and model-listing
# machinery recognizes them) but intentionally NOT of _FREE_VIRTUAL_MODELS:
# fusion/free does its own free-pool selection.
_NEW_FUSION_MODELS: frozenset[str] = frozenset({"llmproxy__fusion", "llmproxy__fusion/free"})
_LEGACY_FUSION_MODELS: frozenset[str] = frozenset({"llmproxy/fusion", "llmproxy/fusion/free"})
_FUSION_VIRTUAL_MODELS: frozenset[str] = _NEW_FUSION_MODELS | _LEGACY_FUSION_MODELS
# Recognized by _is_virtual_model (cache bypass, listing) but dispatched separately.
_VIRTUAL_MODELS = _VIRTUAL_MODELS | _FUSION_VIRTUAL_MODELS
# Virtual models that use capacity-aware free-tier load balancing.
_FREE_VIRTUAL_MODELS: frozenset[str] = frozenset({
    "llmproxy__free", "llmproxy/free",
    *(f"llmproxy__{lvl}/free" for lvl in _REASONING_LEVELS),
    *(f"llmproxy/{lvl}/free" for lvl in _REASONING_LEVELS),
    *(f"llmproxy__{cap}/free" for cap in _CAPABILITY_VIRTUALS),
    *(f"llmproxy/{cap}/free" for cap in _CAPABILITY_VIRTUALS),
})
# Virtual models served strictly from the localhost-backed pool. Mirror of
# _FREE_VIRTUAL_MODELS: the global local aggregator plus the reasoning-level
# /local sub-virtuals. (There are no capability /local virtuals.) Per-provider
# <provider>/local forms are recognized separately via _split_per_provider_virtual.
_LOCAL_VIRTUAL_MODELS: frozenset[str] = frozenset({
    "llmproxy__local", "llmproxy/local",
    *(f"llmproxy__{lvl}/local" for lvl in _REASONING_LEVELS),
    *(f"llmproxy/{lvl}/local" for lvl in _REASONING_LEVELS),
})
# Virtual models backed by an overlay tier whose membership is computed from
# benchmark scores (today: flagship). These are the only pools with a measured
# per-model ranking, so they are ordered by it rather than rotated or
# load-spread — see _flagship_ordered_candidates. Generated from
# _OVERLAY_REASONING_LEVELS so a future overlay tier is picked up automatically.
# The per-provider <provider>/flagship form is recognized separately, via
# _split_per_provider_virtual in _is_flagship_virtual_model.
_FLAGSHIP_VIRTUAL_MODELS: frozenset[str] = frozenset({
    *(f"{pfx}{lvl}" for lvl in _OVERLAY_REASONING_LEVELS
      for pfx in ("llmproxy__", "llmproxy/")),
    *(f"{pfx}{lvl}/{dim}" for lvl in _OVERLAY_REASONING_LEVELS
      for dim in ("free", "local") for pfx in ("llmproxy__", "llmproxy/")),
})
# The cost-tiered "just pick something sensible and cheap" virtual. It owns its
# own ordering (free → local → paid waterfall, optimized per-prompt within each
# tier) and is the ONLY virtual that crosses tiers, so it is deliberately NOT in
# _FREE_VIRTUAL_MODELS / _LOCAL_VIRTUAL_MODELS (single-tier, request-fit-triaged).
_LOADBALANCED_MODELS: frozenset[str] = frozenset({
    "llmproxy__loadbalanced", "llmproxy/loadbalanced",
})

# Maps proxy display ID -> (provider_name, upstream_id).
# Always access under _model_route_cache_lock.
_model_route_cache: dict[str, tuple[str, str]] = {}
_model_route_cache_lock = threading.Lock()

# Per-model context windows discovered from provider /models listings, keyed by
# lowercased "provider/upstream_id". Kept beside the route cache rather than
# folded into its tuple: the route cache is dual-keyed and unpacked as a 2-tuple
# in roughly twenty places plus six test modules, while the ordering passes that
# need context have already lost the proxy_id and hold only
# (provider, cfg, upstream_id) — so the qualified key is the natural shape here.
# Rebuilt atomically with the route cache under the SAME lock, so routing and
# context can never disagree about which models exist.
_model_context_cache: dict[str, int] = {}

# Capabilities each provider publishes about its own models, keyed by
# lowercased "provider/upstream_id". Kept beside the context cache and rebuilt
# with it, from the same listing, so it costs no extra fetch and can never
# describe a model the route cache does not have. Deliberately in memory: it is
# derived from a listing refetched every models_cache_ttl, so persisting it
# would only let it go stale.
_model_capability_cache: dict[str, set[str]] = {}

# Cached full model list returned by GET /v1/models.
# Tuple is (model_list, timestamp).  Protected by _models_list_cache_lock.
_models_list_cache: tuple[list[dict], float] | None = None
_models_list_cache_lock = threading.Lock()
_DEFAULT_MODELS_CACHE_TTL = 60

# Guards the stale-while-revalidate background refresh of _models_list_cache so a
# burst of requests arriving after expiry spawns at most one refresh thread.
_models_refresh_lock = threading.Lock()
_models_refresh_active = False

# ---------------------------------------------------------------------------
# Per-model usage tracking (free-tier capacity-aware load balancing + accounting)
# ---------------------------------------------------------------------------
# In-memory only; resets on server restart.  Each gunicorn worker process
# maintains its own counters — usage tracking is per-worker, not cross-process.
# For cross-process accuracy, configure a single worker or use a shared store.
# The pure counter / cost primitives live in usage.py so the scraper probe can
# reuse them; this section wires them to the live config + believed_free set.

_usage_registry: dict[str, ModelUsage] = {}
_usage_registry_lock = threading.Lock()
_usage_since: str = datetime.datetime.now(datetime.UTC).isoformat()

# believed_free models that served a request reporting a non-zero cost. Surfaced
# via GET /v1/usage and persisted to config['cost_observed_free_tier'] so the
# updater stops re-adding them to believed_free.
_paid_free_flags: dict[str, dict] = {}
_paid_free_lock = threading.Lock()

# Serializes the best-effort config.json append in _persist_cost_observed so two
# concurrent first-observations don't race the read-modify-write.
_cost_observed_persist_lock = threading.Lock()
# At most one background sidecar-update+PR reaction runs at a time; a run reads
# the freshly-persisted config, so it covers every entry recorded before it
# started. Concurrent observations skip rather than pile up duplicate scrapes.
_cost_observed_reaction_lock = threading.Lock()
_cost_observed_reaction_inflight = False
COST_OBSERVED_KEY = "cost_observed_free_tier"


# ---------------------------------------------------------------------------
# Saturation registry — remember quota-exhausted candidates across requests
# ---------------------------------------------------------------------------
#
# The in-request cycling engine already rotates off a 429; this registry makes
# that rotation *sticky*: a candidate that returns a quota/rate-limit error is
# cooled until its documented reset (Retry-After when provided, else a default
# window), so subsequent requests on any virtual endpoint skip it instead of
# re-picking the same depleted model/account first. Per-worker & in-memory, like
# the usage counters. Keyed identically to the usage registry (per account when
# a provider has several); a provider-wide sentinel model opens a circuit for a
# whole provider/account when its shared allowance is depleted.
_saturation_registry: dict[str, float] = {}  # key -> monotonic expiry (seconds)
_saturation_lock = threading.Lock()
_DEFAULT_SATURATION_COOLDOWN_S = 60.0
_MAX_SATURATION_COOLDOWN_S = 3600.0
_PROVIDER_CIRCUIT_MODEL = "__provider__"  # sentinel model for a provider-wide circuit

# --- oversized-request memory ---
#
# A 413 says this request was too big for this endpoint. Cycling past it already
# works (413 is not transient, so the loop fails straight over), but nothing was
# remembered, so the same candidate kept its rank and was tried first again next
# request. Under a random rotation that cost one wasted call in N; under the
# deterministic flagship ranking it costs one on every single request.
#
# Cooling it like a 429 would overshoot: a 413 is a property of *this request's
# size*, not of the candidate's availability, so a blanket cooldown would also
# divert small requests the model would have accepted. Instead remember the
# smallest body each target has ever rejected, and skip it only for requests at
# least that large.
#
# Deliberately in memory and per process. A body limit is cheap to relearn — the
# first oversized request after a restart rediscovers it, at the cost of one
# failover — so persisting it would buy a schema and a staleness problem and
# nothing else. The dict is bounded by the number of routing targets.
_oversize_registry: dict[str, int] = {}  # provider/model -> smallest 413'd body, bytes
_oversize_lock = threading.Lock()

# provider/model -> capabilities the upstream has REFUSED to serve, learned from
# rejections rather than from any listing. The proactive path (a provider's own
# supported_parameters, the catalog, the learned layer) is only as good as what
# providers publish, and a gateway that advertises tool support it cannot route
# to is exactly the case no amount of metadata-reading catches. See
# _record_capability_gap.
_capability_gap_registry: dict[str, set[str]] = {}
_capability_gap_lock = threading.Lock()


def _oversize_key(provider_name: str, upstream_model: str) -> str:
    """Registry key for a routing target's body limit.

    Deliberately NOT account-scoped, which is the one place this differs from
    ``_usage_key``: a request-size limit belongs to the endpoint, not to the
    credential presented to it, so every account of a provider shares one
    watermark and one account's 413 informs the rest.
    """
    return _usage_key(provider_name, upstream_model, None)


def _payload_size_bytes(payload: dict) -> int:
    """Serialized size of the upstream body, in bytes.

    Bytes rather than ``_estimate_payload_tokens``, which counts only message
    text: a 413 is about what goes on the wire, and the tool definitions, base64
    images and attachments that text estimate ignores are exactly what push a
    request over a byte limit.

    Always measure the CLIENT's payload, never the per-candidate upstream body.
    The upstream body substitutes the model id and, on a non-OpenAI surface, is
    dialect-rendered — so its size varies by candidate. Recording one measure
    and comparing against another would make the watermark wrong by however much
    those differ. A consistent proxy for the wire size answers the only question
    asked of it, which is whether this request is at least as large as one that
    was already refused.

    Returns 0 when the payload cannot be serialized, which disables the check
    rather than guessing.
    """
    try:
        return len(json.dumps(payload, default=str).encode("utf-8"))
    except Exception:  # noqa: BLE001 — routing must never fail on a payload shape
        return 0


def _record_oversize(provider_name: str, upstream_model: str, size_bytes: int) -> None:
    """Remember the smallest request *this* target has rejected as too large.

    Keeps the minimum, so a later, smaller rejection tightens the watermark and
    a larger one never loosens it — the limit can only be bounded from above by
    what we have actually observed being refused.
    """
    if size_bytes <= 0:
        return
    key = _oversize_key(provider_name, upstream_model)
    with _oversize_lock:
        prev = _oversize_registry.get(key)
        if prev is None or size_bytes < prev:
            _oversize_registry[key] = size_bytes


def _is_oversize_for(provider_name: str, upstream_model: str, size_bytes: int) -> bool:
    """True when this request is at least as large as one this target refused."""
    if size_bytes <= 0:
        return False
    with _oversize_lock:
        limit = _oversize_registry.get(_oversize_key(provider_name, upstream_model))
    return limit is not None and size_bytes >= limit


# OpenRouter names the filter that eliminated every endpoint; other gateways
# only say it in prose. Both are matched, and narrowly: a false positive here
# sidelines a working model for a capability it actually has.
_CAPABILITY_REJECTION_STEPS: dict[str, str] = {
    "filter by tool compatibility": "tools",
}
_CAPABILITY_REJECTION_PHRASES: tuple[tuple[str, str], ...] = (
    ("no endpoints found that support tool use", "tools"),
    ("no endpoints found that support tool calling", "tools"),
    ("no endpoints found that support image input", "vision"),
    ("no endpoints found that support structured outputs", "json"),
)


def _detect_capability_rejection(status: int | None, body: bytes | None) -> str | None:
    """The capability an upstream rejection proves this target cannot serve.

    Returns a capability name, or None when the body is not capability-shaped.

    This is the reactive counterpart to the spec gate: a provider that lists a
    model it cannot actually route a tool call to produces a non-transient 404
    that no listing predicted. Keeping the matcher narrow is the whole safety
    story, so it fires only on a recognised routing-funnel step or one of a few
    exact phrases, never on the mere presence of the word "tool".
    """
    if not body:
        return None
    if status is not None and status not in (400, 404, 422):
        return None
    try:
        text = body.decode("utf-8", "replace").lower()
    except Exception as e:  # noqa: BLE001 — a diagnostic must never fail a request
        print(f"[server:_detect_capability_rejection] {e}")
        traceback.print_exc()
        return None
    for step, cap in _CAPABILITY_REJECTION_STEPS.items():
        if step in text:
            return cap
    for phrase, cap in _CAPABILITY_REJECTION_PHRASES:
        if phrase in text:
            return cap
    return None


def _record_capability_gap(provider_name: str, upstream_model: str, cap: str) -> None:
    """Remember that this exact target refused to serve *cap*.

    Per routing target, never per normalized model: the whole reason this exists
    is that one deployment of some weights cannot do what another can. Logged at
    warning because silently deciding a model is incapable would be invisible
    exactly when it is wrong.
    """
    if not cap:
        return
    key = _oversize_key(provider_name, upstream_model)
    with _capability_gap_lock:
        known = _capability_gap_registry.setdefault(key, set())
        if cap in known:
            return
        known.add(cap)
    logger.warning(
        "[capability] %s/%s rejected a request for lack of '%s'; it will not be "
        "selected for requests needing that capability",
        provider_name, upstream_model, cap,
    )


def _learned_capability_gaps(provider_name: str, upstream_model: str) -> set[str]:
    """Capabilities this target has been observed refusing."""
    with _capability_gap_lock:
        return set(_capability_gap_registry.get(_oversize_key(provider_name, upstream_model), ()))


def _note_capability_rejection(
    provider_name: str, upstream_model: str, status: int | None, body: bytes | None
) -> str | None:
    """Record a capability gap when *body* is a capability-shaped rejection."""
    cap = _detect_capability_rejection(status, body)
    if cap:
        _record_capability_gap(provider_name, upstream_model, cap)
    return cap


def _note_accepted_size(provider_name: str, upstream_model: str, size_bytes: int) -> None:
    """Drop a target's watermark when it accepts a request that large.

    A demoted candidate is never dropped, so it is still reached as a last
    resort — which means a watermark set by a one-off 413 (a gateway hiccup, a
    limit since raised) can be disproved by an actual success. Without this, one
    spurious rejection would sideline a model for large requests until the
    process restarted.

    Only a success at or above the watermark is evidence: a smaller request
    succeeding says nothing about the limit, so it leaves the watermark alone.
    """
    if size_bytes <= 0:
        return
    key = _oversize_key(provider_name, upstream_model)
    with _oversize_lock:
        limit = _oversize_registry.get(key)
        if limit is not None and size_bytes >= limit:
            del _oversize_registry[key]
            cleared = True
        else:
            cleared = False
    if cleared:
        logger.info(
            "  %s/%s accepted a %d-byte request after refusing one that size — "
            "forgetting its size limit",
            provider_name, upstream_model, size_bytes,
        )


def _reset_oversize() -> None:
    """Drop every watermark. For tests."""
    with _oversize_lock:
        _oversize_registry.clear()


# HTTP statuses that mean "out of quota / rate limited": 402 Payment Required
# (out of credits) and 429 Too Many Requests. Both mark the model unavailable
# until its reset; a plain 5xx is transient and retried without a cooldown.
_QUOTA_STATUSES = frozenset({402, 429})

# Machine-readable quota codes (specific enough to trust on their own) and
# generic phrases (only trusted inside an error-shaped body).
_QUOTA_CODES = ("resource_exhausted", "insufficient_quota", "rate_limit_exceeded")
_QUOTA_PHRASES = ("quota", "rate limit", "too many requests")


# Substrings that identify a read timeout arriving dressed as something else.
# urllib3 raises ReadTimeoutError inside iter_content and requests re-raises it
# as a ConnectionError, so the exception TYPE cannot be trusted to tell "the
# upstream went quiet" from "the socket broke" — only the message can.
_TIMEOUT_MARKERS: tuple[str, ...] = ("timed out", "timeout")

# Attribute stamped on a synthesized 504 so the cycling loops can tell a timeout
# apart from an upstream that genuinely returned 504. Carried on the Response
# object rather than in a header or the body: a header would leak to the client
# on the last candidate, and parsing our own error body back out would couple the
# loops to its wording.
_TIMEOUT_ATTR = "llmproxy_timed_out"


def _looks_like_timeout(exc: BaseException) -> bool:
    """True when *exc* is a timeout, however it is dressed.

    ``requests.exceptions.Timeout`` covers the honest cases. The one that matters
    here is the dishonest one: a mid-stream read timeout surfaces as a
    ``ConnectionError`` whose message is "HTTPSConnectionPool(...): Read timed
    out.", which is exactly the stall this is meant to catch.
    """
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _TIMEOUT_MARKERS)


def _mark_timeout(resp: Response) -> Response:
    """Stamp *resp* as a synthesized timeout and return it."""
    try:
        setattr(resp, _TIMEOUT_ATTR, True)
    except Exception:  # noqa: BLE001 — a missing mark costs a cooldown, not a reply
        pass
    return resp


def _is_timeout_response(resp: Response) -> bool:
    """True for a response this proxy synthesized because a candidate timed out."""
    return bool(getattr(resp, _TIMEOUT_ATTR, False))


def _is_quota_error(status: int | None, body_bytes: bytes | None = None) -> bool:
    """True when a response signals quota / rate-limit exhaustion.

    Fires on HTTP 402/429 and on error bodies (a 200-with-error or a 4xx) whose
    code or message matches a known quota marker — Gemini ``RESOURCE_EXHAUSTED``,
    OpenAI ``insufficient_quota`` / ``rate_limit_exceeded``, or a generic
    "quota" / "rate limit" / "too many requests" phrase inside an error body.
    Deliberately distinct from a plain transient 5xx (retryable, but no cooldown).
    """
    if status in _QUOTA_STATUSES:
        return True
    if not body_bytes:
        return False
    try:
        text = body_bytes.decode("utf-8", "ignore").lower()
    except Exception:  # noqa: BLE001
        return False
    if any(code in text for code in _QUOTA_CODES):
        return True
    if "error" in text and any(p in text for p in _QUOTA_PHRASES):
        return True
    return False


def _parse_retry_after(value) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) to seconds."""
    if not value:
        return None
    value = str(value).strip()
    if value.isdigit():
        return float(value)
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        return max(0.0, (dt - datetime.datetime.now(datetime.UTC)).total_seconds())
    except Exception:  # noqa: BLE001
        return None


def _saturation_cooldown_seconds(retry_after=None) -> float:
    """Resolve the cooldown window: Retry-After if given, else configured default."""
    parsed = _parse_retry_after(retry_after)
    if parsed is not None and parsed > 0:
        return min(parsed, _MAX_SATURATION_COOLDOWN_S)
    try:
        configured = load_config().get("server", {}).get(
            "saturation_cooldown_seconds", _DEFAULT_SATURATION_COOLDOWN_S
        )
        cooldown = float(configured)
    except (TypeError, ValueError):
        cooldown = _DEFAULT_SATURATION_COOLDOWN_S
    return max(0.0, min(cooldown, _MAX_SATURATION_COOLDOWN_S))


def _mark_saturated(key: str, retry_after=None) -> None:
    """Cool *key* (a usage-registry key) until its reset so callers rotate off it."""
    cooldown = _saturation_cooldown_seconds(retry_after)
    if cooldown <= 0:
        return
    with _saturation_lock:
        _saturation_registry[key] = time.monotonic() + cooldown


def _is_saturated(key: str) -> bool:
    """True while *key* is still cooling; lazily evicts expired entries."""
    now = time.monotonic()
    with _saturation_lock:
        expiry = _saturation_registry.get(key)
        if expiry is None:
            return False
        if expiry <= now:
            del _saturation_registry[key]
            return False
        return True


def _mark_provider_circuit(provider_name: str, account_id: str | None = None, retry_after=None) -> None:
    """Open a provider-wide (per-account) circuit so concurrent requests skip it."""
    _mark_saturated(_usage_key(provider_name, _PROVIDER_CIRCUIT_MODEL, account_id), retry_after)


def _is_candidate_saturated(provider_name: str, upstream_model: str, account_id: str | None = None) -> bool:
    """True when either this model/account or its provider-wide circuit is cooling."""
    return (
        _is_saturated(_usage_key(provider_name, upstream_model, account_id))
        or _is_saturated(_usage_key(provider_name, _PROVIDER_CIRCUIT_MODEL, account_id))
    )


def _usage_key(provider_name: str, upstream_model: str, account_id: str | None = None) -> str:
    """Build the registry key for a provider/model, optionally scoped to an account.

    With ``account_id=None`` this reproduces the historical ``provider/model``
    key byte-for-byte, so single-credential providers meter exactly as before.
    When a provider has multiple accounts, each meters its own free-tier quota
    under ``provider#<account_id>/model`` — the ``#`` segment never collides with
    the ``/`` provider-separator or an upstream id.
    """
    if account_id:
        return f"{provider_name}#{account_id}/{upstream_model}".lower()
    return f"{provider_name}/{upstream_model}".lower()


def _get_or_create_tracker(key: str) -> ModelUsage:
    with _usage_registry_lock:
        tracker = _usage_registry.get(key)
        if tracker is None:
            tracker = ModelUsage()
            _usage_registry[key] = tracker
    return tracker


def _flag_paid_free(key: str, cost: float, source: str) -> bool:
    """Record (once-warned) that a believed-free model reported a cost.

    Returns True only on the *first* observation of this model, so the caller can
    persist it to ``cost_observed_free_tier`` exactly once rather than on every
    request.
    """
    with _paid_free_lock:
        entry = _paid_free_flags.get(key)
        if entry is None:
            _paid_free_flags[key] = {
                "observed_cost": round(cost, 8),
                "cost_source": source,
                "samples": 1,
            }
            logger.warning(
                "[usage] believed_free model %s reported a cost (%.8f, source=%s); "
                "adding to %s so the updater stops re-adding it to believed_free.",
                key, cost, source, COST_OBSERVED_KEY,
            )
            return True
        entry["samples"] += 1
        entry["observed_cost"] = round(max(entry["observed_cost"], cost), 8)
        return False


def _persist_cost_observed(qualified_id: str) -> None:
    """Record *qualified_id* as cost-observed in the routing-metadata sidecar.

    Written to the learned layer, never to config.json. A model reporting a real
    cost while marked free is something llmproxy discovered at runtime, so it
    belongs with the rest of what the refresh learns — and a machine process
    editing the file the user hand-edits is exactly what the three-layer split
    exists to stop.

    Best-effort and idempotent: adds the id if absent (case-insensitive), drops
    it from the learned believed_free so the sidecar is self-consistent, and
    invalidates the memo so the next request routes on the new fact. Any failure
    is logged and swallowed — usage accounting must never break a request.
    """
    try:
        provider, _, model = qualified_id.partition("/")
        if not provider or not model:
            return
        with _routing_sidecar_txn() as state:
            by_provider = state.setdefault("by_provider", {})
            if not isinstance(by_provider, dict):
                return
            entry = by_provider.setdefault(provider, {})
            if not isinstance(entry, dict):
                return

            observed = [x for x in entry.get(COST_OBSERVED_KEY) or [] if isinstance(x, str)]
            if model.lower() in {x.lower() for x in observed}:
                return  # already recorded
            observed.append(model)
            entry[COST_OBSERVED_KEY] = sorted(set(observed), key=str.lower)
            # Drop it from the learned believed_free too, so the sidecar does not
            # assert both at once. Routing already avoids it via _is_cost_observed;
            # this keeps the file honest.
            believed = entry.get("believed_free")
            if isinstance(believed, list):
                entry["believed_free"] = [
                    m for m in believed
                    if not (isinstance(m, str) and m.lower() == model.lower())
                ]
            logger.info(
                "[usage] recorded %s as cost-observed in routing_metadata.json",
                qualified_id,
            )
    except Exception as exc:  # noqa: BLE001 — never let persistence break a request
        logger.warning("[usage] could not persist %s to %s: %s",
                       qualified_id, COST_OBSERVED_KEY, exc)
        return
    # Propagate to the bundled sidecar + config.example.json and open a providers
    # PR (best-effort, in the background) so the change isn't just local.
    _react_to_cost_observed_async()


def _react_to_cost_observed_async() -> None:
    """Run the updater + providers-PR in the background after a cost observation.

    Reuses the startup updater path: with the model now in
    ``cost_observed_free_tier``, the updater's denylist removes it from the
    bundled providers.json believed_free, regenerates config.example.json, and
    opens/refreshes the providers PR. Gated on the same opt-in flags as the
    startup flow, and limited to one in-flight run (which picks up every recorded
    entry, so concurrent observations don't spawn duplicate scrapes).
    """
    config = load_config()
    free_tier = config.get("free_tier", {}) if isinstance(config.get("free_tier"), dict) else {}
    pr_enabled = config.get("providers_pr", {}).get("enabled") is True
    if not (pr_enabled or free_tier.get("update_on_startup") is True):
        return  # operator hasn't opted into sidecar updates / PRs

    global _cost_observed_reaction_inflight
    with _cost_observed_reaction_lock:
        if _cost_observed_reaction_inflight:
            return
        _cost_observed_reaction_inflight = True

    def _run() -> None:
        global _cost_observed_reaction_inflight
        try:
            logger.info("[usage] propagating cost_observed change to sidecar / PR")
            _run_free_models_update(load_config(), None)
            with _models_list_cache_lock:
                global _models_list_cache
                _models_list_cache = None
        except Exception as exc:  # noqa: BLE001 — background best-effort
            logger.warning("[usage] cost_observed propagation failed: %s", exc)
        finally:
            with _cost_observed_reaction_lock:
                _cost_observed_reaction_inflight = False

    threading.Thread(target=_run, daemon=True, name="cost-observed-react").start()


def _record_usage(
    provider_name: str,
    upstream_model: str,
    *,
    usage: dict | None = None,
    config: dict | None = None,
    count_request: bool = True,
    account_id: str | None = None,
) -> None:
    """Record a served request and/or its token + cost usage.

    *count_request* increments the request windows used by the free-tier load
    balancer; the streaming path counts the request up front (no usage yet) and
    calls again post-stream with ``count_request=False`` to add the token totals
    parsed from the final SSE chunk.

    *account_id* scopes the request to one of a provider's credentials so each
    account meters its own free-tier quota; ``None`` (the default) keeps the
    historical per-model accounting untouched.
    """
    key = _usage_key(provider_name, upstream_model, account_id)
    tracker = _get_or_create_tracker(key)

    prompt = completion = total = 0
    cost = 0.0
    source: str | None = None
    if usage:
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0) or (prompt + completion)
        cost, source = compute_cost(provider_name, upstream_model, usage, load_pricing_map())

    tracker.record(
        requests=1 if count_request else 0,
        prompt=prompt, completion=completion, total=total,
        cost=cost, cost_source=source,
    )

    if usage and cost > 0:
        cfg = config if config is not None else load_config()
        if _is_model_free(provider_name, upstream_model, cfg):
            # Cost-observation is a property of the *model*, not the account, so
            # it is flagged/persisted at model granularity regardless of account.
            if _flag_paid_free(_usage_key(provider_name, upstream_model), cost, source or "unknown"):
                # First observation — persist the original-cased qualified id so
                # the updater never re-adds it to believed_free.
                _persist_cost_observed(f"{provider_name}/{upstream_model}")


def _record_stream_usage(
    provider_name: str,
    upstream_model: str,
    tail: bytes,
    config: dict | None,
    account_id: str | None = None,
) -> None:
    """Parse the tail of a streamed response and record its tokens/cost (no request count)."""
    usage = parse_stream_usage(tail)
    if usage:
        _record_usage(
            provider_name, upstream_model, usage=usage, config=config,
            count_request=False, account_id=account_id,
        )


def _get_usage_snapshot(key: str) -> tuple[int, int]:
    """Return (requests_last_60s, requests_today) for the given provider/model key."""
    with _usage_registry_lock:
        tracker = _usage_registry.get(key)
    return tracker.snapshot() if tracker else (0, 0)


def _get_token_snapshot(key: str) -> tuple[int, int]:
    """Return (tokens_last_60s, tokens_today) for the given provider/model key."""
    with _usage_registry_lock:
        tracker = _usage_registry.get(key)
    return tracker.token_snapshot() if tracker else (0, 0)


def _get_health_snapshot(key: str) -> tuple[float, float, int]:
    """Return (success_rate, avg_latency_ms, samples) for a provider/model key."""
    with _usage_registry_lock:
        tracker = _usage_registry.get(key)
    return tracker.health_snapshot() if tracker else (1.0, 0.0, 0)


def _record_outcome(
    provider_name: str,
    upstream_model: str,
    ok: bool,
    *,
    latency_ms: float | None = None,
    account_id: str | None = None,
) -> None:
    """Record whether one upstream attempt worked, for health-aware ordering."""
    tracker = _get_or_create_tracker(_usage_key(provider_name, upstream_model, account_id))
    tracker.record_outcome(ok, latency_ms)


# Exception types and message fragments that mean *the client went away* or that
# llmproxy's own stream plumbing closed, rather than that the upstream failed.
#
# Counting these against a provider is the classic circuit-breaker mistake: one
# user pressing Ctrl-C mid-stream would demote a healthy provider, and on a pool
# whose last candidate is the only one left it can dead-end the rotation
# entirely. The distinction matters here more than in most proxies because
# llmproxy streams by default.
_CLIENT_ABORT_MARKERS: tuple[str, ...] = (
    "client disconnected",
    "connection reset by peer",
    "broken pipe",
    "response closed",
    "controller is already closed",
    "request aborted",
    "aborted by the client",
    "generatorexit",
)


def _is_upstream_failure(exc: BaseException | None = None, status: int | None = None) -> bool:
    """True when a failed attempt should count against the provider's health.

    Returns False for client aborts and llmproxy's own stream-lifecycle errors —
    see ``_CLIENT_ABORT_MARKERS``. A 4xx that is not a quota error is the
    caller's fault (a malformed request will fail identically everywhere), so it
    is not counted either; 402/429 already have their own cooldown path and are
    counted so a chronically exhausted candidate also sinks in the ordering.
    """
    if exc is not None:
        if isinstance(exc, GeneratorExit):
            return False
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(marker in text for marker in _CLIENT_ABORT_MARKERS):
            return False
        return True
    if status is None:
        return False
    if status >= 500:
        return True
    if status in _QUOTA_STATUSES:
        return True
    return False


# Health scoring. Below this many observed attempts a candidate is treated as
# healthy: demoting on one or two samples punishes cold models and newly added
# providers for noise, which is exactly backwards.
_HEALTH_MIN_SAMPLES = 5
# Success rate under which a candidate is considered degraded. Sits well below
# 1.0 so ordinary transient failure does not demote, and above 0.5 so a coin-flip
# provider always does.
_HEALTH_DEGRADED_RATE = 0.75
# Floor on the multiplier, so even a fully broken candidate keeps a non-zero
# score and stays in the pool as a last resort rather than being excluded.
_HEALTH_MIN_MULTIPLIER = 0.05
# Latency at or below which a candidate is considered simply "normal speed".
# Below this the latency term must be exactly 1.0, otherwise a candidate with a
# proven-good record would score *below* an untried one — an inversion that
# would quietly punish every model the proxy has actually used.
_HEALTH_FAST_MS = 2000.0


def _health_score(provider_name: str, upstream_model: str, account_id: str | None = None) -> float:
    """Return a health multiplier in [_HEALTH_MIN_MULTIPLIER, 1.0].

    1.0 means "no reason to avoid this"; lower means recent attempts have been
    failing or slow. Applied as a *multiplier* on an existing ordering score
    rather than as a filter, so an unhealthy candidate sinks to the back of the
    rotation but is still tried when nothing better is left — llmproxy never
    drops a candidate, and a degraded provider answering is better than a 503.

    Latency is folded in logarithmically and only above ``_HEALTH_FAST_MS``, so
    that "slow" nudges the ordering without ever dominating "broken": the whole
    latency term spans a factor of two, while a failing provider can lose an
    order of magnitude.
    """
    rate, avg_latency, samples = _get_health_snapshot(
        _usage_key(provider_name, upstream_model, account_id)
    )
    if samples < _HEALTH_MIN_SAMPLES:
        return 1.0

    score = rate
    if rate < _HEALTH_DEGRADED_RATE:
        # Degraded band: penalise beyond the raw rate so a candidate that is
        # failing a quarter of the time ranks clearly below one that is not.
        score *= rate
    if avg_latency > _HEALTH_FAST_MS:
        # Neutral up to _HEALTH_FAST_MS, then decays by a quarter per decade and
        # bottoms out at 0.5: ~0.75 at 20s, 0.5 at 200s. Bounded so a wild
        # latency reading can nudge but never invert the ordering.
        decades = math.log10(avg_latency / _HEALTH_FAST_MS)
        score *= max(0.5, 1.0 - 0.25 * decades)
    return max(_HEALTH_MIN_MULTIPLIER, min(1.0, score))


def _reset_usage() -> None:
    """Clear all in-memory usage counters, paid-free flags, and saturation state."""
    global _usage_since
    with _usage_registry_lock:
        _usage_registry.clear()
    with _paid_free_lock:
        _paid_free_flags.clear()
    with _saturation_lock:
        _saturation_registry.clear()
    _usage_since = datetime.datetime.now(datetime.UTC).isoformat()


# ---------------------------------------------------------------------------
# Local provider startup sync
# ---------------------------------------------------------------------------
# Tracks whether the one-time local model sync has run since startup.
_local_sync_done: bool = False
_local_sync_lock = threading.Lock()

# Tracks whether the one-time startup run of update_free_models has fired.
_startup_update_done: bool = False
_startup_update_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Periodic interval checks (free-models sweep, flagship recompute, cost probe,
# PR creation)
# ---------------------------------------------------------------------------
# State files are re-read at most once per _PROBE_INTERVAL_GATE_SEC so
# concurrent requests don't all hit disk simultaneously. The actual cadences
# come from free_tier.update_frequency_days and
# flagship_tier.refresh_frequency_days, with free_tier.cost_probe.frequency_days
# throttling that one expensive source within a sweep.
_PROBE_INTERVAL_GATE_SEC = 60   # check state files at most once per minute
_last_probe_interval_check: float = 0.0
_probe_interval_check_lock = threading.Lock()

_free_update_inflight: bool = False
_free_update_lock = threading.Lock()
_flagship_refresh_inflight: bool = False
# Latched the first time a flagship pool is served without a ranking, so the
# explanation is logged once rather than on every request. The route reason
# already carries the fact, but nobody reads a header until something looks
# wrong — which is exactly how an inert ranking survived a deploy unnoticed.
_flagship_unranked_warned: bool = False
# Latched the first time the live flagship free pool is found below its floor
# with no recompute able to fix it. Saying it once names a tier that cannot
# meet the guarantee it advertises; saying it every interval tick would bury
# the log.
_flagship_floor_warned: bool = False
_flagship_refresh_lock = threading.Lock()
_cost_probe_inflight: bool = False
_cost_probe_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Short-lived response cache (non-streaming only)
# ---------------------------------------------------------------------------
# Keyed on SHA-256(endpoint + sorted JSON payload).  Only 2xx responses are
# stored.  Entries expire after server.response_cache_ttl seconds (default 120).

# The entry carries the selected model alongside the body: a cache hit never
# reaches an upstream, so it is the only way a replayed reply can still say
# which model produced it.
_response_cache: dict[str, tuple[bytes, int, str, str | None, float]] = {}
_response_cache_lock = threading.Lock()
_DEFAULT_RESPONSE_CACHE_TTL = 120


def _response_cache_key(endpoint: str, payload: dict, auth: str = "") -> str:
    """Stable hash of the request, scoped by caller identity and excluding 'stream'."""
    filtered = {k: v for k, v in payload.items() if k != "stream"}
    raw = json.dumps({"_endpoint": endpoint, "_auth": auth, **filtered}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _response_cache_prune(ttl: int) -> None:
    """Evict all expired entries. Must be called with _response_cache_lock held."""
    now = time.monotonic()
    expired = [k for k, (*_, ts) in _response_cache.items() if now - ts > ttl]
    for k in expired:
        del _response_cache[k]


def _response_cache_get(key: str, ttl: int) -> tuple[bytes, int, str, str | None] | None:
    """Return ``(content, status, content_type, selected_model)`` or None."""
    with _response_cache_lock:
        _response_cache_prune(ttl)
        entry = _response_cache.get(key)
    if entry is None:
        return None
    content, status, content_type, selected_model, _ = entry
    return content, status, content_type, selected_model


def _response_cache_put(
    key: str,
    content: bytes,
    status: int,
    content_type: str,
    ttl: int,
    selected_model: str | None = None,
) -> None:
    with _response_cache_lock:
        _response_cache_prune(ttl)
        _response_cache[key] = (content, status, content_type, selected_model, time.monotonic())


# ---------------------------------------------------------------------------
# Recent upstream failures
# ---------------------------------------------------------------------------
#
# Health scores say a candidate is unwell; they cannot say WHY, because
# ``record_outcome`` stores a bare boolean. When a pool is exhausted and the
# client gets a 502, the operator's next question is always which models failed
# and with what — and the answer should not require turning on the request log
# and grepping it, not least because the log is off by default.
#
# So failures are recorded structurally, in a bounded in-process ring: newest
# first, capped, and pruned by age. Per worker, exactly like /v1/usage, which is
# one more reason server.workers defaults to 1.

_FAILURE_LOG_MAX: int = 250
_FAILURE_LOG_TTL_S: float = 6 * 60 * 60
_FAILURE_DETAIL_MAX_CHARS: int = 300

_failure_log: deque = deque(maxlen=_FAILURE_LOG_MAX)
_failure_log_lock = threading.Lock()

# Credential shapes that must never reach the failure report. Request headers
# are never recorded at all (see the audit-record note below, and _log_request),
# but an upstream is free to quote a key back inside an error MESSAGE, and that
# message is the one field here that comes from outside.
_SECRET_PATTERNS: tuple[re.Pattern, ...] = (
    # A labelled credential, consuming the value that follows the label so
    # "Authorization: Bearer abc123" does not leave "abc123" behind.
    re.compile(
        r"(?i)\b(?:authorization|api[-_ ]?key|access[-_ ]?token|token|bearer)\b"
        r"\s*[:=]?\s*(?:bearer\s+)?[A-Za-z0-9._\-]{6,}"
    ),
    # Vendor-prefixed keys, which are recognisable on their own.
    re.compile(r"\b(?:sk|pk|rk|xoxb|ghp|gho|glpat)-[A-Za-z0-9_\-]{8,}"),
    # A long unbroken opaque run. Deliberately excludes hyphens and requires 40
    # characters, so hyphenated model ids and ordinary prose survive intact
    # while base64/hex key material does not.
    re.compile(r"\b[A-Za-z0-9_]{40,}\b"),
)


def _scrub_secrets(text: str) -> str:
    """Redact anything credential-shaped from a string bound for the report."""
    if not text:
        return ""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


# Cloudflare's own error codes, as they appear in its interstitial HTML. Only
# the ones an API caller can actually hit and act on are named; anything else
# falls back to the bare code.
_CDN_BLOCK_CODES = {
    "1010": "browser signature",
    "1015": "rate limited",
    "1020": "firewall rule",
    "1006": "IP banned",
    "1009": "country blocked",
}
# Two independent markers must BOTH appear. A single one is not enough: an
# upstream that serves a model called "cloudflare/llama-3" would otherwise have
# every one of its ordinary JSON errors relabelled as a CDN block, which is
# worse than the bare status this replaces.
_CDN_MARKERS = re.compile(
    r"cloudflare|cf-error-details|__cf_|attention required|cf-ray",
    re.IGNORECASE,
)
_CDN_ERROR_CODE_RE = re.compile(r"error\s+code[:\s]+(\d{4})", re.IGNORECASE)


def _cdn_block_detail(text: str) -> str | None:
    """A one-line explanation when *text* is a CDN block page, else None.

    An upstream behind a CDN answers a refused request with an HTML
    interstitial, not with an API error. Recorded raw, that lands in the failure
    ring as "Backend request failed with status 403" plus four kilobytes of
    markup, and the one fact that would explain it — that the CDN, not the API,
    said no — is the fact that gets lost. Naming it turns an afternoon of
    bisecting headers into a glance at /v1/failures.
    """
    if "<html" not in text.lower() and "<!doctype" not in text.lower():
        return None
    if not _CDN_MARKERS.search(text):
        return None
    match = _CDN_ERROR_CODE_RE.search(text)
    if match:
        code = match.group(1)
        meaning = _CDN_BLOCK_CODES.get(code)
        named = f"error {code} ({meaning})" if meaning else f"error {code}"
    else:
        named = "no error code in the page"
    return (f"CDN blocked the request before it reached the API: Cloudflare "
            f"{named}. The upstream never saw it.")


def _failure_detail(body: bytes | str | None) -> str:
    """A short, scrubbed, human-readable excerpt of an upstream error body.

    Prefers the ``error.message`` an OpenAI-compatible error carries, since that
    is the sentence an operator wants; falls back to the raw body. Always
    scrubbed and always truncated — this is a diagnostic, not a transcript.
    """
    if not body:
        return ""
    if isinstance(body, bytes):
        body = body.decode("utf-8", "replace")
    text = body.strip()
    # Checked BEFORE the JSON parse: a block page is not JSON, so the parse
    # would fall through and hand back a slice of raw markup.
    cdn = _cdn_block_detail(text)
    if cdn:
        return cdn
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                text = err["message"]
            elif isinstance(err, str):
                text = err
    except Exception:  # noqa: BLE001 — a non-JSON body is fine, use it as-is
        pass
    text = _scrub_secrets(" ".join(text.split()))
    if len(text) > _FAILURE_DETAIL_MAX_CHARS:
        text = text[:_FAILURE_DETAIL_MAX_CHARS].rstrip() + "…"
    return text


def _record_failure(
    provider_name: str,
    upstream_model: str,
    *,
    status: int | None = None,
    kind: str = "upstream",
    detail: bytes | str | None = None,
    virtual_model: str | None = None,
    duration_ms: float | None = None,
) -> None:
    """Append one structured failure record to the ring.

    *kind* classifies the failure for the report: ``timeout``, ``connection``,
    ``quota``, ``capability``, ``oversize``, ``stream``, ``server`` or
    ``upstream``. Never raises: a diagnostic that can fail a request is worse
    than no diagnostic.

    *duration_ms* is what separates "this pool answers fast and wrongly" from
    "this pool burns a full candidate timeout each time", which are different
    operational problems with the same status code.
    """
    try:
        detail_text = _failure_detail(detail)
        # A CDN refusal and an API error are different problems wearing the same
        # status code, so they get different kinds. Only an otherwise-unclassified
        # "upstream" failure is upgraded: a caller that already named the kind
        # (a timeout, a capability rejection) knows more than this does.
        if kind == "upstream" and detail_text.startswith("CDN blocked"):
            kind = "cdn_block"
        record = {
            "at": datetime.datetime.now(datetime.UTC).isoformat(),
            "ts": time.time(),
            "provider": provider_name,
            "model": upstream_model,
            "target": f"{provider_name}/{upstream_model}",
            "virtual_model": virtual_model,
            "status": status,
            "kind": kind,
            "detail": detail_text,
            "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
        }
        with _failure_log_lock:
            _failure_log.append(record)
    except Exception as e:  # noqa: BLE001 — never fail a request over a diagnostic
        print(f"[server:_record_failure] {e}")
        traceback.print_exc()


# Statuses that describe THE REQUEST rather than the upstream, and so remain
# true no matter which candidate served it. Passed through on exhaustion only
# when every candidate agreed, because one gateway rejecting a parameter another
# accepts is a routing fact, not a verdict on the request.
_CLIENT_FAULT_STATUSES: frozenset[int] = frozenset({400, 413, 422})


def _exhausted_pool_status(attempted: list[tuple[str, str, int | None]]) -> int:
    """The status llmproxy should return once every candidate has failed.

    llmproxy is a gateway, so the status it returns describes ITS boundary. The
    client asked for a virtual id that exists; replaying the last candidate's
    status attributes the upstream's problem to the caller. A relayed 404 is the
    damaging case: to an OpenAI-compatible client it means "no such model",
    which is terminal, so a client with a perfectly good retry budget abandons
    the request instead of retrying a transient pool outage.

    The rule is deliberately narrow, so only the misleading case changes:

    * a UNANIMOUS status that is already honest is relayed untouched — a
      client-fault 400/413/422 (if every candidate rejects the request
      identically, the request really is the problem), a 429 (accurate, and it
      clears by itself), or any 5xx (already says "server side", and relaying it
      preserves the upstream diagnostic);
    * anything else -> 502. That is every 4xx which attributes the failure to
      the caller when it belongs to the upstream — 404, 401, 403 and whatever
      a gateway invents next — plus any MIXTURE, where no single upstream
      status can speak for the pool.

    503 is deliberately not produced here; it keeps its existing meaning of
    "there was nothing to try", so the status line alone distinguishes an empty
    pool from an exhausted one.
    """
    statuses = {st for _p, _m, st in attempted if st is not None}
    if len(statuses) == 1:
        only = next(iter(statuses))
        if only in _CLIENT_FAULT_STATUSES or only == 429 or only >= 500:
            return only
    return 502


def _exhausted_pool_body(
    label: str, attempted: list[tuple[str, str, int | None]], detail: str
) -> dict:
    """The error body for an exhausted pool, naming every candidate tried.

    The per-candidate roll-call is the point: "all candidates failed" with no
    list is precisely the message that sends an operator to the logs. Details
    are scrubbed on the way in (see ``_failure_detail``).
    """
    return {
        "error": {
            "message": (
                f"All {len(attempted)} '{label}' candidate(s) failed. "
                f"Last upstream error: {detail}" if detail
                else f"All {len(attempted)} '{label}' candidate(s) failed."
            ),
            "type": "upstream_error",
            "code": "all_candidates_failed",
            "llmproxy_candidates": [
                {"target": f"{pn}/{um}", "status": st} for pn, um, st in attempted
            ],
        }
    }


def _classify_failure(
    status: int | None,
    body: bytes | str | None,
    *,
    capability: str | None = None,
    timed_out: bool = False,
) -> str:
    """Bucket one failure for the report, most specific cause first.

    The ordering matters: a capability rejection and an oversize rejection are
    both 4xx, and both would otherwise read as a generic "upstream" failure,
    which is exactly the ambiguity this report exists to remove.
    """
    if timed_out:
        return "timeout"
    if capability:
        return "capability"
    if status == 413:
        return "oversize"
    if _is_quota_error(status, body):
        return "quota"
    if status is not None and status >= 500:
        return "server"
    return "upstream"


def _note_candidate_failure(
    attempted: list[tuple[str, str, int | None]],
    provider_name: str,
    upstream_model: str,
    *,
    status: int | None = None,
    kind: str = "upstream",
    detail: bytes | str | None = None,
    virtual_model: str | None = None,
    duration_ms: float | None = None,
) -> str:
    """Record one candidate's failure, for both the report and the final status.

    Returns the scrubbed detail it recorded, so a caller that must describe the
    failure later (the exhausted-pool reply) reuses exactly what the report
    shows rather than re-deriving it from a raw body.

    Recording a failover takes two coupled steps — the cycling loop's roll-call
    of what it tried, and the ring the failure report reads — and keeping them
    as separate statements is how six of the streaming loop's eight failover
    paths ended up recording neither. They are one call now so a new path cannot
    silently record half of it.
    """
    attempted.append((provider_name, upstream_model, status))
    _record_failure(
        provider_name, upstream_model,
        status=status, kind=kind, detail=detail,
        virtual_model=virtual_model, duration_ms=duration_ms,
    )
    return _failure_detail(detail)


def _exception_failure_kind(exc: BaseException) -> str:
    """Bucket a connect- or stream-level exception for the report.

    A timeout and a refused connection look alike in a log line and mean quite
    different things: one is an upstream that accepted the socket and then went
    quiet, the other is one that was never there.
    """
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "connection"
    return "upstream"


def _failure_records(since_ts: float | None = None) -> list[dict]:
    """Recent failures, newest first, pruned of anything past the TTL."""
    cutoff = time.time() - _FAILURE_LOG_TTL_S
    with _failure_log_lock:
        rows = [r for r in _failure_log if r.get("ts", 0) >= cutoff]
        if len(rows) != len(_failure_log):
            _failure_log.clear()
            _failure_log.extend(rows)
    if since_ts is not None:
        rows = [r for r in rows if r.get("ts", 0) >= since_ts]
    return list(reversed(rows))


_SINCE_UNITS: dict[str, int] = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_since(raw: str | None) -> float | None:
    """A ``since`` query parameter as an absolute unix timestamp, or None.

    Accepts a bare unix timestamp or a relative age (``30s``, ``15m``, ``2h``,
    ``1d``). Anything unparseable means "no filter" rather than an error: this
    is a diagnostic endpoint and a typo should not turn into a 400.
    """
    if not raw:
        return None
    raw = raw.strip().lower()
    try:
        unit = _SINCE_UNITS.get(raw[-1:])
        if unit is not None and raw[:-1]:
            return time.time() - float(raw[:-1]) * unit
        value = float(raw)
    except (TypeError, ValueError):
        return None
    # A small number is an age in seconds; a large one is an absolute epoch.
    return value if value > 10_000_000 else time.time() - value


def _reset_failures() -> None:
    """Clear the failure ring."""
    with _failure_log_lock:
        _failure_log.clear()


# ---------------------------------------------------------------------------
# Per-request audit records
# ---------------------------------------------------------------------------
#
# The human log says what llmproxy is doing; this says what it did. The two ``→``
# / ``←`` lines cannot be paired under concurrency (no id), carry no model,
# tokens or route reason, and — because Flask runs ``after_request`` before the
# WSGI server iterates a streamed body — report time-to-headers rather than the
# real duration of the request that matters most. One structured record per
# request, emitted when the response has actually finished, fixes all three.
#
# Request headers are never recorded. They carry the client's Authorization, and
# a record stream that must be handled as credential material is one nobody will
# keep. The bodies are recorded in ``full`` mode, so prompts and completions do
# land in the stream; that is the mode's whole purpose and is why it is off by
# default and documented as a data-handling decision rather than a log level.


def _body_capture_allowed(path: str) -> bool:
    """False for paths whose bodies carry credentials rather than content.

    The admin API takes API keys in the clear — that is how you set one — so in
    ``full`` mode its request bodies would put plaintext provider keys into the
    record stream. ``admin.py`` masks keys on the way *out* for exactly this
    reason; this is the same rule applied on the way in. Admin requests are
    still recorded, just without their bodies, so the audit trail keeps the fact
    that a config change happened without becoming credential material.
    """
    return not path.startswith("/admin")


def _json_or_text(raw: bytes | None, limit: int) -> object:
    """Decode a body for the record: parsed JSON where possible, else text.

    Parsing rather than escaping keeps a record one object instead of an object
    wrapping a long JSON string, which is what makes ``jq`` useful over the
    stream. A body that is not JSON (an SSE stream, a provider's HTML error
    page) is kept as text so nothing is silently dropped.
    """
    if not raw:
        return None
    truncated = bool(limit) and len(raw) > limit
    body = raw[:limit] if truncated else raw
    text = body.decode("utf-8", "replace")
    if not truncated:
        try:
            return json.loads(text)
        except Exception:  # noqa: BLE001 — not JSON is normal, not an error
            pass
    return {"truncated": True, "bytes": len(raw), "text": text} if truncated else text


def _emit_request_record(
    status: int,
    elapsed_ms: float,
    *,
    mode: str,
    method: str,
    path: str,
    request_id: str,
    requested_model: object = None,
    request_body: bytes | None = None,
    response_body: bytes | None = None,
    streamed: bool = False,
    selected_model: str | None = None,
    route_reason: str | None = None,
) -> None:
    """Write one JSON record for a completed request. Never raises."""
    try:
        record: dict = {
            "object": "request.record",
            "id": request_id,
            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": round(elapsed_ms, 1),
            "streamed": streamed,
            "model": requested_model,
            "selected_model": selected_model,
            "route_reason": route_reason,
        }
        # failed_over is the question anyone reading these actually asks, and it
        # is already encoded in the reason; surfacing it saves parsing.
        if route_reason:
            record["failed_over"] = ROUTE_SOURCE_FAILOVER in route_reason
        if mode == "full":
            if _body_capture_allowed(path):
                limit = _config_int("request_log_max_body_bytes",
                                    _DEFAULT_REQUEST_LOG_MAX_BODY)
                record["request_body"] = _json_or_text(request_body, limit)
                record["response_body"] = _json_or_text(response_body, limit)
            else:
                record["bodies_omitted"] = "credential-bearing path"
        # A record carrying bodies is DEBUG-tier by nature: it puts prompts and
        # completions wherever stdout goes, for as long as those logs are kept.
        # Emitting it at DEBUG means "full" alone is not enough — server.log_level
        # has to be DEBUG too — so content needs two deliberate switches rather
        # than one, while a metadata-only audit trail still runs at INFO.
        if mode == "full":
            request_logger.debug(json.dumps(record, default=str))
        else:
            request_logger.info(json.dumps(record, default=str))
    except Exception as e:  # noqa: BLE001 — a record must never break a reply
        print(f"[server:_emit_request_record] {e}")
        traceback.print_exc()


def _record_wrapped_stream(
    iterable, on_done: Callable[[bytes], None]
) -> Iterator[bytes]:
    """Relay *iterable* unchanged, then hand the joined bytes to *on_done*.

    This is what moves the record past the end of a streamed reply. Flask's
    ``after_request`` fires before the WSGI server pulls a single chunk, so a
    record written there would report the wrong duration and an empty body. The
    ``finally`` covers the client hanging up mid-stream too, which is a real
    outcome worth recording rather than a reason to lose the record.
    """
    chunks: list[bytes] = []
    try:
        for chunk in iterable:
            if chunk:
                chunks.append(chunk if isinstance(chunk, bytes) else bytes(chunk))
            yield chunk
    finally:
        try:
            on_done(b"".join(chunks))
        except Exception as e:  # noqa: BLE001
            print(f"[server:_record_wrapped_stream] {e}")
            traceback.print_exc()


@app.before_request
def _log_request() -> None:
    g._start_time = time.monotonic()
    # A correlation id, so the two human log lines and the audit record for one
    # request can be tied together under concurrency — which the ``→``/``←``
    # pair alone never could. Echoed back as X-LLMProxy-Request-Id so a client
    # reporting a problem can name the exact request.
    g.llmproxy_request_id = uuid.uuid4().hex
    g.llmproxy_log_mode = _request_log_mode()
    if g.llmproxy_log_mode != "off":
        # Capture the requested model HERE, before the view canonicalises
        # payload["model"] in place (llmproxy/flagship -> llmproxy__flagship, a
        # slash form -> the internal one). The record's job is to say what was
        # asked for; what it resolved to is selected_model. Read afterwards, the
        # record would report the rewritten id instead.
        try:
            parsed = request.get_json(silent=True)
            g.llmproxy_requested_model = (
                parsed.get("model") if isinstance(parsed, dict) else None
            )
        except Exception:  # noqa: BLE001 — auditing never fails a request
            g.llmproxy_requested_model = None
    if g.llmproxy_log_mode == "full":
        # Read the body here too, while it is certainly still readable. Flask
        # caches it, so the view's own get_json() is unaffected, and a view that
        # fails before parsing still leaves the record with what was actually
        # sent.
        try:
            g.llmproxy_request_body = request.get_data(cache=True)
        except Exception:  # noqa: BLE001
            g.llmproxy_request_body = None
    # Fire the one-time startup tasks (warm the virtual-model route cache and,
    # if enabled, run the free-models updater). This is a fallback safety net for
    # deployments where the eager per-worker trigger in __main__ did not fire; it
    # is a no-op after the first invocation.
    _run_startup_tasks_once()
    # Check probe / PR frequency intervals on every request (debounced by
    # _PROBE_INTERVAL_GATE_SEC so state files are read at most once per minute).
    _maybe_fire_interval_probes()
    logger.info("→ %s %s", request.method, request.path)


@app.after_request
def _log_response(response: Response) -> Response:
    elapsed_ms = (time.monotonic() - g._start_time) * 1000
    logger.info("← %s %s  %d  %.0fms", request.method, request.path, response.status_code, elapsed_ms)
    # Last stop before the bytes leave the app: apply the route provenance
    # recorded at selection time. See _stamp_route_provenance for why the
    # guarantee has to be enforced here rather than at each return site.
    response = _stamp_route_provenance(response)
    return _attach_request_record(response, elapsed_ms)


def _attach_request_record(response: Response, elapsed_ms: float) -> Response:
    """Emit this request's audit record, or arrange for it to be emitted.

    A buffered reply is complete right now, so its record goes out immediately.
    A streamed one is not: ``after_request`` runs before the WSGI server pulls a
    single chunk, so the record is deferred onto the end of the stream, where
    the duration is the real one and the body exists. Everything the record
    needs is captured into locals first, because the request context is gone by
    the time that generator finishes.
    """
    mode = g.get("llmproxy_log_mode", "off")
    request_id = g.get("llmproxy_request_id") or ""
    if request_id:
        response.headers["X-LLMProxy-Request-Id"] = request_id
    if mode == "off":
        return response
    try:
        common = {
            "mode": mode,
            "method": request.method,
            "path": request.path,
            "request_id": request_id,
            "requested_model": _requested_model_for_record(),
            "request_body": g.get("llmproxy_request_body"),
            "selected_model": g.get("llmproxy_selected_model"),
            "route_reason": g.get("llmproxy_route_reason"),
        }
        status = response.status_code
        started = g._start_time
        if response.is_streamed:
            def _done(body: bytes, _s=status, _c=common, _t=started) -> None:
                _emit_request_record(
                    _s, (time.monotonic() - _t) * 1000,
                    response_body=body, streamed=True, **_c,
                )
            response.response = _record_wrapped_stream(response.response, _done)
            return response
        _emit_request_record(
            status, elapsed_ms,
            response_body=response.get_data() if mode == "full" else None,
            streamed=False, **common,
        )
    except Exception as e:  # noqa: BLE001 — a record must never break a reply
        print(f"[server:_attach_request_record] {e}")
        traceback.print_exc()
    return response


def _requested_model_for_record() -> object:
    """The model id the client asked for, as the client wrote it.

    Captured in ``before_request`` rather than read here, because by now the
    proxy has canonicalised ``payload["model"]`` in place. Returns None for a
    request that names no model, which is most of the non-proxy surface.
    """
    return g.get("llmproxy_requested_model")


# ---------------------------------------------------------------------------
# Utility: build upstream headers
# ---------------------------------------------------------------------------

_FORWARDED_REQUEST_HEADERS = {
    "Content-Type",
    "HTTP-Referer",
    "X-Title",
    "User-Agent",
    "X-Request-ID",
    # Anthropic gates several features behind a beta opt-in header, prompt
    # caching among them historically. Dropping it meant a client could place
    # perfectly good cache_control breakpoints and still be billed uncached,
    # with nothing in the response to say why. The header only ever names
    # features, never credentials, so relaying it leaks nothing.
    "anthropic-beta",
}


def _upstream_headers(provider_cfg: dict) -> dict:
    """
    Build the header dict to send to an upstream provider.

    Always injects the provider's API key as the Bearer token.  Selected
    client-supplied headers are forwarded where the upstream is likely to
    consume them (e.g., HTTP-Referer for OpenRouter rate-limit attribution),
    including the ``User-Agent`` that ``_forwarded_client_headers`` resolves.

    NOTE: nothing in the live request path calls this any more — every one of
    them goes through a dialect adapter's ``build_request`` instead, and only
    tests reference this. It is left in place, and wired to the same resolver,
    so it cannot quietly become the one header builder that still leaks a bare
    library default if something starts calling it again.
    """
    headers = {"Content-Type": "application/json"}
    api_key = provider_api_key(provider_cfg)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(_forwarded_client_headers())
    return headers


# Bare library and runtime defaults. These identify an HTTP STACK rather than a
# client, and they are exactly what CDN bot filters match on: a relayed
# "Python-urllib/3.11" is refused by Cloudflare's Browser Integrity Check with a
# 403 and an HTML block page, measured against a real provider.
#
# curl and wget are deliberately absent. Both pass that check, and rewriting
# them would mislead anyone reproducing a problem by hand -- which is most
# people, most of the time.
#
# Anything naming a product ("OpenAI/Python 2.24.0") passes through untouched:
# it is a real client identity, upstreams use it for attribution, and replacing
# it would throw away information the operator may be relying on.
_GENERIC_CLIENT_UA_RE = re.compile(
    r"^(?:python-urllib|urllib|python-requests|requests|python-httpx|httpx"
    r"|aiohttp|go-http-client|java|okhttp|libwww-perl|ruby|php|node-fetch"
    r"|axios|apache-httpclient|guzzlehttp)[/ ]",
    re.IGNORECASE,
)


def _configured_user_agent(config: dict | None = None) -> str:
    """The string llmproxy calls itself, honouring ``server.user_agent``."""
    try:
        cfg = config if config is not None else load_config()
        raw = cfg.get("server", {}).get("user_agent")
    except Exception:  # noqa: BLE001 -- identifying ourselves must never raise
        raw = None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return USER_AGENT


def _outbound_user_agent(client_ua: str | None, config: dict | None = None) -> str | None:
    """The ``User-Agent`` to send upstream for a request from *client_ua*.

    ``server.forward_user_agent`` picks the policy:

    * ``"auto"`` (default) -- replace a MISSING or generic-library UA with our
      own, pass anything else through. This keeps attribution for clients that
      identify themselves properly while making sure a caller's choice of HTTP
      library cannot decide whether an upstream answers.
    * ``true`` -- relay whatever arrived and nothing else, which is exactly the
      behaviour that shipped before this setting existed. Returns None when the
      client sent none, leaving the HTTP library's default in place.
    * ``false`` -- always send our own, never relay.

    Returns None only in the ``true`` case with no inbound UA; every other path
    returns a string, so llmproxy is identifiable by default.
    """
    try:
        cfg = config if config is not None else load_config()
        mode = cfg.get("server", {}).get("forward_user_agent", "auto")
    except Exception:  # noqa: BLE001
        cfg, mode = None, "auto"

    ours = _configured_user_agent(cfg)
    # `true`/`false` are the natural things to write for a setting spelled
    # "forward_user_agent", so both the booleans and the string modes are
    # accepted. An unrecognised value reads as "auto" rather than as an error:
    # a typo should not silently restore the behaviour this exists to fix.
    if mode is True or str(mode).strip().lower() == "true":
        return client_ua or None
    if mode is False or str(mode).strip().lower() == "false":
        return ours
    if not client_ua or not client_ua.strip():
        return ours
    if _GENERIC_CLIENT_UA_RE.match(client_ua.strip()):
        return ours
    return client_ua


def _forwarded_client_headers() -> dict:
    """Selected client headers we relay upstream (OpenRouter attribution etc.).

    Returns only the present subset of _FORWARDED_REQUEST_HEADERS. Outbound
    dialect adapters decide whether to merge these (the OpenAI adapter does;
    native Anthropic/Gemini ignore them).

    The ``User-Agent`` is resolved through ``_outbound_user_agent`` rather than
    relayed blindly: see the note on ``_GENERIC_CLIENT_UA_RE``. Doing it here,
    in the single producer of this dict, is what gets every outbound request
    path -- buffered, streaming, cycling and fusion -- without four separate
    edits that could drift apart.

    Off the request thread (a background or fusion worker) there is no client to
    relay, so the result carries our own identity alone rather than being empty.
    That is the case that previously sent ``python-requests/x``. Such callers
    should still capture the full set on the request thread
    and pass them down (see _proxy_fusion's panel fan-out, which forwards them via
    ``_proxy_request(..., forwarded_headers=...)``).
    """
    if not has_request_context():
        return {"User-Agent": _configured_user_agent()}
    out: dict = {}
    for header in _FORWARDED_REQUEST_HEADERS - {"Content-Type", "User-Agent"}:
        value = request.headers.get(header)
        if value:
            out[header] = value
    resolved = _outbound_user_agent(request.headers.get("User-Agent"))
    if resolved:
        out["User-Agent"] = resolved
    return out


# ---------------------------------------------------------------------------
# Utility: error response helpers
# ---------------------------------------------------------------------------

def _error(message: str, status: int = 400, code: str = "invalid_request_error") -> Response:
    """Return an OpenAI-schema-compatible JSON error response."""
    return make_response(jsonify({
        "error": {
            "message": message,
            "type": code,
            "code": None,
        }
    }), status)


def _error_body_response(body: dict, status: int) -> Response:
    """Return a pre-built error object as JSON, for errors richer than _error().

    ``_error`` builds the body from a single message; an exhausted pool needs to
    carry a per-candidate roll-call alongside it, which is the whole reason that
    response is worth reading.
    """
    # Built without jsonify: the cycling loops can reach this outside a Flask
    # application context, and a diagnostic must not fail for want of one.
    return Response(json.dumps(body), status=status, content_type="application/json")


def _upstream_error(provider_name: str, e: Exception, status: int = 502) -> Response:
    logger.error("[server:upstream_error] provider=%s: %s", provider_name, e)
    traceback.print_exc()
    return _error(
        f"Upstream provider '{provider_name}' returned an error: {e}",
        status=status,
        code="upstream_error",
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health() -> Response:
    """Simple health check endpoint."""
    config = load_config()
    providers = list(config.get("providers", {}).keys())
    return jsonify({"status": "ok", "providers": providers})


# ---------------------------------------------------------------------------
# /version
# ---------------------------------------------------------------------------

@app.route("/version", methods=["GET"])
def version() -> Response:
    """Report the running llmproxy version.

    Clients and uptime probes commonly poll /version. Without this explicit
    route Flask returns 404, since the /v1/<path> pass-through only covers
    /v1/* paths.
    """
    return jsonify({"name": "llmproxy", "version": __version__})


# ---------------------------------------------------------------------------
# /v1/models  (GET)
# ---------------------------------------------------------------------------

def _flatten_display_model(stripped: str) -> str:
    """Collapse all but the LAST '/' in an upstream model id into '_'.

    This keeps the proxy display id (``provider__model``) to at most one slash so
    that "__" is the unambiguous provider separator and "/" appears at most once.
    A 0- or 1-slash id is returned unchanged.

    Examples
    --------
    >>> _flatten_display_model("gpt-4o")
    'gpt-4o'
    >>> _flatten_display_model("anthropic/claude-3.5-sonnet")
    'anthropic/claude-3.5-sonnet'
    >>> _flatten_display_model("meta-llama/llama-3/instruct")
    'meta-llama_llama-3/instruct'
    """
    last = stripped.rfind("/")
    if last == -1:
        return stripped
    return stripped[:last].replace("/", "_") + "/" + stripped[last + 1:]


def _virtual_display_name(canonical_vid: str) -> str:
    """Return a human-readable ``name`` for a virtual model id.

    The name is suitable for display in client model pickers: it contains no
    ``/`` so clients that derive a label by stripping to the last ``/`` (e.g.
    opencode's lmstudio plugin) show it verbatim rather than just a trailing
    segment like a bare ``free``.

    Examples
    --------
    >>> _virtual_display_name("llmproxy__free")
    '[llmproxy] Free'
    >>> _virtual_display_name("llmproxy__exploratory/free")
    '[llmproxy] Exploratory — Free'
    >>> _virtual_display_name("llmproxy__openrouter/free")
    '[llmproxy] Openrouter — Free'
    """
    _, sep, rest = canonical_vid.partition("__")
    if not sep:
        return canonical_vid
    parts = [p.replace("_", " ").title() for p in rest.replace("/", "__").split("__")]
    return "[llmproxy] " + " — ".join(parts)


def _display_id(canonical_id: str) -> str:
    """Convert a canonical ``provider__model`` id to the advertised ``provider/model`` form.

    The first ``__`` (the provider separator) becomes ``/`` and every remaining ``/``
    inside the model portion becomes ``__``, so the advertised id carries exactly one
    ``/`` — right after the provider. Clients that derive a display *name* by stripping
    to the last ``/`` (e.g. opencode's lmstudio plugin) then show the full model portion
    instead of just a trailing path segment like a bare ``free``.

    Examples
    --------
    >>> _display_id("openrouter__deepseek/deepseek-chat-v3")
    'openrouter/deepseek__deepseek-chat-v3'
    >>> _display_id("llmproxy__exploratory/free")
    'llmproxy/exploratory__free'
    >>> _display_id("llmproxy__free")
    'llmproxy/free'

    The inverse is handled inbound by ``_canonicalize_model_id``. Ids with no ``__``
    (already-foreign ``provider/model`` ids) are returned unchanged.

    Used both inbound (to dual-key the route cache) and outbound (to advertise
    virtual model ids in ``/v1/models`` so clients like opencode show a distinct
    label per virtual, e.g. ``llmproxy/deep__free``, ``llmproxy/loadbalanced``).
    """
    provider, sep, model = canonical_id.partition("__")
    if not sep:
        return canonical_id
    return provider + "/" + model.replace("/", "__")


def _architecture_block(
    input_mods: "list | None", output_mods: "list | None",
) -> dict:
    """Build an OpenRouter-style ``architecture`` block from modality lists.

    Falls back to text-only when a side is missing/empty. ``modality`` is the
    compact OpenRouter string form, e.g. ``"text+image->text"``.
    """
    inp = [m for m in (input_mods or []) if isinstance(m, str)] or ["text"]
    out = [m for m in (output_mods or []) if isinstance(m, str)] or ["text"]
    return {
        "input_modalities": inp,
        "output_modalities": out,
        "modality": "+".join(inp) + "->" + "+".join(out),
    }


def _supported_parameters(
    provider_name: str,
    upstream_id: str,
    config: dict,
    cap_map: "dict[str, set[str]] | None" = None,
    reasoning: "dict[str, str] | None" = None,
) -> list[str]:
    """OpenRouter-style ``supported_parameters`` derived from llmproxy config.

    Surfaces the tool/reasoning capabilities llmproxy already tracks (and uses
    for capability/reasoning virtual models) so clients can classify a model
    without a separate probe. ``cap_map``/``reasoning`` may be passed in to avoid
    recomputing them per model when annotating a whole list.
    """
    if cap_map is None:
        cap_map = _model_capabilities(config)
    if reasoning is None:
        reasoning = _get_model_reasoning(config)
    params: list[str] = []
    if _model_has_capability(provider_name, upstream_id, "tools", cap_map):
        params += ["tools", "tool_choice"]
    if _lookup_model_fact(reasoning, provider_name, upstream_id):
        params.append("reasoning")
    return params


def _describe_fetch_failure(url: str, resp: "requests.Response | None") -> str:
    """
    Build a secret-free diagnostic suffix for a failed /models fetch.

    Includes the request URL and, when a response was received, the HTTP
    status, the upstream Content-Type, and a short snippet of the response
    body. Request headers (which carry the Authorization bearer token) are
    never included, and the body snippet is truncated so we don't dump large
    upstream payloads into the logs.
    """
    parts = [f" [url={url}"]
    if resp is not None:
        parts.append(f" status={resp.status_code}")
        content_type = resp.headers.get("Content-Type", "")
        if content_type:
            parts.append(f" content_type={content_type}")
        body = (resp.text or "").strip()
        if body:
            # Collapse every line-break flavour (LF, CR, CRLF) to a single
            # space so multi-line bodies don't break log formatting.
            snippet = " ".join(body[:200].splitlines())
            suffix = "…" if len(body) > 200 else ""
            parts.append(f" body={snippet!r}{suffix}")
    parts.append("]")
    return "".join(parts)


def _fetch_provider_models(provider_name: str, provider_cfg: dict, timeout: int) -> list[dict]:
    """
    Fetch the model list from a single provider, apply any configured filter,
    and build a proxy model ID in the '<provider_name>__<upstream_model_id>'
    format.

    Returns an empty list on any failure so that one bad provider does not
    prevent the aggregate response from including all healthy providers.
    """
    base_url = provider_base_url(provider_cfg)
    # Most providers list models at <base_url>/models. A few expose the catalog
    # at a different path entirely (e.g. Cloudflare Workers AI has no
    # GET /v1/models and lists at /ai/models/search). Allow a per-provider
    # override so those upstreams can still be discovered.
    url = resolve_env_refs(provider_cfg.get("models_url")) or f"{base_url}/models"
    # Field on each model object that carries the upstream model id. Defaults to
    # the OpenAI "id"; Cloudflare's /ai/models/search puts the usable id (the
    # "@cf/..." name) in "name" and reserves "id" for an internal UUID.
    id_field = provider_cfg.get("models_id_field") or "id"
    # Optional task filter: when set, keep only models whose task.name matches
    # (case-insensitive). Cloudflare's catalog mixes Text Generation, embeddings,
    # image, etc. into one list; this restricts it to chat-capable models.
    keep_task = provider_cfg.get("models_keep_task")
    # Identify ourselves. This one matters more than the request paths: it
    # builds the route cache, so a CDN refusing it makes the provider vanish
    # from every pool at once, which reads as "that provider has no models"
    # rather than as a block. It also runs on a background thread, where there
    # is no client UA to relay in the first place.
    headers = {"Content-Type": "application/json",
               "User-Agent": _configured_user_agent()}
    api_key = provider_api_key(provider_cfg)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = None
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Normalize the various shapes upstreams return for /models:
        #   - OpenAI style: {"data": [...]}
        #   - Cloudflare / some gateways: {"result": [...]}
        #   - Together, GitHub catalog, and others return a bare JSON array: [...]
        raw_models: list[dict]
        if isinstance(data, list):
            raw_models = data
        elif isinstance(data, dict):
            # Pick the first key that is actually present rather than the first
            # truthy value: an upstream that legitimately returns an empty
            # {"data": []} must not fall through to "result".
            if "data" in data:
                raw_models = data["data"]
            elif "result" in data:
                raw_models = data["result"]
            else:
                raw_models = []
            if not isinstance(raw_models, list):
                raise ValueError(
                    f"unexpected 'data'/'result' type {type(raw_models).__name__}; "
                    f"top-level keys: {sorted(data.keys())}"
                )
        else:
            raise ValueError(
                f"unexpected /models payload type {type(data).__name__}"
            )
    except Exception as e:
        logger.warning(
            "[server:_fetch_provider_models] provider=%s fetch failed: %s: %s%s",
            provider_name,
            type(e).__name__,
            e,
            _describe_fetch_failure(url, resp),
        )
        model_filter = provider_cfg.get("model_filter")
        if not model_filter:
            return []
        logger.info(
            "[server:_fetch_provider_models] provider=%s: /models unavailable; "
            "synthesizing %d model(s) from model_filter",
            provider_name, len(model_filter),
        )
        raw_models = [{id_field: uid, "object": "model"} for uid in model_filter]
    model_filter = provider_cfg.get("model_filter")

    result = []
    for model in raw_models:
        upstream_id: str = model.get(id_field, "")
        if model_filter is not None and upstream_id not in model_filter:
            continue
        # Drop models whose task doesn't match the configured filter (e.g.
        # Cloudflare's catalog includes Text-to-Image and embedding tasks that
        # cannot serve chat/completions).
        if keep_task is not None:
            task = model.get("task")
            task_name = task.get("name", "") if isinstance(task, dict) else ""
            if task_name.lower() != keep_task.lower():
                logger.info(
                    "  skipping %s/%s (task=%r != %r)",
                    provider_name, upstream_id, task_name, keep_task,
                )
                continue
        # Skip embedding models — clients that validate modalities (e.g.
        # opencode) reject "embedding" as an output type, and these models
        # cannot be used for chat/completions anyway.  Check the modalities
        # field when present, and fall back to the model name for upstreams
        # (e.g. nvidia) that don't include modalities in their /models response.
        output_modalities = model.get("modalities", {}).get("output", [])
        is_embedding = "embedding" in output_modalities or (
            not output_modalities and "embed" in upstream_id.lower()
        )
        if is_embedding:
            logger.info(
                "  skipping embedding model %s/%s", provider_name, upstream_id,
            )
            continue
        # Build a proxy-facing model object.  Drop non-standard fields that
        # some upstreams (e.g. LM Studio) add and that strict clients reject.
        proxy_model = {k: v for k, v in model.items() if k != "modalities"}
        # Re-expose the upstream modalities as an OpenRouter-style ``architecture``
        # block instead of the raw ``modalities`` field.  Raw ``modalities`` is
        # dropped because strict clients (e.g. opencode) reject unexpected values
        # there, but the classification signal it carries is exactly what clients
        # need to infer a model's type, so we surface it in the well-defined
        # ``architecture`` shape that OpenRouter-schema clients (e.g. Hermes) read
        # and OpenAI-strict clients ignore.
        modalities = model.get("modalities") if isinstance(model.get("modalities"), dict) else {}
        proxy_model["architecture"] = _architecture_block(
            modalities.get("input"), modalities.get("output"),
        )
        # OpenAI-standard ``created`` (unix ts): keep upstream's value when present,
        # else fall back to a stable per-process timestamp so clients that require
        # the field don't choke.
        proxy_model.setdefault("created", _SERVER_EPOCH)
        # Normalize the context window onto the OpenRouter-standard key.
        if "context_length" not in proxy_model and "context_window" in proxy_model:
            proxy_model["context_length"] = proxy_model["context_window"]
        # Strip a duplicate provider prefix so "nvidia/nvidia/llama-x" → "llama-x".
        auto_prefix = provider_name + "/"
        stripped = upstream_id[len(auto_prefix):] if upstream_id.startswith(auto_prefix) else upstream_id
        # Use "provider__model" as the proxy ID.  The double-underscore separator
        # satisfies two constraints that previous formats failed:
        #   - no spaces or parens, so strict clients (e.g. Hermes) that validate
        #     model names against a "no whitespace / no special chars" rule accept it
        #   - no "/", so clients that silently truncate at the first "/" still show
        #     the full id in their menus
        # The provider goes first to mirror the canonical "provider/model" slash form
        # used everywhere else in the codebase.
        # Any spaces in the upstream model id or provider name are replaced with "_"
        # for the same reason — strict validators reject whitespace in model names.
        # Upstream ids with multiple slashes are flattened so the display id carries
        # at most one "/" (see _flatten_display_model); this keeps the proxy grammar
        # unambiguous — "__" always separates the provider and there is never more
        # than a single "/" — which matters for per-provider virtual-model parsing.
        # The route cache keys on this sanitized display id; routing still uses the
        # original upstream_id when forwarding to the provider.
        safe_stripped = _flatten_display_model(stripped).replace(" ", "_")
        safe_provider = provider_name.replace(" ", "_")
        proxy_id = f"{safe_provider}__{safe_stripped}"
        proxy_model["id"] = proxy_id
        proxy_model["name"] = proxy_id
        proxy_model["_upstream_id"] = upstream_id
        proxy_model["_route"] = (provider_name, upstream_id)
        proxy_model["_provider"] = provider_name
        result.append(proxy_model)

    filter_desc = f"filter={model_filter}" if model_filter is not None else "no filter"
    logger.info(
        "[server:_fetch_provider_models] provider=%s: %d/%d models kept (%s)",
        provider_name, len(result), len(raw_models), filter_desc,
    )
    return result


def _rebuild_route_cache(providers_cfg: dict, timeout: int,
                         only_if_empty: bool = False) -> list[dict]:
    """
    Fetch models from all providers concurrently, rebuild _model_route_cache
    atomically, and return the full flat model list.

    The cache is replaced wholesale on each call so that removed or renamed
    upstream models do not linger as stale entries.

    When ``only_if_empty`` is set (the warm-on-empty paths), the freshly fetched
    cache is applied only if the live cache is *still* empty under the lock — the
    network fetch can take seconds, and a concurrent request (or, in tests, a
    direct seed) may have populated the cache meanwhile. Clobbering it then would
    wipe live routing data; the warm should defer to whoever populated it first.
    The flat model list is still returned either way.
    """
    if not providers_cfg:
        with _model_route_cache_lock:
            _model_route_cache.clear()
        return []

    all_models: list[dict] = []
    # Announce the sweep BEFORE it runs, not only when it finishes. A rebuild
    # fans out to every provider's /models and can take seconds; logged only on
    # completion, a slow one is indistinguishable from a hang, and the request
    # waiting on it shows nothing between its arrival line and its first
    # candidate. One line here is the difference between "it is working" and an
    # hour of guessing.
    _rebuild_started = time.monotonic()
    logger.info(
        "[server:_rebuild_route_cache] fetching listings from %d provider(s)…",
        len(providers_cfg),
    )

    with ThreadPoolExecutor(max_workers=min(len(providers_cfg), 10)) as executor:
        futures = {}
        for name, cfg in providers_cfg.items():
            if name in RESERVED_PROVIDER_NAMES:
                logger.error(
                    "[server:_rebuild_route_cache] Provider name %r is reserved; "
                    "skipping it to avoid virtual-model namespace collision. "
                    "Rename it in your config.",
                    name,
                )
                continue
            futures[executor.submit(_fetch_provider_models, name, cfg, timeout)] = name
        if not futures and providers_cfg:
            logger.error(
                "[server:_rebuild_route_cache] All configured provider names are reserved "
                "(%s); no real providers will be queried.",
                ", ".join(repr(n) for n in providers_cfg),
            )
        for future in as_completed(futures):
            try:
                all_models.extend(future.result())
            except Exception as e:
                provider_name = futures[future]
                logger.warning(
                    "[server:_rebuild_route_cache] Unexpected error from provider %s: %s",
                    provider_name, e,
                )

    new_cache: dict[str, tuple[str, str]] = {}
    new_context: dict[str, int] = {}
    new_caps: dict[str, set[str]] = {}
    for m in all_models:
        route = m.pop("_route", None)
        window = _coerce_context_length(m.get("context_length"))
        caps = capabilities_from_listing(m)
        if route:
            # Dual-key on both the canonical "provider__model" id and the advertised
            # "provider/model" form so an inbound id in either form resolves to the
            # exact upstream losslessly (no string-level reverse needed on the hot path).
            new_cache[m["id"]] = route
            new_cache[_display_id(m["id"])] = route
            if window is not None:
                new_context[f"{route[0]}/{route[1]}".lower()] = window
            if caps:
                new_caps[f"{route[0]}/{route[1]}".lower()] = caps

    with _model_route_cache_lock:
        if only_if_empty and _model_route_cache:
            logger.info(
                "[server:_rebuild_route_cache] cache populated concurrently "
                "(%d entries); keeping it, discarding warm result.",
                len(_model_route_cache),
            )
            return all_models
        _model_route_cache.clear()
        _model_route_cache.update(new_cache)
        # Swapped inside the same acquisition as the route cache: a reader must
        # never see routes from one rebuild and context from another.
        _model_context_cache.clear()
        _model_context_cache.update(new_context)
        _model_capability_cache.clear()
        _model_capability_cache.update(new_caps)
    # The listing layer is built from the capability snapshot just swapped, so
    # every memoized merge is now stale. Outside the lock: the bump takes a
    # different one and nothing here needs them held together.
    _bump_routing_generation()

    logger.info(
        # Report MODELS, not cache entries. The cache is dual-keyed, so
        # len(new_cache) is twice the model count while len(new_context) is not
        # — printed side by side they read as though at most half the models
        # have a known context window.
        "[server:_rebuild_route_cache] %d model(s) (%d with a known context window) in %.1fs",
        len(set(new_cache.values())), len(new_context),
        time.monotonic() - _rebuild_started,
    )
    return all_models


def _coerce_context_length(value) -> int | None:
    """Coerce an upstream ``context_length``/``context_window`` to a positive int.

    Returns None for anything unusable: missing, bool (an int subclass, so it has
    to be excluded explicitly), zero or negative, or non-numeric. Callers treat
    None as *unknown*, which is neutral — deliberately, because reading junk as a
    tiny window would demote a perfectly good model.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _get_model_context_snapshot() -> dict[str, int]:
    """Point-in-time copy of the discovered per-model context windows.

    Unlike ``_get_route_cache_snapshot`` this never warms the cache: it is only
    consulted on a path where the route cache has already been read, and an empty
    map simply means "no context metadata", which every consumer treats as
    neutral.
    """
    with _model_route_cache_lock:
        return dict(_model_context_cache)


def _get_model_capability_snapshot() -> dict[str, set[str]]:
    """Point-in-time copy of what each provider says its own models can do."""
    with _model_route_cache_lock:
        return {k: set(v) for k, v in _model_capability_cache.items()}


def _get_route_cache_snapshot() -> dict[str, tuple[str, str]]:
    """
    Return a point-in-time copy of the route cache.

    If the cache is empty (e.g. gunicorn worker that has not yet served a
    /v1/models request), the cache is warmed on-demand before the snapshot
    is taken so that virtual model routing works from the very first request.
    """
    with _model_route_cache_lock:
        if _model_route_cache:
            return dict(_model_route_cache)

    config = load_config()
    providers_cfg = config.get("providers", {})
    timeout = config.get("server", {}).get("request_timeout", 120)
    _rebuild_route_cache(providers_cfg, timeout, only_if_empty=True)

    with _model_route_cache_lock:
        return dict(_model_route_cache)


def _get_distinct_routes() -> list[tuple[str, str]]:
    """Every ``(provider, upstream_id)`` in the route cache, each exactly once.

    The cache is dual-keyed: ``_rebuild_route_cache`` stores every model under
    both the canonical ``provider__model`` id and the advertised
    ``provider/model`` form, so an inbound id in either shape resolves without
    string surgery. That is right for a lookup and wrong for a walk — iterating
    ``.items()`` yields every model twice, which silently doubled every virtual
    candidate pool in the proxy.

    Callers building a pool want routing targets, not cache keys, so they walk
    this instead. First-seen order is preserved, which is canonical-id insertion
    order, so the sequence every downstream ordering pass builds on is exactly
    what it was minus the duplicate.
    """
    return list(dict.fromkeys(_get_route_cache_snapshot().values()))


def _sync_local_provider_models_once() -> None:
    """
    On first call after startup, poll every localhost provider's /models endpoint
    and sync unprefixed model IDs into config['model_reasoning'].

    Local providers participate in the dedicated llmproxy__local and
    llmproxy__<level>/local virtual-endpoint families, not in llmproxy__free.
    Models served from a localhost URL are therefore NOT added to
    config['believed_free'] — believed_free is reserved for provider grace
    tiers ("free" as in dollars), while /local routes on host topology.

    Behaviour:
      - Models with "/" in their ID are skipped (they are externally namespaced
        passthroughs like 'openai/gpt-4o' piped through OpenWebUI).
      - Models no longer returned by a local provider are pruned from
        model_reasoning.
      - As a one-time cleanup, any pre-existing believed_free / free_limits
        entries for a local provider are also pruned (corrects historical
        configs that were polluted before this fix landed).
      - If the config changes it is persisted to disk so the sync survives
        server restart.
      - Runs in a background thread so it never blocks the first /v1/models
        response.
    """
    global _local_sync_done
    with _local_sync_lock:
        if _local_sync_done:
            return
        _local_sync_done = True

    def _run() -> None:
        config = load_config(force_reload=True)
        providers: dict = config.get("providers", {})

        local_providers = {
            name: cfg for name, cfg in providers.items()
            if _is_local_url(provider_base_url(cfg))
        }
        if not local_providers:
            return

        # These land in the sidecar's curated section, not config.json. This
        # sync is a machine process, and having it rewrite the file a person
        # hand-edits is what made config.json unusable as a record of intent —
        # it silently re-seeded the very keys a migration had just stripped.
        #
        # Read without the write lock: this is a read, and taking the
        # transaction here would rewrite the file just to look at it.
        _curated = (_load_routing_sidecar() or {}).get("curated") or {}
        existing_kf: list = list(_curated.get("believed_free") or [])
        existing_mr: dict = dict(_curated.get("model_reasoning") or {})
        existing_fl: dict = dict(_curated.get("free_limits") or {})
        # Deltas rather than a wholesale snapshot. The network calls below take
        # seconds, and assigning the whole section afterwards would discard any
        # edit made in between — including an admin correction to a model this
        # sync knows nothing about.
        drop_kf: set[str] = set()
        drop_fl: set[str] = set()
        drop_mr: set[str] = set()
        add_mr: dict[str, str] = {}
        modified = False

        for provider_key, provider_cfg in local_providers.items():
            base_url = provider_base_url(provider_cfg)
            api_key = provider_api_key(provider_cfg)
            local_headers = {"User-Agent": _configured_user_agent()}
            if api_key:
                local_headers["Authorization"] = f"Bearer {api_key}"
            try:
                resp = requests.get(
                    f"{base_url}/models",
                    headers=local_headers,
                    timeout=8,
                )
                resp.raise_for_status()
                live_ids = {m.get("id", "") for m in resp.json().get("data", []) if m.get("id")}
            except Exception as exc:
                logger.warning("[local-sync] Could not reach '%s': %s", provider_key, exc)
                continue

            # Compute expected set for this provider (unprefixed only)
            prefix = f"{provider_key}/"
            expected = {f"{prefix}{mid}" for mid in live_ids if mid and "/" not in mid}

            # Prune ALL believed_free entries for this provider — local models
            # never belong here (one-time cleanup for historically polluted configs).
            stale_kf = [e for e in existing_kf if e.startswith(prefix)]
            for e in stale_kf:
                drop_kf.add(e)
                modified = True
                logger.info(
                    "[local-sync] Removed %s from believed_free "
                    "(local provider — routed via llmproxy__local instead).", e,
                )

            # Same for free_limits — local models don't use the capacity-aware
            # free-tier scheduler.
            stale_fl = [k for k in existing_fl if isinstance(k, str) and k.startswith(prefix)]
            for k in stale_fl:
                drop_fl.add(k)
                modified = True
                logger.info("[local-sync] Removed %s from free_limits.", k)

            # Prune stale model_reasoning entries that this provider previously
            # contributed but no longer serves.
            stale_mr = [k for k in existing_mr if k.startswith(prefix) and k not in expected]
            for k in stale_mr:
                drop_mr.add(k)
                modified = True
                logger.info("[local-sync] Pruned stale model_reasoning: %s", k)

            # Add new model_reasoning entries so /local/<level> routing works.
            for qualified in expected:
                if qualified not in existing_mr:
                    model_id = qualified[len(prefix):]
                    add_mr[qualified] = _infer_local_reasoning_level(model_id)
                    modified = True
                    logger.info("[local-sync] Added model_reasoning: %s -> %s",
                                qualified, add_mr[qualified])

        if modified:
            # Apply the deltas to whatever the file says NOW, so nothing written
            # while the providers were being polled is lost.
            with _routing_sidecar_txn() as state:
                curated = _curated_facts(state)
                kf = curated.setdefault("believed_free", [])
                curated["believed_free"] = [e for e in kf if e not in drop_kf]
                fl = curated.setdefault("free_limits", {})
                for k in drop_fl:
                    fl.pop(k, None)
                mr = curated.setdefault("model_reasoning", {})
                for k in drop_mr:
                    mr.pop(k, None)
                for k, level in add_mr.items():
                    mr.setdefault(k, level)   # never over-write a later edit

    import threading as _t
    _t.Thread(target=_run, daemon=True, name="local-model-sync").start()


class _LineLoggingStream(io.TextIOBase):
    """Write-only text stream that emits each completed line via a callback.

    Passed to contextlib.redirect_stdout so a subprocess-free script that reports
    progress with print() has each line streamed to the server log in real time,
    rather than buffered and dumped all at once when the script returns. ANSI
    color codes are stripped before logging.
    """

    _ansi = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, log_fn: Callable[[str], None]) -> None:
        self._log = log_fn
        self._buf = ""

    def write(self, s: str) -> int:  # noqa: D102
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def _emit(self, line: str) -> None:
        clean = self._ansi.sub("", line).rstrip()
        if clean:
            self._log(clean)

    def flush(self) -> None:  # noqa: D102
        if self._buf:
            self._emit(self._buf)
            self._buf = ""


def _warm_route_cache_if_empty() -> None:
    """Populate the virtual-model route cache from config, unless already warm.

    Skips the rebuild when the cache is already populated (e.g. a /v1/models
    request beat us to it) so the eager startup warm never clobbers an existing
    cache nor issues a redundant upstream fetch. Mirrors the provider selection in
    list_models() (reserved names skipped). Best-effort: failures are logged,
    never raised.
    """
    with _model_route_cache_lock:
        if _model_route_cache:
            return
    config = load_config()
    providers_cfg = {
        k: v for k, v in config.get("providers", {}).items()
        if k not in RESERVED_PROVIDER_NAMES
    }
    if not providers_cfg:
        return
    timeout = config.get("server", {}).get("request_timeout", 120)
    try:
        _rebuild_route_cache(providers_cfg, timeout, only_if_empty=True)
    except Exception as exc:  # noqa: BLE001 — warming must never crash the worker
        logger.warning("[startup] route-cache warm failed: %s", exc)


def _sync_believed_free_from_sidecar(config_path: str | None) -> bool:
    """Reconcile the live config.json's free-tier sections from the bundled sidecar.

    Unlike _run_free_models_update this does **no** network scraping and never
    rewrites providers.json / config.example.json — it only reconciles
    believed_free / free_limits / model_reasoning / model_capabilities into the
    user config from the data that already ships in providers.json. Because it
    never writes the sidecar, it works even when the sidecar is read-only (an
    installed package or a container image layer), so the shipped/merged
    free-tier data reaches the live config without the full updater.

    Returns True if the sync ran (so the caller can refresh the models cache).
    """
    import os
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from scripts.update_free_models import main as _update_main
    except Exception as exc:  # noqa: BLE001 — updater is optional at runtime
        logger.warning(
            "[startup-sync] updater unavailable in this deployment "
            "(scripts/ not found next to the package): %s", exc,
        )
        return False
    path = config_path
    if not path:
        try:
            from .config import get_config_path
            path = str(get_config_path(None))
        except Exception:  # noqa: BLE001
            path = None
    if not path:
        logger.warning("[startup-sync] no config path resolved; skipping live sync")
        return False
    logger.info("[startup-sync] reconciling %s from bundled providers.json", path)
    stream = _LineLoggingStream(lambda line: logger.info("[startup-sync] %s", line))
    # Serialize against admin config edits: the reconcile is a read-modify-write
    # of config.json, and the admin API guards its writes with the same lock, so
    # a concurrent admin edit can't clobber it (and vice versa).
    from .admin import _locked  # local import: admin is wired after routes
    try:
        with _locked(), contextlib.redirect_stdout(stream):
            _update_main(["--sync-config-only", "--config", path])
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 — never let a sync failure crash the worker
        logger.warning("[startup-sync] failed: %s", exc)
        return False
    finally:
        stream.flush()
    return True


def _run_free_models_update(config: dict, config_path: str | None) -> bool:
    """Run scripts/update_free_models and stream its output to the server log.

    Returns True if the updater actually ran (so the caller knows to refresh the
    virtual-model cache afterwards), False if the updater package is unavailable.
    The updater refreshes believed_free / free_limits / pricing in providers.json,
    regenerates config.example.json, and syncs the user config; its changes are
    picked up by the normal mtime-based config reload.

    When free_tier.cost_probe.enabled is true the updater also actively probes
    believed_free models for cost (see scripts/sources/cost_probe.py).
    """
    # The scraper lives in the repo-root `scripts/` package, which sits next to
    # the installed `llmproxy/` package but may not be on sys.path (e.g. under
    # gunicorn). Add the package's parent dir so `import scripts` resolves
    # whenever scripts/ shipped alongside the package.
    import os
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from scripts.update_free_models import main as _update_main
    except Exception as exc:  # noqa: BLE001 — updater is optional at runtime
        logger.warning(
            "[startup-update] updater unavailable in this deployment "
            "(scripts/ not found next to the package): %s", exc,
        )
        return False
    argv: list[str] = []
    path = config_path
    if not path:
        try:
            from .config import get_config_path
            path = str(get_config_path(None))
        except Exception:  # noqa: BLE001
            path = None
    if path:
        argv += ["--config", path]
    logger.info("[startup-update] running update_free_models %s", argv or "(sidecar only)")
    # Snapshot the sidecar so we can tell whether the scrape actually changed it
    # (and therefore whether a PR is warranted).
    from . import providers as _providers_mod
    sidecar_path = _providers_mod.DATA_PATH
    before = sidecar_path.read_bytes() if sidecar_path.exists() else b""
    # Stream the updater's print() progress (providers.json / config.example.json
    # writes, believed_free adds/removes, config sync) to the server log line by
    # line so it is visible in docker logs as it happens.
    stream = _LineLoggingStream(lambda line: logger.info("[startup-update] %s", line))
    try:
        with contextlib.redirect_stdout(stream):
            _update_main(argv)
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 — never let a scrape failure crash the worker
        logger.warning("[startup-update] failed: %s", exc)
    finally:
        stream.flush()
    after = sidecar_path.read_bytes() if sidecar_path.exists() else b""
    providers_text: str | None = None
    example_text: str | None = None
    if after != before:
        # Sidecar persisted normally (writable deployment).
        logger.info("[startup-update] providers.json changed")
        providers_text = after.decode("utf-8")
        example_file = os.path.join(repo_root, "config.example.json")
        if os.path.exists(example_file):
            with open(example_file, encoding="utf-8") as fh:
                example_text = fh.read()
    else:
        # Bundled copy unchanged. On a read-only image the updater mirrors the
        # computed artifacts to the (writable) user-config dir; use those so a
        # PR can still be opened even though providers.json couldn't be persisted.
        try:
            from .config import get_config_path
            fb_dir = get_config_path(config_path).parent
            fb_providers = fb_dir / "providers.json"
            if fb_providers.exists() and fb_providers.read_bytes() != before:
                providers_text = fb_providers.read_text(encoding="utf-8")
                fb_example = fb_dir / "config.example.json"
                if fb_example.exists():
                    example_text = fb_example.read_text(encoding="utf-8")
                logger.info("[startup-update] providers.json changed "
                            "(computed; bundled copy is read-only)")
        except Exception as exc:  # noqa: BLE001 — fallback detection is best-effort
            logger.warning("[startup-update] fallback sidecar check failed: %s", exc)

    if providers_text is not None:
        _maybe_open_providers_pr(config, providers_text, example_text)
    else:
        logger.info("[startup-update] providers.json unchanged")
    logger.info("[startup-update] complete")
    return True


def _maybe_fire_interval_probes(config_path: str | None = None) -> None:
    """Check frequency intervals for the free-models refresh, the cost probe,
    and PR creation.

    Fires each as a background daemon thread if its interval has elapsed.
    Gated by _PROBE_INTERVAL_GATE_SEC so state files are not read on every
    single request — the actual cadence is set in config.json.

    Free-models refresh: gated by sync_on_startup OR update_on_startup, and
      throttled by free_tier.update_frequency_days (default 7).
    Cost probe: gated by update_on_startup AND cost_probe.enabled.
    PR creation: checked independently of startup flags.
    """
    global _last_probe_interval_check
    now = time.monotonic()
    with _probe_interval_check_lock:
        if now - _last_probe_interval_check < _PROBE_INTERVAL_GATE_SEC:
            return
        _last_probe_interval_check = now

    try:
        config = load_config()
    except Exception:  # noqa: BLE001
        return
    free_tier = config.get("free_tier", {}) if isinstance(config.get("free_tier"), dict) else {}

    # Full free-models refresh — gated by sync_on_startup OR update_on_startup.
    if free_tier.get("sync_on_startup") or free_tier.get("update_on_startup"):
        _maybe_fire_free_models_update(config, free_tier, config_path)

    # Flagship membership — independent cadence, gated by flagship_tier.enabled.
    _maybe_fire_flagship_refresh(config, flagship_tier_cfg(config), config_path)
    # Routing metadata — its own cadence, so turning the flagship tier off does
    # not also stop llmproxy learning what its models can do.
    _maybe_fire_routing_metadata_refresh(
        config, routing_metadata_cfg(config), config_path)

    # Cost probe — gated by update_on_startup + cost_probe.enabled.
    if free_tier.get("update_on_startup") and free_tier.get("cost_probe", {}).get("enabled"):
        _maybe_fire_cost_probe(config, free_tier, config_path)

    # PR creation interval — independent of startup flags.
    _maybe_fire_pr_if_due(config, config_path)


# Canonical value lives in config.DEFAULT_FREE_TIER_CONFIG so the runtime
# fallback and the generated config.example.json cannot drift apart.
DEFAULT_UPDATE_FREQUENCY_DAYS = DEFAULT_FREE_TIER_CONFIG["update_frequency_days"]


def _free_update_due(free_tier: dict, config_path: str | None) -> bool:
    """Whether the full free-models refresh is due per update_frequency_days.

    Returns False when the updater is not importable in this deployment, since
    there is then nothing to run. A frequency of 0 or less means "every time",
    matching the other throttles in this module.
    """
    try:
        import os as _os
        import sys
        repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from llmproxy.config import load_update_state
        from scripts.update_free_models import _probe_due
    except Exception:  # noqa: BLE001 — scripts/ may not be available
        return False
    freq_days = free_tier.get("update_frequency_days", DEFAULT_UPDATE_FREQUENCY_DAYS)
    state = load_update_state(config_path)
    due, _ = _probe_due(state.get("last_update_at"), freq_days)
    return due


def _live_flagship_free_count(config: dict, config_path: str | None) -> int:
    """Distinct models in the flagship tier that are free *right now*.

    Membership is cached at refresh time, but free-ness is re-evaluated on every
    read from live config. A member that becomes cost-observed after the refresh
    therefore drops out of ``flagship__free`` silently, and the tier shrinks
    below the floor it promised until the next cadence tick — up to
    ``refresh_frequency_days``, seven by default.

    Counted in distinct models rather than routing targets, matching the unit
    ``min_flagship_free_models`` is expressed in: the same weights served free
    by two providers are two members but one model.

    Reads the live member set through ``_get_flagship_models``, so pins and
    excludes from the current config are already applied.
    """
    from .flagship import normalize_model_id

    members = _get_flagship_models(config, config_path)
    if not members:
        return 0
    believed = _normalized_believed_free(config)
    observed = _normalized_cost_observed(config)
    scoped = _provider_scoped_ids(config)
    paid_providers = _cost_observed_providers(config)

    keys: set[str] = set()
    for qualified in members:
        provider, _, upstream = qualified.partition("/")
        if not upstream:
            continue
        if _is_model_free_with(provider, upstream, believed, observed, scoped,
                               paid_providers):
            keys.add(normalize_model_id(upstream))
    return len(keys)


def _flagship_refresh_due(tier_cfg: dict, config_path: str | None,
                          config: dict | None = None) -> bool:
    """Whether the flagship membership recompute is due.

    Mirrors _free_update_due. A frequency of 0 or less means "every time", and
    a disabled tier is never due.
    """
    if not tier_cfg.get("enabled", True):
        return False
    try:
        import os as _os
        import sys
        repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.update_free_models import _probe_due
    except Exception:  # noqa: BLE001
        return False
    state = load_flagship_state(config_path)
    # A cache written before scores were persisted carries membership but no
    # ranking, and the router silently falls back to the old ordering until it
    # is rewritten. That is a schema upgrade, not a staleness question, so the
    # cadence must not gate it: every existing deployment already holds a recent
    # last_refresh_at, and would otherwise keep the pre-ranking behaviour for up
    # to refresh_frequency_days after upgrading — a week, by default.
    #
    # Test for the KEY, not a truthy value. A deployment no benchmark source
    # covers legitimately has `model_scores: {}`, and a falsiness check would
    # make it due on every interval tick — a catalog re-fetch every minute,
    # forever. Absence of the key is true only of a genuinely pre-upgrade file,
    # so this fires once and the ordinary cadence resumes.
    if state.get("members") and "model_scores" not in state:
        logger.info(
            "[flagship] cached membership predates benchmark scores — recomputing "
            "so the tier can be ranked"
        )
        return True
    due, _ = _probe_due(state.get("last_refresh_at"),
                        tier_cfg.get("refresh_frequency_days", 7))
    if due:
        return True

    # The floor is a promise about the tier, not about the bar walk that built
    # it, so it is re-checked against what the tier holds now. A member that
    # was free at refresh time and is cost-observed now has already left
    # `flagship__free`; waiting out the cadence leaves the pool short for up to
    # a week. Same precedent as the schema check above: an inadequate cache is
    # a correctness question, and the cadence must not gate it.
    min_free = int(tier_cfg.get("min_flagship_free_models") or 0)
    previously = state.get("free_models")
    if min_free <= 0 or not state.get("members") or not isinstance(previously, list):
        return False
    # Only when a recompute could actually help. If the LAST run already came
    # up short, this deployment simply does not have that many free models, and
    # recomputing would re-fetch every catalog on every interval tick forever.
    # Compare against what the previous run achieved, not against the floor
    # alone.
    if len(previously) < min_free:
        _warn_flagship_floor(len(previously), min_free)
        return False
    cfg = config if config is not None else load_config()
    live = _live_flagship_free_count(cfg, config_path)
    if live >= min_free:
        return False
    logger.info(
        "[flagship] only %d of %d free model(s) remain in the tier "
        "(excluded, or no longer free) — recomputing ahead of the cadence",
        live, min_free,
    )
    return True


def _warn_flagship_floor(achieved: int, min_free: int) -> None:
    """Say once that the floor cannot be met, naming the shortfall.

    A tier that under-fills for a reason no recompute can fix should say so
    rather than quietly serving fewer models than its config asks for. Latched,
    because the check runs on every interval tick.
    """
    global _flagship_floor_warned
    if _flagship_floor_warned:
        return
    _flagship_floor_warned = True
    logger.warning(
        "[flagship] min_flagship_free_models is %d but the last refresh found "
        "only %d free model(s) this deployment can reach — serving what there "
        "is. Add a provider, or lower flagship_tier.min_flagship_free_models.",
        min_free, achieved,
    )


def _flagship_specs_for(
    provider_name: str,
    upstream_id: str,
    profile: dict,
    cap_snapshot: dict[str, set[str]],
    ctx_snapshot: dict[str, int],
) -> tuple[int | None, bool | None]:
    """The ``(context_length, supports_tools)`` to gate one routing target on.

    Three sources, most specific first, because a spec belongs to the endpoint a
    request actually hits rather than to the weights in the abstract:

    1. **The provider's own listing for this exact qualified id.** A gateway is
       authoritative about its own routing target. A non-empty capability set
       that omits ``tools`` means False, not unknown: ``_rebuild_route_cache``
       only files a set when it is non-empty, so "no entry" genuinely means the
       provider said nothing and correctly falls through.
    2. **The catalog's entry for this exact id** (``profile["by_id"]``). For a
       gateway whose ids are the catalog's own, this is what the catalog said
       about *this* variant rather than about its siblings.
    3. **The merged profile**, joined on the normalized key. This is the
       documented cross-provider carry-across: a provider that publishes no
       capability data of its own is still gated on the same weights served
       elsewhere. It stays ``None`` on a miss, so a model nothing can verify
       still fails the spec gate.

    The first two exist because the merged view cannot distinguish a billing
    variant from its parent: ``z-ai/glm-5.2:free`` normalizes onto
    ``z-ai/glm-5.2`` and would otherwise inherit tool support it does not have
    and a 1M context window that is really 32k. Deliberately NOT
    ``_lookup_capabilities`` or ``_lookup_model_fact``: both reach the
    normalized form, the first by unioning it and the second by falling back to
    it, which is exactly the inheritance this is here to prevent.
    """
    qualified = f"{provider_name}/{upstream_id}".lower()

    listed_caps = cap_snapshot.get(qualified)
    supports_tools: bool | None = None
    if listed_caps:
        supports_tools = "tools" in listed_caps

    context_length = ctx_snapshot.get(qualified)

    by_id = (profile.get("by_id") or {}).get(upstream_id.lower()) or {}
    if supports_tools is None and "supports_tools" in by_id:
        supports_tools = by_id["supports_tools"]
    if context_length is None:
        context_length = by_id.get("context_length")

    if supports_tools is None:
        supports_tools = profile.get("supports_tools")
    if context_length is None:
        context_length = profile.get("context_length")
    return context_length, supports_tools


def _recompute_flagship_members(config: dict, config_path: str | None) -> dict | None:
    """Recompute flagship membership against everything this deployment sees.

    The candidate pool is the whole route cache — every model of every
    configured provider, paid included — not just believed_free and not just
    one gateway's catalog. Free status is evaluated per candidate, so the same
    weights can be free on one provider and paid on another.

    Capability specs come from the provider's own listing for this exact
    routing target where it publishes one, then from the catalog's entry for
    this exact id, and only then from the benchmark profile joined on the
    normalised model key — see ``_flagship_specs_for``. That last join is
    heuristic, which is why a pin exists to override it, and why the two
    exact-id sources take precedence: a ``:free`` variant must not be admitted
    on the tool support and context window of the paid sibling it normalises
    onto.

    Returns the state written to flagship_models.json, or None if the refresh
    could not run.
    """
    from .flagship import (
        Candidate,
        fetch_profiles,
        normalize_model_id,
        select_flagship,
    )

    tier_cfg = flagship_tier_cfg(config)
    profiles = fetch_profiles(tier_cfg.get("sources"))
    if not profiles:
        logger.warning("[flagship] no benchmark source returned data; keeping previous membership")
        return None

    cap_snapshot = _get_model_capability_snapshot()
    ctx_snapshot = _get_model_context_snapshot()

    candidates: list[Candidate] = []
    for provider_name, upstream_id in _get_distinct_routes():
        provider_cfg = get_provider(config, provider_name)
        if not provider_cfg or not _provider_exposes_to_virtual_models(provider_cfg):
            continue
        profile = profiles.get(normalize_model_id(upstream_id), {})
        context_length, supports_tools = _flagship_specs_for(
            provider_name, upstream_id, profile, cap_snapshot, ctx_snapshot,
        )
        candidates.append(Candidate(
            provider=provider_name,
            upstream_id=upstream_id,
            is_free=(not _is_local_url(provider_base_url(provider_cfg))
                     and _is_model_free(provider_name, upstream_id, config)),
            context_length=context_length,
            supports_tools=supports_tools,
            scores=dict(profile.get("scores") or {}),
        ))

    selection = select_flagship(candidates, tier_cfg)
    if selection.unverified_pins:
        logger.warning(
            "[flagship] pinned but not found among this deployment's models "
            "(admitted anyway, and unverifiable against the spec gate): %s",
            ", ".join(selection.unverified_pins),
        )
    state = {
        "last_refresh_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "bar": selection.bar,
        "members": selection.members,
        # The score that admitted each member, and the score of every model this
        # deployment can see. `members` alone says who is in the tier but not how
        # strong each one is, and the router needs the latter to walk the pool
        # strongest-first. See _get_flagship_scores.
        "scores": selection.scores,
        "model_scores": selection.model_scores,
        "distinct_models": selection.distinct_models,
        "free_models": selection.free_models,
        "candidates_considered": len(candidates),
    }
    save_flagship_state(state, config_path)
    logger.info(
        "[flagship] %d routing target(s) across %d distinct model(s), "
        "%d of them free, from %d candidate(s)",
        len(selection.members), len(selection.distinct_models),
        len(selection.free_models), len(candidates),
    )
    return state


# How many models of a family must carry observed capabilities before the family
# is allowed to lend its unanimous set to a sibling that carries none. Two
# models agreeing is not evidence about a third; measured against the shipped
# providers.json, three is where the families that emerge are ones you would
# recognise (glm -> reasoning, gemma -> vision) rather than accidents.
# Overridable as routing_metadata.min_family_members.
_DEFAULT_MIN_FAMILY_MEMBERS = 3


def _routing_metadata_due(meta_cfg: dict, config_path: str | None) -> bool:
    """Whether the routing-metadata recompute is due. Mirrors the flagship gate."""
    if not meta_cfg.get("enabled", True):
        return False
    try:
        import os as _os
        import sys as _sys
        repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if repo_root not in _sys.path:
            _sys.path.insert(0, repo_root)
        from scripts.update_free_models import _probe_due
    except Exception:  # noqa: BLE001
        return False
    state = load_routing_metadata(config_path)
    due, _ = _probe_due(state.get("last_refresh_at"),
                        meta_cfg.get("refresh_frequency_days", 7))
    return due


def _family_capability_profiles(
    observed: dict[str, set[str]],
    raw_for_key: dict[str, str],
    min_members: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """What every model in a family agrees it can do.

    Returns ``(by_generation, by_bare)`` mapping a family key to the capability
    set UNANIMOUS among its members that carry observed data. Unanimity, not a
    majority: a family spanning coder, omni and vision variants agrees on what
    the weights share and disagrees on the rest, and only the agreement is safe
    to lend to a sibling we know nothing about. Measured against the shipped
    providers.json, this is what makes ``glm`` a reasoning family while ``qwen``
    contributes only ``tools`` across its twenty members.

    Families smaller than *min_members* say nothing, because one or two models
    agreeing is not evidence about a third.
    """
    from .providers import family_key

    gen_members: dict[str, list[set[str]]] = {}
    bare_members: dict[str, list[set[str]]] = {}
    for key, caps in observed.items():
        if not caps:
            continue
        raw = raw_for_key.get(key) or key
        gen = family_key(raw)
        bare = family_key(raw, generation=False)
        if gen:
            gen_members.setdefault(gen, []).append(set(caps))
        if bare:
            bare_members.setdefault(bare, []).append(set(caps))

    def _unanimous(groups: dict[str, list[set[str]]]) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for fam, sets in groups.items():
            if len(sets) < min_members:
                continue
            shared = set.intersection(*sets)
            if shared:
                out[fam] = shared
        return out

    return _unanimous(gen_members), _unanimous(bare_members)


def _family_capabilities_for(
    model_key: str, raw_id: str,
    by_gen: dict[str, set[str]], by_bare: dict[str, set[str]],
) -> tuple[set[str] | None, str | None]:
    """The capability set a family will lend this model, and which family lent it.

    Three attempts, most specific first:

    1. the model's own GENERATION family, so llama-4's tools never reach llama-2;
    2. its BARE family, for a generation too sparse to speak;
    3. any known family that appears as a SUBSTRING of its normalized key, with
       the LONGEST match winning.

    The third exists because deriving a family from the id only works when the
    vendor sits behind a path separator. A provider that folds it into the name
    — ``zai-glm-5-turbo``, ``claude-haiku-4.5-us-east-1`` — derives ``zaiglm5``
    or ``claudehaiku45useast1``, families of one with nothing to lend, while
    ``glm`` and ``claudehaiku`` sit right there with observed data. Matching on
    the normalized key, which has already had its separators stripped, finds
    them.

    Longest wins because a more specific family must be able to supersede a more
    general one: ``llama32`` has to beat ``llama3``, since llama-3.2's vision
    variants do not share llama-3's capability set. Shortest-wins would actively
    mislead there.

    Families shorter than ``FAMILY_MIN_SUBSTRING_LENGTH`` are excluded from the
    substring pass only. An exact match is trusted at any length.
    """
    from .providers import FAMILY_MIN_SUBSTRING_LENGTH, family_key

    gen = family_key(raw_id)
    if gen and gen in by_gen:
        return by_gen[gen], gen
    bare = family_key(raw_id, generation=False)
    if bare and bare in by_bare:
        return by_bare[bare], bare

    best_fam: str | None = None
    best_caps: set[str] | None = None
    for table in (by_gen, by_bare):
        for fam, caps in table.items():
            if len(fam) < FAMILY_MIN_SUBSTRING_LENGTH or fam not in model_key:
                continue
            if best_fam is None or len(fam) > len(best_fam):
                best_fam, best_caps = fam, caps
    return best_caps, best_fam


def _merge_model_facts(
    previous: dict, learned: dict[str, dict], fact_keys: tuple[str, ...] = ("capabilities", "reasoning"),
) -> tuple[dict[str, dict], dict[str, int]]:
    """Fold this pass's findings into what earlier passes knew.

    Two rules, and the whole point of the function is that neither was honoured
    before: a model this pass could not see KEEPS what it had, so one provider
    being down at refresh time cannot thin the routing data; and a fact may be
    replaced only by one of equal or greater provenance, so an inference can
    never overwrite a reading and nothing can overwrite a hand correction.

    Replacing ``by_model`` wholesale, as this used to, destroyed every
    ``reasoning`` tag on the first run — the refresh writes only capabilities,
    so there was nothing to carry the tiers forward.
    """
    from .providers import fact_rank

    merged: dict[str, dict] = {}
    for key, facts in (previous or {}).items():
        if isinstance(key, str) and isinstance(facts, dict):
            merged[key] = dict(facts)

    counts = {"kept": 0, "written": 0, "refused": 0}
    for key, facts in learned.items():
        entry = merged.setdefault(key, {})
        for fact in fact_keys:
            if fact not in facts:
                continue
            incoming_rank = fact_rank(facts.get(f"{fact}_source"))
            if fact in entry and fact_rank(entry.get(f"{fact}_source")) > incoming_rank:
                counts["refused"] += 1
                continue
            entry[fact] = facts[fact]
            entry[f"{fact}_source"] = facts.get(f"{fact}_source", "observed")
            counts["written"] += 1
    # Models this pass never mentioned, which is what "carried forward" means.
    # Subtracting the two sizes counted nothing of the sort once they overlapped,
    # and read 0 on a pass that carried hundreds — misleading exactly the person
    # reading this log after the next data-loss report.
    counts["kept"] = len(set(merged) - set(learned))
    return merged, counts


def _recompute_routing_metadata(config: dict, config_path: str | None) -> dict | None:
    """Relearn what this deployment's models are and can do.

    Four sources, weakest first, each recorded with its provenance so a later
    pass can tell a reading from a guess:

      inferred  the model's own name, via ``infer_reasoning_level``
      family    unanimous across the models sharing its family
      observed  the OpenRouter catalog, then the provider's own listing

    Capabilities are recorded per NORMALIZED MODEL rather than per routing
    target, so one fact covers every provider serving those weights — the same
    join that turns one benchmark score into a correctly ranked flagship tier.
    They are UNIONED across providers, never assigned: a gateway that omits a
    tag is silent, not authoritative, so a terse listing must not erase a richer
    one. Free status and rate limits stay per provider, because that is what
    they actually describe.

    Every derivation reads the RAW upstream id. ``normalize_model_id`` strips
    separators, so a regex run against its output reads digits that were never a
    parameter count: ``llama-3.1-8b`` becomes ``llama318b`` and infers "deep".

    Returns the state written, or None when nothing could be learned.
    """
    from .flagship import fetch_openrouter_profiles, normalize_model_id
    from .providers import infer_reasoning_level

    meta_cfg = routing_metadata_cfg(config)
    min_members = meta_cfg.get("min_family_members", _DEFAULT_MIN_FAMILY_MEMBERS)
    infer_tiers = meta_cfg.get("infer_reasoning", True)
    infer_family = meta_cfg.get("infer_family_capabilities", True)

    observed: dict[str, set[str]] = {}
    raw_for_key: dict[str, str] = {}

    # Keys this deployment actually serves. The catalog covers thousands of
    # models most deployments never touch: its capabilities are still wanted,
    # both as a base for the models we DO serve and as family evidence, but
    # writing a fact for every catalog entry would bloat the sidecar with
    # models that have no route and make "N models known" mean something other
    # than it reads as.
    local_keys: set[str] = set()

    def _observe(raw_id: str, caps: set[str]) -> None:
        key = normalize_model_id(raw_id)
        if not key:
            return
        local_keys.add(key)
        # A provider's own spelling wins over the catalog's for the same key,
        # so inference runs on the id this deployment actually calls.
        raw_for_key[key] = raw_id
        if caps:
            observed.setdefault(key, set()).update(caps)

    # Base layer: the catalog. Reuses the fetch the flagship refresh already
    # makes. Until now this read a `capabilities` key that fetch_openrouter_
    # profiles never set, so the whole layer was dead and models whose gateway
    # publishes a bare OpenAI object could never earn a tag.
    try:
        for key, profile in (fetch_openrouter_profiles() or {}).items():
            caps = set(profile.get("capabilities") or ())
            raw_for_key.setdefault(key, profile.get("model_id") or key)
            if caps:
                observed.setdefault(key, set()).update(caps)
        # Deliberately no local_keys update: the catalog says what a model can
        # do, not that this deployment has a route to it.
    except Exception as exc:  # noqa: BLE001 — a dead catalog degrades, never fails
        logger.warning("[routing-metadata] catalog fetch failed: %s", exc)

    # Overlay: each provider's own listing. Unioned with the catalog rather than
    # replacing it, and unioned across providers serving the same weights.
    for qualified, caps in _get_model_capability_snapshot().items():
        upstream = qualified.split("/", 1)[1] if "/" in qualified else qualified
        _observe(upstream, set(caps or ()))

    # Every distinct route, not just the ones carrying capabilities — a model
    # with no capability data is exactly the one that needs a tier inferred.
    routes: list[tuple[str, str]] = []
    try:
        for provider_name, upstream_id in _get_distinct_routes():
            routes.append((provider_name, upstream_id))
            _observe(upstream_id, set())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[routing-metadata] route enumeration failed: %s", exc)

    learned: dict[str, dict] = {}
    for key, caps in observed.items():
        if key in local_keys:
            learned[key] = {"capabilities": sorted(caps),
                            "capabilities_source": "observed"}

    # Family layer: lend a family's unanimous capabilities to a member that has
    # none of its own. Never to one that does — a reading always beats a guess.
    n_family = 0
    if infer_family:
        # Families are computed over EVERY observation, catalog included, so a
        # deployment serving three members of a family still benefits from what
        # the catalog knows about the other twenty.
        by_gen, by_bare = _family_capability_profiles(observed, raw_for_key, min_members)
        for key in local_keys:
            raw_id = raw_for_key.get(key) or key
            if observed.get(key):
                continue
            shared, _fam = _family_capabilities_for(key, raw_id, by_gen, by_bare)
            if shared:
                learned.setdefault(key, {}).update(
                    {"capabilities": sorted(shared), "capabilities_source": "family"})
                n_family += 1

    # Tier layer: inferred from the raw id. Weakest grade, so a curated or
    # migrated tier already in the sidecar survives untouched.
    n_tier = 0
    if infer_tiers:
        for key in local_keys:
            raw_id = raw_for_key.get(key) or key
            tier = infer_reasoning_level(raw_id)
            if tier:
                learned.setdefault(key, {}).update(
                    {"reasoning": tier, "reasoning_source": "inferred"})
                n_tier += 1

    if not learned:
        logger.warning(
            "[routing-metadata] learned nothing this pass; keeping the previous state"
        )
        return None

    # Everything above is network work, done before the lock is taken so a slow
    # catalog fetch cannot hold it against a cost observation.
    with _routing_sidecar_txn(config_path) as state:
        merged, counts = _merge_model_facts(state.get("by_model") or {}, learned)
        state["last_refresh_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        state["by_model"] = merged
        state["models_considered"] = len(routes)
        # by_provider and curated are left exactly as found: this pass relearns
        # nothing about free status, quota, or anything a person set by hand.
    logger.info(
        "[routing-metadata] %d model(s) served, %d in the sidecar: %d observed, "
        "%d by family, %d tiers inferred; %d fact(s) written, %d refused to a "
        "stronger source, %d carried forward, from %d route(s)",
        len(local_keys), len(merged), len(set(observed) & local_keys),
        n_family, n_tier, counts["written"], counts["refused"], counts["kept"],
        len(routes),
    )
    return state


_routing_metadata_inflight: bool = False
_routing_metadata_lock = threading.Lock()


def _maybe_fire_routing_metadata_refresh(
    config: dict, meta_cfg: dict, config_path: str | None
) -> None:
    """Relearn routing metadata in the background when its cadence is due."""
    if not _routing_metadata_due(meta_cfg, config_path):
        return
    global _routing_metadata_inflight
    with _routing_metadata_lock:
        if _routing_metadata_inflight:
            return
        _routing_metadata_inflight = True

    def _run() -> None:
        global _routing_metadata_inflight
        try:
            logger.info("[routing-metadata] refresh interval due — relearning")
            _recompute_routing_metadata(load_config(), config_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[routing-metadata] refresh failed: %s", exc)
        finally:
            with _routing_metadata_lock:
                _routing_metadata_inflight = False

    threading.Thread(target=_run, daemon=True, name="routing-metadata-refresh").start()


def _maybe_fire_flagship_refresh(
    config: dict, tier_cfg: dict, config_path: str | None
) -> None:
    """Recompute flagship membership in the background when its cadence is due.

    Modelled on _maybe_fire_free_models_update: the tier has to maintain itself
    without anyone running anything, since the whole point is that models enter
    as they ship and leave as the field moves past them.
    """
    if not _flagship_refresh_due(tier_cfg, config_path, config):
        return

    global _flagship_refresh_inflight
    with _flagship_refresh_lock:
        if _flagship_refresh_inflight:
            return
        _flagship_refresh_inflight = True

    def _run() -> None:
        global _flagship_refresh_inflight
        try:
            logger.info("[flagship] refresh interval due — recomputing membership")
            if _recompute_flagship_members(load_config(), config_path):
                with _models_list_cache_lock:
                    global _models_list_cache
                    _models_list_cache = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("[flagship] refresh failed: %s", exc)
        finally:
            with _flagship_refresh_lock:
                _flagship_refresh_inflight = False

    threading.Thread(target=_run, daemon=True, name="flagship-refresh").start()


def _maybe_fire_free_models_update(
    config: dict, free_tier: dict, config_path: str | None
) -> None:
    """Run the full free-models refresh in the background when its cadence is due.

    This is the scheduled sweep: it re-scrapes every default source, so new free
    models (including unsuffixed cloaked ones, which are detected by $0 pricing
    rather than by a ":free" suffix) are picked up, repriced models lose the free
    tag, and models withdrawn upstream are dropped. The cadence is
    free_tier.update_frequency_days, default 7. Sources run as part of the sweep
    and so cannot run more often than it does; the cost probe throttles itself
    further via free_tier.cost_probe.frequency_days, because it spends real
    quota, while the endpoint probe simply runs every sweep.
    """
    if not _free_update_due(free_tier, config_path):
        return

    global _free_update_inflight
    with _free_update_lock:
        if _free_update_inflight:
            return
        _free_update_inflight = True

    def _run() -> None:
        global _free_update_inflight
        try:
            logger.info("[free-update] refresh interval due — running free-models update")
            _run_free_models_update(load_config(), config_path)
            with _models_list_cache_lock:
                global _models_list_cache
                _models_list_cache = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("[free-update] failed: %s", exc)
        finally:
            with _free_update_lock:
                _free_update_inflight = False

    threading.Thread(target=_run, daemon=True, name="free-models-update-interval").start()


def _maybe_fire_cost_probe(
    config: dict, free_tier: dict, config_path: str | None
) -> None:
    cost_probe_cfg = free_tier.get("cost_probe", {})
    freq_days = cost_probe_cfg.get("frequency_days", 0)
    try:
        import os as _os
        import sys
        repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from llmproxy.config import load_cost_probe_state
        from scripts.update_free_models import _probe_due
    except Exception:  # noqa: BLE001
        return
    state = load_cost_probe_state(config_path)
    due, _ = _probe_due(state.get("last_probe_at"), freq_days)
    if not due:
        return

    global _cost_probe_inflight
    with _cost_probe_lock:
        if _cost_probe_inflight:
            return
        _cost_probe_inflight = True

    def _run() -> None:
        global _cost_probe_inflight
        try:
            logger.info("[cost-probe] interval due — running cost probe")
            _run_free_models_update(load_config(), config_path)
            with _models_list_cache_lock:
                global _models_list_cache
                _models_list_cache = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("[cost-probe] failed: %s", exc)
        finally:
            with _cost_probe_lock:
                _cost_probe_inflight = False

    threading.Thread(target=_run, daemon=True, name="cost-probe-interval").start()


def _maybe_fire_pr_if_due(config: dict, config_path: str | None) -> None:
    """Open a providers PR if providers_pr.frequency_days has elapsed since last PR."""
    pr_cfg = config.get("providers_pr", {})
    if pr_cfg.get("enabled") is not True:
        return
    freq_days = pr_cfg.get("frequency_days", 0)
    if not freq_days or freq_days <= 0:
        return  # no throttle configured — PR is opened immediately after updates
    try:
        import os as _os
        import sys
        repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from llmproxy.config import load_pr_state
        from llmproxy.providers import DATA_PATH as _DATA_PATH
        from scripts.update_free_models import _probe_due
    except Exception:  # noqa: BLE001
        return
    pr_state = load_pr_state(config_path)
    due, days_since = _probe_due(pr_state.get("last_pr_at"), freq_days)
    if not due:
        return
    # Read current sidecar to pass to _maybe_open_providers_pr.
    try:
        providers_text = _DATA_PATH.read_text(encoding="utf-8") if _DATA_PATH.exists() else None
    except Exception:  # noqa: BLE001
        providers_text = None
    if providers_text is None:
        return
    logger.info("[providers-pr] frequency_days interval elapsed — checking for PR")
    _maybe_open_providers_pr(config, providers_text)


_config_migration_done: bool = False
_config_migration_lock = threading.Lock()


# One lock for every read-modify-write of routing_metadata.json. Before this
# the refresh took none at all and _persist_cost_observed took a different one,
# so the two could interleave and lose an update; and neither took a FILE lock,
# so under gunicorn nothing serialised the workers against each other.
_routing_sidecar_write_lock = threading.Lock()


@contextlib.contextmanager
def _routing_sidecar_txn(config_path: str | None = None):
    """Exclusive read-modify-write of the sidecar, yielding the state to mutate.

    Saves and invalidates the read cache on a clean exit; an exception leaves
    the file untouched. Callers must do their NETWORK work before entering, so a
    slow catalog fetch cannot hold the lock against a cost observation.

    Mirrors ``admin._locked``: a thread lock for this process and an advisory
    file lock for the others, degrading to the thread lock alone where fcntl is
    unavailable rather than blocking every write.
    """
    with _routing_sidecar_write_lock:
        handle = None
        try:
            if fcntl is not None:
                lock_path = str(get_routing_metadata_path(config_path)) + ".lock"
                try:
                    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
                    handle = open(lock_path, "w")
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                except OSError as e:  # noqa: BLE001 — degrade, never block writes
                    print(f"[server:_routing_sidecar_txn] {e}")
                    traceback.print_exc()
                    if handle is not None:
                        handle.close()
                        handle = None
            state = load_routing_metadata(config_path)
            yield state
            save_routing_metadata(state, config_path)
            _reset_routing_sidecar_cache()
        finally:
            if handle is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()


def _curated_facts(state: dict) -> dict:
    """The sidecar's hand-set section, created if absent."""
    curated = state.setdefault("curated", {})
    if not isinstance(curated, dict):
        curated = state["curated"] = {}
    return curated


def _migrate_config_routing_keys(config_path: str | None = None) -> dict | None:
    """Move the five routing keys out of config.json into the sidecar.

    config.json stopped being a routing layer because it could not be a stable
    record of intent: it is the file a person hand edits, and machine processes
    were writing it too — the local-model sync tags every model a local provider
    serves and saves the file back. Anything already there is intent, though, so
    it is carried over rather than dropped, landing in the sidecar's `curated`
    section at the same precedence it had.

    The shape is preserved exactly, so the migrated data resolves through the
    same lookup it always did and moving it cannot change a routing decision.

    Existing curated entries WIN over the incoming config, so running this again
    after someone has edited a fact in the admin UI cannot resurrect the older
    config.json value over their correction.

    Backs up config.json first. Returns a report of what moved, or None when
    there was nothing to move.
    """
    config = load_config(config_path, force_reload=True)
    present = {k: config.get(k) for k in _ROUTING_CONFIG_KEYS if config.get(k)}
    if not present:
        return None

    try:
        cfg_file = get_config_path(config_path)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = cfg_file.with_name(f"{cfg_file.name}.backup-{stamp}")
        shutil.copy2(cfg_file, backup)
    except Exception as e:  # noqa: BLE001 — never migrate without a way back
        print(f"[server:_migrate_config_routing_keys] {e}")
        traceback.print_exc()
        logger.warning("[config-migration] could not back up %s — not migrating", config_path)
        return None

    report: dict[str, int] = {}
    with _routing_sidecar_txn(config_path) as state:
        curated = _curated_facts(state)
        for key, value in present.items():
            if key in _ROUTING_LIST_KEYS and isinstance(value, list):
                existing = curated.get(key)
                existing = list(existing) if isinstance(existing, list) else []
                seen = dict.fromkeys(e.lower() for e in existing if isinstance(e, str))
                added = 0
                for entry in value:
                    if isinstance(entry, str) and entry.lower() not in seen:
                        seen[entry.lower()] = None
                        added += 1
                curated[key] = list(seen)
                report[key] = added
            elif key in _ROUTING_DICT_KEYS and isinstance(value, dict):
                existing = curated.get(key)
                target = dict(existing) if isinstance(existing, dict) else {}
                added = 0
                for k, v in value.items():
                    if isinstance(k, str) and k.lower() not in target:
                        target[k.lower()] = v      # an existing curated fact wins
                        added += 1
                curated[key] = target
                report[key] = added

    for key in present:
        config.pop(key, None)
    save_config(config, config_path)

    logger.info(
        "[config-migration] moved %s from config.json into the sidecar's curated "
        "section (backup: %s)",
        ", ".join(f"{k}={n}" for k, n in sorted(report.items()) if n) or "nothing new",
        backup.name,
    )
    return report


def _run_startup_tasks_once(config_path: str | None = None) -> None:
    """Run the one-time per-worker startup tasks in a background daemon thread.

    The thread never blocks request handling and is guarded so it fires at most
    once per worker process. It:

      1. Warms the virtual-model route cache immediately, so GET /v1/models (and
         virtual-model routing) work from the very first request rather than only
         after a client has happened to hit /v1/models.
      2. Unless config['sync_believed_free_on_startup'] is false, reconciles the
         live config.json's free-tier sections from the bundled providers.json
         sidecar (no network, safe on a read-only sidecar).
      3. When config['update_believed_free_on_startup'] is true, additionally runs
         the full free-models updater (streaming its progress to the log), as
         long as free_tier.update_frequency_days has elapsed since the last run.
      4. Invalidates the cached /v1/models list after either step changes the
         config, so the synthetic 'free' virtual models are rebuilt from the
         updated believed_free data instead of the pre-update snapshot. (The route
         cache maps proxy id -> upstream model and does not depend on
         believed_free, so it does not need rebuilding here.)
    """
    global _startup_update_done
    with _startup_update_lock:
        if _startup_update_done:
            return
        _startup_update_done = True

    # Synchronous, and before the background work: until this has run, a
    # config.json written against the old layering is the only record of the
    # user's hand-set facts, and nothing reads it any more.
    global _config_migration_done
    with _config_migration_lock:
        if not _config_migration_done:
            _config_migration_done = True
            try:
                _migrate_config_routing_keys(config_path)
            except Exception as e:  # noqa: BLE001 — never fail startup over it
                print(f"[server:_run_startup_tasks_once] {e}")
                traceback.print_exc()

    def _run() -> None:
        # 1. Warm immediately so virtual models exist before the first request.
        logger.info("[startup] warming virtual-model route cache…")
        _warm_route_cache_if_empty()

        config = load_config()

        # 2. Lightweight live-config sync from the bundled sidecar (no network,
        #    safe on a read-only sidecar). On by default so the shipped/merged
        #    believed_free data reaches the live config.json every boot; opt out
        #    with sync_believed_free_on_startup: false.
        synced = False
        if config.get("free_tier", {}).get("sync_on_startup", True) is not False:
            synced = _sync_believed_free_from_sidecar(config_path)

        # 3. Optionally run the full network updater (refreshes + persists the
        #    sidecar, then syncs the live config from the freshly-scraped data).
        #    Throttled by update_frequency_days so a restart-heavy deployment
        #    re-scrapes on its configured cadence rather than on every boot.
        ran = False
        startup_free_tier = config.get("free_tier", {})
        if startup_free_tier.get("update_on_startup") is True:
            if _free_update_due(startup_free_tier, config_path):
                ran = _run_free_models_update(config, config_path)
            else:
                logger.info(
                    "[startup] free-models update not due yet "
                    "(free_tier.update_frequency_days); skipping"
                )

        # 4. Drop the cached /v1/models list so the synthetic 'free' set is rebuilt
        #    from the updated believed_free on the next request.
        if synced or ran:
            global _models_list_cache
            with _models_list_cache_lock:
                _models_list_cache = None
            logger.info("[startup] virtual-model list cache invalidated after update")

        # 5. Pre-build the full /v1/models response so the first external request is
        #    a cache HIT rather than a MISS that re-fetches every provider. Reloads
        #    config so it reflects any believed_free changes from steps 2–3. Runs
        #    per-worker; best-effort, never blocks or crashes the worker.
        try:
            warm_cfg = load_config()
            warm_providers = _enabled_providers(warm_cfg)
            if warm_providers:
                server_cfg = warm_cfg.get("server", {})
                models_ttl = server_cfg.get("models_cache_ttl", _DEFAULT_MODELS_CACHE_TTL)
                if models_ttl > 0:
                    logger.info("[startup] pre-building /v1/models response cache…")
                    _build_models_list(
                        warm_providers,
                        warm_cfg,
                        server_cfg.get("request_timeout", 120),
                        models_ttl,
                        only_if_empty=True,
                    )
        except Exception as exc:  # noqa: BLE001 — warming must never crash the worker
            logger.warning("[startup] /v1/models cache warm failed: %s", exc)

        # 6. Check frequency intervals for the free-models refresh, the
        #    flagship recompute, cost probe, and PR creation. Fires background
        #    threads for any that are due. The flagship pass runs here rather
        #    than earlier because it reads the route cache, which steps 1 and 5
        #    have just warmed — a fresh deployment would otherwise compute
        #    membership from an empty candidate pool.
        _maybe_fire_interval_probes(config_path)

    threading.Thread(target=_run, daemon=True, name="startup-tasks").start()


def _promote_sidecar_to_providers(
    providers_text: str, config_path: str | None = None
) -> tuple[str, dict]:
    """Fold what this deployment learned into the providers.json about to be PR'd.

    The sidecar is per deployment and never committed; providers.json is the
    shipped default every deployment inherits. Promotion is how one deployment's
    observations become everyone's starting point, which is the whole reason the
    PR flow exists.

    Facts are carried WITH their provenance so a reviewer can tell a reading
    from a guess. That matters more here than anywhere else: a wrong capability
    tag in providers.json makes every deployment route a request to a model that
    cannot serve it, where a wrong one in a local sidecar costs one deployment a
    retry.

    Only providers already present in providers.json are touched. A provider
    someone added locally is theirs, not a default for everyone, and inventing a
    catalog entry for it from one deployment's config would be a bigger claim
    than the data supports.

    Returns ``(text, report)``; the text is unchanged when nothing was promoted.
    """
    from .flagship import normalize_model_id
    from .providers import DEFAULT_FACT_SOURCE

    report: dict = {"providers": {}, "skipped_providers": [], "total": 0}
    try:
        data = json.loads(providers_text)
    except Exception as e:  # noqa: BLE001 — never break the PR over a parse
        print(f"[server:_promote_sidecar_to_providers] {e}")
        traceback.print_exc()
        return providers_text, report
    known = data.get("providers")
    if not isinstance(known, dict):
        return providers_text, report

    state = _load_routing_sidecar(config_path)
    by_model = state.get("by_model") or {}
    curated = state.get("curated") or {}
    by_provider = state.get("by_provider") or {}

    # Curated facts are keyed however the user keyed them; index them by the
    # same three forms the router resolves so they can be matched to a route.
    curated_caps = {k.lower(): v for k, v in (curated.get("model_capabilities") or {}).items()
                    if isinstance(k, str)}
    curated_tier = {k.lower(): v for k, v in (curated.get("model_reasoning") or {}).items()
                    if isinstance(k, str)}

    def _bump(provider: str, grade: str) -> None:
        report["providers"].setdefault(provider, {})
        report["providers"][provider][grade] = \
            report["providers"][provider].get(grade, 0) + 1
        report["total"] += 1

    changed = False
    for provider_name, upstream_id in _get_distinct_routes():
        if provider_name not in known:
            if provider_name not in report["skipped_providers"]:
                report["skipped_providers"].append(provider_name)
            continue
        entry = known[provider_name]
        if not isinstance(entry, dict):
            continue
        qualified = f"{provider_name}/{upstream_id}".lower()
        key = normalize_model_id(upstream_id)
        facts = by_model.get(key) if isinstance(by_model.get(key), dict) else {}

        caps = curated_caps.get(qualified) or curated_caps.get(upstream_id.lower())
        grade = "curated"
        if caps is None:
            caps = facts.get("capabilities")
            grade = facts.get("capabilities_source") or DEFAULT_FACT_SOURCE
        if isinstance(caps, list) and caps:
            target = entry.setdefault("model_capabilities", {})
            if target.get(qualified) != sorted(caps):
                target[qualified] = sorted(caps)
                changed = True
                _bump(provider_name, grade)

        tier = curated_tier.get(qualified) or curated_tier.get(upstream_id.lower())
        tgrade = "curated"
        if tier is None:
            tier = facts.get("reasoning")
            tgrade = facts.get("reasoning_source") or DEFAULT_FACT_SOURCE
        if isinstance(tier, str) and tier:
            target = entry.setdefault("model_reasoning", {})
            if target.get(qualified) != tier:
                target[qualified] = tier
                changed = True
                _bump(provider_name, tgrade)

    # Free status and quota belong to the provider, so they promote directly.
    for provider_name, info in by_provider.items():
        if provider_name not in known or not isinstance(info, dict):
            continue
        entry = known[provider_name]
        if not isinstance(entry, dict):
            continue
        free = [f"{provider_name}/{m}".lower() for m in info.get("believed_free") or []
                if isinstance(m, str)]
        if free:
            existing = entry.setdefault("believed_free", [])
            added = [m for m in free if m not in existing]
            if added:
                entry["believed_free"] = sorted(set(existing) | set(free))
                changed = True
                for _ in added:
                    _bump(provider_name, "observed")
        for model, limits in (info.get("free_limits") or {}).items():
            if not isinstance(model, str) or not isinstance(limits, dict):
                continue
            target = entry.setdefault("free_limits", {})
            qualified = f"{provider_name}/{model}".lower()
            if target.get(qualified) != limits:
                target[qualified] = limits
                changed = True
                _bump(provider_name, "observed")

    if not changed:
        return providers_text, report
    try:
        # dump_sidecar canonicalizes internally; canonicalize_sidecar mutates in
        # place and returns None, so nesting the two fed it None and silently
        # fell through to the un-canonicalized fallback below.
        from scripts.update_free_models import dump_sidecar
        return dump_sidecar(data), report
    except Exception as e:  # noqa: BLE001 — fall back to plain json
        print(f"[server:_promote_sidecar_to_providers] {e}")
        traceback.print_exc()
        return json.dumps(data, indent=2) + "\n", report


def _promotion_body(report: dict) -> str:
    """The PR body's provenance summary.

    The body was a fixed two sentences with no diff summary at all. Since
    inferred facts are promoted alongside observed ones, a reviewer needs to see
    at a glance which is which — that visibility is what makes promoting a guess
    reasonable rather than reckless.
    """
    if not report.get("total"):
        return ""
    order = ("curated", "observed", "family", "inferred")
    meaning = {
        "curated": "set by hand in the admin UI",
        "observed": "published by the provider or the OpenRouter catalog",
        "family": "unanimous across the model's family, not stated by the provider",
        "inferred": "derived from the model's name, not stated by the provider",
    }
    totals: dict[str, int] = {}
    for grades in report["providers"].values():
        for grade, n in grades.items():
            totals[grade] = totals.get(grade, 0) + n

    lines = ["", "### Where these facts came from", ""]
    lines.append(f"{report['total']} fact(s) promoted from a running deployment:")
    lines.append("")
    for grade in order:
        if totals.get(grade):
            lines.append(f"* **{grade}** — {totals[grade]}, {meaning[grade]}.")
    extra = sorted(set(totals) - set(order))
    for grade in extra:
        lines.append(f"* **{grade}** — {totals[grade]}.")
    lines.append("")
    lines.append("`family` and `inferred` entries are llmproxy's guesses rather than "
                 "anything a provider published. They are worth more scrutiny than the "
                 "rest, because a wrong capability tag here routes every deployment's "
                 "request to a model that cannot serve it.")
    lines.append("")
    lines.append("| provider | " + " | ".join(order) + " |")
    lines.append("|---|" + "---|" * len(order))
    for provider in sorted(report["providers"]):
        grades = report["providers"][provider]
        lines.append(f"| `{provider}` | "
                     + " | ".join(str(grades.get(g, 0)) for g in order) + " |")
    if report.get("skipped_providers"):
        lines.append("")
        lines.append("Not promoted, because they are not providers this repo ships: "
                     + ", ".join(f"`{p}`" for p in sorted(report["skipped_providers"]))
                     + ".")
    return "\n".join(lines)


def _maybe_open_providers_pr(config: dict, providers_text: str, example_text: str | None = None) -> None:
    """When config['providers_pr']['enabled'] is true, open a PR with the refreshed
    providers.json (+ config.example.json) against the configured base branch.

    *providers_text* / *example_text* are the computed file contents — passed in
    rather than read from disk, so a read-only deployment that couldn't persist
    the bundled copies can still open the PR from the in-memory/mirrored result.

    Uses the GitHub API directly (see github_pr.py) — it never touches the local
    git checkout. Requires a token (GITHUB_TOKEN / GH_TOKEN env, or
    config['providers_pr']['token'] which may be a ${VAR} ref) and the target repo
    as config['providers_pr']['repo'] = "owner/repo". Base branch defaults to
    "main" (config['providers_pr']['base']); branch name to "llmproxy-auto/providers"
    (config['providers_pr']['branch']). Best-effort: every missing prerequisite or
    API error is logged and skipped, never raised.
    """
    if config.get("providers_pr", {}).get("enabled") is not True:
        return

    # Throttle PR creation to at most once every providers_pr.frequency_days.
    pr_cfg = config.get("providers_pr", {})
    freq_days = pr_cfg.get("frequency_days", 0)
    if freq_days and freq_days > 0:
        try:
            import os as _os2
            import sys as _sys
            repo_root = _os2.path.dirname(_os2.path.dirname(_os2.path.abspath(__file__)))
            if repo_root not in _sys.path:
                _sys.path.insert(0, repo_root)
            from scripts.update_free_models import _probe_due

            from .config import load_pr_state
        except Exception:  # noqa: BLE001
            _probe_due = None  # type: ignore[assignment]
        if _probe_due is not None:
            pr_state = load_pr_state()
            due, days_since = _probe_due(pr_state.get("last_pr_at"), freq_days)
            if not due:
                logger.info(
                    "[providers-pr] throttled — %.1f day(s) since last PR "
                    "(frequency_days=%s); skipping.",
                    days_since, freq_days,
                )
                return

    import os

    from .github_pr import create_or_update_pr

    token = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
        or resolve_env_refs(config.get("providers_pr", {}).get("token"))
        or ""
    )
    if not token:
        logger.warning(
            "[providers-pr] providers_pr.enabled is on but no token found "
            "(set GITHUB_TOKEN / GH_TOKEN env or config['providers_pr']['token']); skipping PR."
        )
        return
    slug = config.get("providers_pr", {}).get("repo")
    if not (isinstance(slug, str) and "/" in slug):
        logger.warning(
            "[providers-pr] set config['providers_pr']['repo'] to \"owner/repo\" to open a PR; skipping."
        )
        return
    owner, repo = slug.split("/", 1)
    base = config.get("providers_pr", {}).get("base", "main")
    branch = config.get("providers_pr", {}).get("branch", "llmproxy-auto/providers")

    # Fold in what this deployment learned, so the PR carries observations and
    # not just whatever the scraper happened to see this run.
    promotion: dict = {}
    try:
        providers_text, promotion = _promote_sidecar_to_providers(providers_text)
        if promotion.get("total"):
            logger.info("[providers-pr] promoted %d learned fact(s) from the sidecar",
                        promotion["total"])
    except Exception as exc:  # noqa: BLE001 — a PR without promotion beats no PR
        logger.warning("[providers-pr] could not promote sidecar facts: %s", exc)

    files = {"llmproxy/providers.json": providers_text}
    if example_text is not None:
        files["config.example.json"] = example_text

    logger.info("[providers-pr] opening PR against %s/%s (%s)…", owner, repo, base)
    try:
        url = create_or_update_pr(
            token=token, owner=owner, repo=repo, base=base, branch=branch,
            files=files,
            title="chore: automated providers.json refresh (llmproxy)",
            body=(
                "Automated `providers.json` refresh opened by a running llmproxy "
                "deployment (`providers_pr.enabled`). Free-tier status is best-effort, "
                "so review the diff before merging."
                + _promotion_body(promotion)
            ),
            # config.example.json is derived from providers.json; only a real
            # providers.json change should open/refresh a PR. This stops a
            # regenerated-but-equivalent (or version-skewed) example from churning
            # a fresh PR on every startup.
            decisive_paths=["llmproxy/providers.json"],
        )
        if url:
            logger.info("[providers-pr] %s", url)
            try:
                from datetime import UTC as _UTC
                from datetime import datetime as _datetime

                from .config import save_pr_state as _save_pr_state2
                _save_pr_state2({"last_pr_at": _datetime.now(_UTC).isoformat()})
            except Exception as _exc:  # noqa: BLE001
                logger.warning("[providers-pr] could not save pr_state: %s", _exc)
    except Exception as exc:  # noqa: BLE001 — PR creation is best-effort
        logger.warning("[providers-pr] failed to open PR: %s", exc)


def _infer_reasoning_level(model_id: str) -> str:
    """
    Infer exploratory / standard / deep from a model id/name alone.

    Used both during startup sync for locally-served models and as a fallback
    "sophistication" signal for untagged models when ordering loadbalanced
    candidates (see _quality_key).
    """
    import re as _re
    s = model_id.lower()
    if any(p in s for p in ["qwq", "deepseek-r1", "deepseek-r2", "magistral",
                              ":r1", "-r1", "o1-", "o3-", "reasoning"]):
        return "deep"
    m = _re.search(r'(\d+(?:\.\d+)?)\s*b\b', s)
    if m:
        params = float(m.group(1))
        if params >= 100:
            return "deep"
        if params >= 15:
            return "standard"
        return "exploratory"
    if any(p in s for p in ["large", "medium", "mixtral", "70", "72", "32"]):
        return "standard"
    return "exploratory"


def _infer_local_reasoning_level(model_id: str) -> str:
    """Back-compat alias used by the startup sync for Ollama / OpenWebUI models."""
    return _infer_reasoning_level(model_id)


@app.route("/v1/models", methods=["GET"])
@app.route("/models", methods=["GET"])
def list_models() -> Response:
    """
    Aggregate model listings from all configured providers.

    Also served bare at ``/models``, for the same reason ``/version`` is: clients
    configured with a base URL that already ends in the API root probe it there,
    and without the rule Flask returns 404 — the ``/v1/<path>`` pass-through only
    covers ``/v1/*``. It costs nothing, since it is the same view.

    That one rule covers two cases, because ``_StripApiPrefix`` runs first: a
    client pointed at the bare root, and one pointed at ``/api``, whose
    ``/api/models`` is rewritten to ``/models`` before routing.

    Each provider is queried concurrently.  Providers that fail are logged as
    warnings and omitted rather than causing an overall failure.  The route
    cache is rebuilt atomically on each call so stale entries do not linger.

    Two synthetic virtual models are prepended when their backing candidates
    exist: 'free' (cycles through models whose ID contains 'free' or appears
    in config['believed_free']) and 'local' (cycles through models on localhost
    providers).

    Results are cached for models_cache_ttl seconds (default 60) to avoid
    redundant upstream fetches when clients issue multiple requests in quick
    succession (e.g. at startup).
    """
    _sync_local_provider_models_once()
    config = load_config()
    providers = _enabled_providers(config)
    server_cfg: dict = config.get("server", {})
    timeout: int = server_cfg.get("request_timeout", 120)
    models_ttl: int = server_cfg.get("models_cache_ttl", _DEFAULT_MODELS_CACHE_TTL)

    if not providers:
        # Re-read from disk before declaring the config empty. A stale in-process
        # cache (or a config still being written at startup) can momentarily yield
        # zero providers; force_reload confirms the on-disk truth so this warning
        # only ever fires when the config genuinely has no providers.
        config = load_config(force_reload=True)
        providers = _enabled_providers(config)
    if not providers:
        return jsonify({
            "object": "list",
            "data": [],
            "_warning": "No providers configured. Run 'llmproxy --setup'.",
        })

    # Return cached model list if still fresh. When the cache is present but stale,
    # serve it immediately and refresh in the background (stale-while-revalidate)
    # so only the very first request after a TTL window ever waits.
    if models_ttl > 0:
        with _models_list_cache_lock:
            cached = _models_list_cache
        if cached is not None:
            cached_data, cached_ts = cached
            age = time.monotonic() - cached_ts
            if age < models_ttl:
                logger.info("  [models cache] HIT (%.0fs old)", age)
                return jsonify({"object": "list", "data": cached_data})
            logger.info("  [models cache] STALE (%.0fs old) — serving stale, refreshing", age)
            _spawn_models_list_refresh()
            return jsonify({"object": "list", "data": cached_data})

    full_list = _build_models_list(providers, config, timeout, models_ttl)
    return jsonify({"object": "list", "data": full_list})


def _enabled_providers(config: dict) -> dict:
    """Return config['providers'] with reserved names stripped.

    Reserved names are removed before any presence check so a config that
    contains only reserved providers triggers the "no providers configured"
    warning rather than returning a silently empty model list.
    """
    return {
        k: v for k, v in config.get("providers", {}).items()
        if k not in RESERVED_PROVIDER_NAMES
    }


def _spawn_models_list_refresh() -> None:
    """Rebuild the cached /v1/models list in a daemon thread (at most one at a time).

    Used by the stale-while-revalidate path so an expired cache is refreshed off
    the request's critical path. Reloads config fresh so the rebuild reflects the
    latest on-disk providers/believed_free rather than a captured snapshot.
    """
    global _models_refresh_active
    with _models_refresh_lock:
        if _models_refresh_active:
            return
        _models_refresh_active = True

    def _run() -> None:
        global _models_refresh_active
        try:
            cfg = load_config()
            providers = _enabled_providers(cfg)
            if providers:
                server_cfg = cfg.get("server", {})
                _build_models_list(
                    providers,
                    cfg,
                    server_cfg.get("request_timeout", 120),
                    server_cfg.get("models_cache_ttl", _DEFAULT_MODELS_CACHE_TTL),
                )
        except Exception as exc:  # noqa: BLE001 — background refresh must never crash
            logger.warning("[models cache] background refresh failed: %s", exc)
        finally:
            with _models_refresh_lock:
                _models_refresh_active = False

    threading.Thread(target=_run, daemon=True, name="models-list-refresh").start()


def _build_models_list(providers: dict, config: dict, timeout: int, models_ttl: int,
                       only_if_empty: bool = False) -> list[dict]:
    """Build the full GET /v1/models data list and populate _models_list_cache.

    Aggregates each provider's models (rebuilding the route cache), prepends the
    synthetic virtual models whose backing candidates exist, annotates real models
    with classification fields, and rewrites virtual ids to display form. The
    result is cached (when models_ttl > 0) and returned. Called both on a cache
    miss in list_models() and from the startup warmup so the first external
    request is served from cache.

    ``only_if_empty`` is forwarded to _rebuild_route_cache: the startup warmup
    sets it so a route cache already populated by a concurrent request (or a test
    seed) is preserved rather than clobbered by the warm fetch.
    """
    global _models_list_cache

    all_models = _rebuild_route_cache(providers, timeout, only_if_empty=only_if_empty)

    # Prepend synthetic virtual models when their backing candidates exist.
    with _model_route_cache_lock:
        snapshot = dict(_model_route_cache)
    synthetic: list[dict] = []
    # The cost-tiered default: advertised whenever any virtual-eligible model
    # exists, since it spans the whole pool (free → local → paid).
    if any(
        _provider_exposes_to_virtual_models(cfg)
        for pn, _ in snapshot.values()
        if (cfg := get_provider(config, pn))
    ):
        synthetic.append({
            "id": "llmproxy__loadbalanced",
            "object": "model",
            "owned_by": "llmproxy",
            "name": "llmproxy__loadbalanced",
            "_note": "Virtual model: cost-tiered waterfall — prefers free-tier "
                     "(with session capacity), then local, then the cheapest "
                     "capable paid model, optimized per request. Fails over "
                     "silently to keep cost near zero.",
        })
    believed_free = _normalized_believed_free(config)
    has_free = any(
        "free" in uid.lower()
        or uid.lower() in believed_free
        or f"{pn}/{uid}".lower() in believed_free
        for pn, uid in snapshot.values()
    )
    if has_free:
        synthetic.append({
            "id": "llmproxy__free",
            "object": "model",
            "owned_by": "llmproxy",
            "name": "llmproxy__free",
            "_note": "Virtual model: cycles through all models whose ID contains 'free' (or appears in config['believed_free']) until one succeeds.",
        })
    if any(
        _is_local_url(provider_base_url(cfg))
        for pn, _ in snapshot.values()
        if (cfg := get_provider(config, pn))
    ):
        synthetic.append({
            "id": "llmproxy__local",
            "object": "model",
            "owned_by": "llmproxy",
            "name": "llmproxy__local",
            "_note": "Virtual model: cycles through all models served on localhost until one succeeds.",
        })

    for level in _REASONING_LEVELS:
        if _get_reasoning_model_candidates(level):
            synthetic.append({
                "id": f"llmproxy__{level}",
                "object": "model",
                "owned_by": "llmproxy",
                "name": f"llmproxy__{level}",
                "_note": f"Virtual model: cycles through all models tagged '{level}' reasoning until one succeeds.",
            })
        if _get_reasoning_free_candidates(level):
            synthetic.append({
                "id": f"llmproxy__{level}/free",
                "object": "model",
                "owned_by": "llmproxy",
                "name": f"llmproxy__{level}/free",
                "_note": f"Virtual model: cycles through free-tier models tagged '{level}' reasoning.",
            })
        if _get_reasoning_local_candidates(level):
            synthetic.append({
                "id": f"llmproxy__{level}/local",
                "object": "model",
                "owned_by": "llmproxy",
                "name": f"llmproxy__{level}/local",
                "_note": f"Virtual model: cycles through local models tagged '{level}' reasoning.",
            })

    for cap in _CAPABILITY_VIRTUALS:
        if _get_capability_model_candidates(cap):
            synthetic.append({
                "id": f"llmproxy__{cap}",
                "object": "model",
                "owned_by": "llmproxy",
                "name": f"llmproxy__{cap}",
                "_note": f"Virtual model: cycles through all models tagged '{cap}' in config['model_capabilities'], failing over until one succeeds.",
            })
        if _get_capability_free_candidates(cap):
            synthetic.append({
                "id": f"llmproxy__{cap}/free",
                "object": "model",
                "owned_by": "llmproxy",
                "name": f"llmproxy__{cap}/free",
                "_note": f"Virtual model: cycles through free-tier models tagged '{cap}' in config['model_capabilities'].",
            })

    # Fusion (multi-model deliberation) virtual models. Advertised when fusion is
    # enabled and at least MIN_PANEL eligible models back the variant: the full
    # non-local pool (or an explicit fusion.panel) for bare fusion, and the free
    # pool for fusion/free.
    fcfg = _fusion.get_fusion_config(config)
    if fcfg.get("enabled") is not False:
        if fcfg.get("panel"):
            bare_pool = _resolve_panel_list(fcfg["panel"], config)
        else:
            bare_pool = _get_all_model_candidates()
            if not fcfg.get("allow_paid", True):
                bare_pool = [c for c in bare_pool if _is_model_free(c[0], c[2], config)]
        if len(bare_pool) >= _fusion.MIN_PANEL:
            synthetic.append({
                "id": "llmproxy__fusion",
                "object": "model",
                "owned_by": "llmproxy",
                "name": "llmproxy__fusion",
                "_note": (
                    "Virtual model: fans the prompt out to a panel of models, a judge "
                    "compares their answers, and a synthesizer writes the final reply "
                    "(reported in the llmproxy_fusion field / X-LLMProxy-Fusion header)."
                ),
            })
        if len(_get_free_model_candidates()) >= _fusion.MIN_PANEL:
            synthetic.append({
                "id": "llmproxy__fusion/free",
                "object": "model",
                "owned_by": "llmproxy",
                "name": "llmproxy__fusion/free",
                "_note": (
                    "Virtual model: fusion deliberation drawn entirely from the "
                    "capacity-ordered free-tier pool (panel, judge, and synthesizer)."
                ),
            })

    # Per-provider virtual models: llmproxy__<provider> (cycles all of the
    # provider's models) and llmproxy__<provider>/<dimension>.  Advertised only
    # for enabled, non-local, virtual-exposing providers, and only when the
    # provider actually has a backing model for that dimension.  Ids that collide
    # with a global virtual name are skipped (global form takes precedence).
    for provider_name in sorted(providers):
        provider_cfg = providers[provider_name]
        if _is_local_url(provider_base_url(provider_cfg)):
            continue
        if not _provider_exposes_to_virtual_models(provider_cfg):
            continue
        for dim in ("",) + _PER_PROVIDER_DIMENSIONS:
            vid = f"llmproxy__{provider_name}" + (f"/{dim}" if dim else "")
            if vid in _VIRTUAL_MODELS:
                continue  # global virtual of the same name takes precedence
            if _get_provider_virtual_candidates(provider_name, dim):
                scope = "all" if dim == "" else f"'{dim}'"
                synthetic.append({
                    "id": vid,
                    "object": "model",
                    "owned_by": "llmproxy",
                    "name": vid,
                    "_note": (
                        f"Virtual model: cycles through {scope} of provider "
                        f"'{provider_name}'s models until one succeeds."
                    ),
                })

    # Annotate real models with (believed_free) and/or (local) suffixes in name,
    # and with OpenRouter-style supported_parameters so clients can classify them.
    cap_map = _model_capabilities(config)
    reasoning = _get_model_reasoning(config)
    for model in all_models:
        route = snapshot.get(model["id"])
        if not route:
            continue
        provider_name, upstream_id = route
        suffixes: list[str] = []
        uid_lower = upstream_id.lower()
        if (
            "free" in uid_lower
            or uid_lower in believed_free
            or f"{provider_name}/{uid_lower}" in believed_free
        ):
            suffixes.append("believed_free")
        provider_cfg = get_provider(config, provider_name)
        if provider_cfg and _is_local_url(provider_base_url(provider_cfg)):
            suffixes.append("local")
        if suffixes:
            model["name"] = model["name"] + " (" + ", ".join(suffixes) + ")"
        params = _supported_parameters(
            provider_name, upstream_id, config, cap_map=cap_map, reasoning=reasoning,
        )
        if params:
            model["supported_parameters"] = params

    # Enrich synthetic virtual models with the same classification fields so
    # clients can type them too (these are the entries clients most want to
    # classify, e.g. llmproxy__tools / llmproxy__vision).
    for vmodel in synthetic:
        tokens = set(re.split(r"[^a-z0-9]+", vmodel["id"].lower()))
        vmodel.setdefault("created", _SERVER_EPOCH)
        vmodel["architecture"] = _architecture_block(
            ["text", "image"] if "vision" in tokens else ["text"], ["text"],
        )
        vparams: list[str] = []
        if "tools" in tokens:
            vparams += ["tools", "tool_choice"]
        if tokens & set(_REASONING_LEVELS):
            vparams.append("reasoning")
        if vparams:
            vmodel["supported_parameters"] = vparams

    full_list = synthetic + all_models

    # Advertise virtual models in "llmproxy/model" slash form so opencode's
    # model picker shows a distinct, readable label for each virtual. opencode
    # groups entries by the segment before the first "/" — using "llmproxy/" as
    # the prefix puts all virtuals in one group with unique suffixes like
    # "deep__free", "loadbalanced", "free". Internal "/" in the model part
    # becomes "__" so the suffix is unambiguous (e.g. "llmproxy/deep__free").
    # The friendly `name` field is also set for clients that use it.
    # Internal state (route cache, frozensets) stays canonical; only the
    # outbound `id` and `name` fields are rewritten here.
    for model in full_list:
        if _is_virtual_model(model["id"]):
            model["name"] = _virtual_display_name(model["id"])
            model["id"] = _display_id(model["id"])

    if models_ttl > 0:
        with _models_list_cache_lock:
            # When called as a startup warmup (only_if_empty=True), skip the write
            # if something already populated the cache — a concurrent request or a
            # cross-test daemon thread from a previous test beat us here.
            if not only_if_empty or _models_list_cache is None:
                _models_list_cache = (full_list, time.monotonic())

    return full_list


@app.route("/v1/models/<path:model_id>", methods=["GET"])
@app.route("/models/<path:model_id>", methods=["GET"])
def get_model(model_id: str) -> Response:
    """
    Return metadata for a single proxy model ID.

    Aliased bare at ``/models/<id>`` alongside the listing, so a client that
    found a model through ``/models`` does not hit a 404 on the very next call.

    Accepts the display format returned by /v1/models ("provider__model"),
    two legacy display formats kept for backward compatibility
    ("model__provider" from PR #27 and "model (provider)" from before that),
    and the canonical slash format ("provider/upstream_model").  The route
    cache is checked first so display-format IDs resolve correctly without
    parsing.

    All virtual models (e.g. "llmproxy__free", "llmproxy__standard/local",
    plus the legacy "llmproxy/free" forms) are
    handled here via the _VIRTUAL_MODELS membership check.
    """
    model_id = _canonicalize_model_id(model_id, load_config())
    if _is_virtual_model(model_id):
        candidates = _get_virtual_candidates(model_id)
        # A flagship pool is walked in benchmark order, so report it in that
        # order too: _candidates is the documented way to inspect a pool, and a
        # list that did not match the actual failover sequence would be worse
        # than none. Other virtuals order per request (capacity, request fit,
        # rotation), so there is no single order to show for them.
        if _is_flagship_virtual_model(model_id):
            candidates = _flagship_ordered_candidates(
                candidates, _get_flagship_scores(),
                _get_normalized_free_limits(load_config()))
        return jsonify({
            "id": _display_id(model_id),
            "object": "model",
            "owned_by": "llmproxy",
            "name": _virtual_display_name(model_id),
            "_note": f"Virtual model: '{model_id}' cycles through matching candidates until one succeeds.",
            "_candidates": [f"{pn}/{um}" for pn, _, um in candidates],
        })

    # Prefer the route cache so clients can use the display ID they got from
    # /v1/models directly.  Fall back to parse_model_string for slash format.
    with _model_route_cache_lock:
        cached = _model_route_cache.get(model_id)
    if cached:
        provider_name, upstream_model = cached
    else:
        try:
            provider_name, upstream_model = parse_model_string(model_id)
        except ValueError as e:
            return _error(str(e), status=400)

    config = load_config()
    provider_cfg = get_provider(config, provider_name)
    if not provider_cfg:
        return _error(f"Unknown provider: '{provider_name}'", status=404)

    if not model_is_allowed(provider_cfg, upstream_model):
        return _error(
            f"Model '{upstream_model}' is not in the allowed list for provider '{provider_name}'.",
            status=404,
        )

    timeout = config.get("server", {}).get("request_timeout", 120)

    # Fetch this provider's models and merge new route entries into the cache.
    provider_models = _fetch_provider_models(provider_name, provider_cfg, timeout)
    new_routes: dict[str, tuple[str, str]] = {}
    for m in provider_models:
        if "_route" in m:
            r = m.pop("_route")
            new_routes[m["id"]] = r
            new_routes[_display_id(m["id"])] = r
    with _model_route_cache_lock:
        _model_route_cache.update(new_routes)

    # Match by proxy display ID or by upstream model ID (clients may use either).
    for m in provider_models:
        if m.get("id") == model_id or m.get("_upstream_id") == upstream_model:
            params = _supported_parameters(provider_name, upstream_model, config)
            if params:
                m["supported_parameters"] = params
            return jsonify(m)

    # The model passed the filter check but was not returned by the upstream
    # /models listing (e.g. a free-tier model that only appears after a
    # request).  Return a minimal valid object rather than a 404.
    fallback = {
        "id": model_id,
        "object": "model",
        "owned_by": provider_name,
        "_upstream_id": upstream_model,
        "_provider": provider_name,
        "_note": "Model not returned by upstream /models listing; filter check passed.",
    }
    return jsonify(fallback)


# ---------------------------------------------------------------------------
# Generic upstream proxy (non-streaming)
# ---------------------------------------------------------------------------

def _proxy_request(
    endpoint: str,
    provider_name: str,
    provider_cfg: dict,
    payload: dict,
    timeout: int,
    *,
    outbound=None,
    forwarded_headers: dict | None = None,
) -> Response:
    """
    Forward a non-streaming request to the upstream provider and return the
    response in the **canonical OpenAI** representation (status code, body,
    content-type).

    The provider's ``protocol`` selects an outbound dialect adapter that builds
    the native request and translates the native response back to canonical
    OpenAI form. For the default ``openai`` protocol the adapter is the identity,
    so the body is forwarded and returned verbatim — behavior is unchanged.

    Parameters
    ----------
    endpoint : str
        The API path suffix, e.g. 'chat/completions'.
    provider_name : str
        Used only for error message attribution.
    provider_cfg : dict
        Provider configuration (base_url, api_key, optional protocol).
    payload : dict
        Canonical OpenAI request body (with the upstream model ID already set).
    timeout : int
        Request timeout in seconds.
    outbound : OutboundAdapter, optional
        Override the adapter resolved from ``provider_cfg['protocol']``.
    forwarded_headers : dict, optional
        Pre-captured client headers to relay upstream. When omitted they are read
        from the active request via ``_forwarded_client_headers()``. Callers that
        dispatch off the request thread (e.g. the fusion panel fan-out on worker
        threads) must capture these on the request thread and pass them in, since
        Flask's ``request`` is not available — and not safe to reach for — there.
    """
    base_url = provider_base_url(provider_cfg)
    outbound = outbound or get_outbound(provider_cfg.get("protocol"))
    if forwarded_headers is None:
        forwarded_headers = _forwarded_client_headers()
    url, headers, body = outbound.build_request(
        endpoint, base_url, provider_cfg, payload,
        stream=False, forwarded_headers=forwarded_headers,
    )

    logger.info("  upstream POST %s  model=%s", url, payload.get("model", "?"))
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        logger.info("  upstream %d  %.0fms", resp.status_code, resp.elapsed.total_seconds() * 1000)
        if outbound.is_identity:
            content = resp.content
            content_type = resp.headers.get("Content-Type", "application/json")
        else:
            # Translate native success bodies to canonical OpenAI; leave upstream
            # error bodies (4xx/5xx) intact so the client sees the real diagnostic.
            content = (
                outbound.translate_response(resp.content)
                if 200 <= resp.status_code < 300 else resp.content
            )
            content_type = "application/json"
        out = Response(content, status=resp.status_code, content_type=content_type)
        # Preserve Retry-After on quota/rate-limit responses so the cycling loop
        # can cool the candidate for exactly as long as the upstream asks.
        if resp.status_code in _QUOTA_STATUSES:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                out.headers["Retry-After"] = retry_after
        return out
    except requests.exceptions.Timeout:
        return _mark_timeout(_error(
            f"Upstream provider '{provider_name}' timed out after {timeout}s.",
            status=504,
            code="timeout",
        ))
    except requests.exceptions.ConnectionError as e:
        # urllib3 re-raises a read timeout as a ConnectionError rather than a
        # ReadTimeout, so the string is the only thing that separates "the
        # upstream went quiet" from "the socket broke". Both are the candidate
        # failing to produce bytes, but only the former is a timeout for the
        # purposes of cooling it, so check before falling through to 502.
        if _looks_like_timeout(e):
            return _mark_timeout(_error(
                f"Upstream provider '{provider_name}' timed out after {timeout}s.",
                status=504,
                code="timeout",
            ))
        return _upstream_error(provider_name, e)
    except Exception as e:
        return _upstream_error(provider_name, e)


# ---------------------------------------------------------------------------
# Generic upstream proxy (streaming / SSE)
# ---------------------------------------------------------------------------

def _translated_stream_response(
    upstream_resp,
    outbound,
    inbound,
    provider_name: str,
    upstream_model: str,
    config: dict | None,
    prefix: bytes = b"",
    account_id: str | None = None,
) -> Response:
    """Pipe a *non-identity* upstream SSE stream through the dialect adapters.

    ``prefix`` carries any first chunk already pulled off the wire by a peek
    (see ``_peek_stream``); it is replayed ahead of the remaining stream so no
    bytes are lost.

    ``outbound.parse_stream`` turns the provider-native event stream into
    canonical OpenAI chunk dicts; usage is tee'd off those canonical chunks; then
    ``inbound.render_stream`` renders them into the client's dialect. The
    canonical-in-the-middle design means any inbound × upstream combination works.
    """
    @stream_with_context
    def generate(r=upstream_resp):
        captured: dict = {}
        try:
            with r:
                def raw():
                    if prefix:
                        yield prefix
                    for chunk in r.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk

                def teed(canon):
                    for c in canon:
                        if isinstance(c, dict) and c.get("usage"):
                            captured.update(c["usage"])
                        yield c

                yield from inbound.render_stream(teed(outbound.parse_stream(raw())))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.error("[server:_translated_stream] provider=%s timed out", provider_name)
            _demote_on_mid_stream_failure(provider_name, upstream_model, e, account_id)
            for frame in _mid_stream_frames(
                "Upstream stream timed out.", upstream_model, inbound,
            ):
                yield frame
        except Exception as e:  # noqa: BLE001
            logger.error("[server:_translated_stream] provider=%s: %s", provider_name, e)
            traceback.print_exc()
            _demote_on_mid_stream_failure(provider_name, upstream_model, e, account_id)
            msg = str(e).replace('"', "'")
            for frame in _mid_stream_frames(
                f"Upstream error: {msg}", upstream_model, inbound,
            ):
                yield frame
        finally:
            if config is not None and upstream_model and captured:
                _record_usage(provider_name, upstream_model, usage=captured,
                              config=config, account_id=account_id)

    return Response(generate(), content_type="text/event-stream")


# Terminal SSE frames for a stream that dies *after* llmproxy has committed to
# relaying it. Until now such a stream simply stopped: no ``finish_reason``, no
# ``[DONE]`` sentinel. A client cannot distinguish that from a slow upstream, so
# the OpenAI SDKs either hang until their own read timeout or raise a generic
# "stream ended unexpectedly", and an agent that was mid-tool-call is left
# holding a truncated ``arguments`` string it will try to parse. Emitting a real
# terminating sequence turns a corrupt stream into a cleanly short one.
#
# ``length`` is deliberate: it is the only standard finish_reason meaning "this
# turn was cut off", and clients already treat it as incomplete. ``stop`` would
# claim the turn finished normally, which is exactly the wrong thing to tell an
# agent holding half a tool call.
_STREAM_TRUNCATED_FINISH_REASON = "length"


def _stream_error_frames(message: str, model: str = "") -> list[bytes]:
    """SSE frames that terminate a committed stream after an upstream failure.

    Returns, in order: an ``error`` frame carrying *message*, a final
    chat-completion chunk whose one choice has an empty delta and a concrete
    ``finish_reason``, and the ``[DONE]`` sentinel.

    The finish chunk matters more than the error frame for agentic clients: it
    is what tells a delta accumulator that a partially-streamed ``tool_calls``
    argument string will receive no further fragments, so the client discards the
    incomplete call instead of invoking a tool with truncated JSON.
    """
    try:
        err = json.dumps({"error": {"message": message, "type": "upstream_error"}})
        final = json.dumps({
            "object": "chat.completion.chunk",
            "model": model or "",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": _STREAM_TRUNCATED_FINISH_REASON,
            }],
        })
        return [
            f"data: {err}\n\n".encode(),
            f"data: {final}\n\n".encode(),
            b"data: [DONE]\n\n",
        ]
    except Exception as e:  # noqa: BLE001
        # Serialization must never be what prevents a stream from terminating.
        print(f"[server:_stream_error_frames] {e}")
        traceback.print_exc()
        return [b'data: {"error":{"message":"Upstream error."}}\n\n', b"data: [DONE]\n\n"]


def _mid_stream_frames(message: str, model: str, inbound) -> list[bytes]:
    """Terminal frames for a committed stream, in the *client's* dialect.

    The full OpenAI terminating sequence (error frame, finish chunk, ``[DONE]``)
    is only meaningful to an OpenAI-dialect client. Injecting a
    ``chat.completion.chunk`` into a native Anthropic or Gemini event stream
    would hand the client a frame its parser has no case for, so a non-identity
    inbound gets the bare error frame it has always received — still an
    improvement over silence, without inventing events for a dialect we would be
    guessing at.
    """
    if getattr(inbound, "is_identity", False):
        return _stream_error_frames(message, model)
    try:
        err = json.dumps({"error": {"message": message, "type": "upstream_error"}})
        return [f"data: {err}\n\n".encode()]
    except Exception as e:  # noqa: BLE001
        print(f"[server:_mid_stream_frames] {e}")
        traceback.print_exc()
        return [b'data: {"error":{"message":"Upstream error."}}\n\n']


def _cool_on_timeout(
    exc: BaseException,
    label: str,
    provider_name: str,
    provider_cfg: dict | None,
    upstream_model: str,
    account_id: str | None = None,
) -> None:
    """Cool a candidate that timed out, exactly as a 429 would.

    A timeout and a rate limit say the same thing to the *next* request: this
    candidate is not answering right now. Without a cooldown a timing-out model
    keeps its place in the ordering and is picked first again, so every request
    pays the full timeout before failing over. Health does not cover this on its
    own — ``_health_score`` needs several samples before it moves at all, which
    is several more wasted timeouts.

    Only genuine timeouts are cooled. A connection reset or a DNS failure is a
    different fault and already fails over on its own; cooling it too would take
    a candidate out of rotation for a transient blip that cost nothing.
    """
    try:
        if not _looks_like_timeout(exc):
            return
        logger.warning(
            "  [%s] %s/%s timed out mid-stream (%s) — cooling it like a 429",
            label, provider_name, upstream_model, exc,
        )
        if provider_cfg is not None:
            _record_quota_saturation(provider_name, provider_cfg, upstream_model, None)
        else:
            # Post-commit teardown has no provider_cfg in scope. Cool the
            # candidate itself; the provider-wide circuit needs the config to
            # know whether a shared free allowance exists, and guessing wrong
            # there would take every model of the provider out of rotation.
            _mark_saturated(_usage_key(provider_name, upstream_model, account_id))
    except Exception as e:  # noqa: BLE001 — never break a teardown over accounting
        print(f"[server:_cool_on_timeout] {e}")
        traceback.print_exc()


def _demote_on_mid_stream_failure(
    provider_name: str,
    upstream_model: str,
    exc: BaseException,
    account_id: str | None = None,
    label: str = "stream",
) -> None:
    """Count a post-commit stream failure against the provider's health.

    A streamed candidate is recorded as a success the moment its stream survives
    the pre-commit check, because that is the last point at which the request can
    still fail over. Nothing ever revisited that verdict, so a provider that
    reliably accepts a request and then dies four fifths of the way through a
    generation kept a perfect health score and kept being ranked first — the one
    failure mode the ordering could not see.

    ``_is_upstream_failure`` does the discrimination that makes this safe: a
    client disconnect, a closed generator, or llmproxy's own stream teardown are
    not provider faults (see ``_CLIENT_ABORT_MARKERS``), so a user pressing
    Ctrl-C still cannot demote a healthy upstream.
    """
    try:
        if upstream_model and _is_upstream_failure(exc):
            _record_outcome(provider_name, upstream_model, False, account_id=account_id)
            # A stream that went quiet is the failure this cannot otherwise
            # reach. The request itself is already lost — bytes have shipped, so
            # there is nothing to fail over to — but cooling the candidate keeps
            # the NEXT request off it, which is the only repair available here.
            _cool_on_timeout(exc, label, provider_name, None, upstream_model, account_id)
    except Exception as e:  # noqa: BLE001
        # Health accounting must never be what breaks a stream's teardown.
        print(f"[server:_demote_on_mid_stream_failure] {e}")
        traceback.print_exc()


def _proxy_streaming(
    endpoint: str,
    provider_name: str,
    provider_cfg: dict,
    payload: dict,
    timeout: int,
    config: dict | None = None,
    *,
    outbound=None,
    inbound=None,
) -> Response:
    """
    Forward a streaming request to the upstream and relay the SSE stream back to
    the client.

    When the outbound (provider protocol) and inbound (client dialect) adapters
    are both identities — the common openai→openai case — the raw upstream bytes
    are relayed without buffering or parsing, exactly as before. Otherwise the
    stream is piped through the dialect adapters (canonical OpenAI in the middle).

    When *config* is provided, the request is counted up front and usage is
    recorded post-stream.
    """
    base_url = provider_base_url(provider_cfg)
    outbound = outbound or get_outbound(provider_cfg.get("protocol"))
    inbound = inbound or get_inbound("openai")
    url, headers, body = outbound.build_request(
        endpoint, base_url, provider_cfg, payload,
        stream=True, forwarded_headers=_forwarded_client_headers(),
    )
    upstream_model = payload.get("model", "")

    if config is not None and upstream_model:
        _record_usage(provider_name, upstream_model, usage=None, config=config)

    logger.info("  upstream POST %s  model=%s  [streaming]", url, payload.get("model", "?"))

    # Translation path: open eagerly so a pre-stream upstream error surfaces as a
    # normal response, then pipe through the adapters.
    if not (outbound.is_identity and inbound.is_identity):
        try:
            upstream_resp = requests.post(url, headers=headers, json=body, stream=True, timeout=timeout)
        except requests.exceptions.Timeout:
            return _error(f"Upstream provider '{provider_name}' timed out after {timeout}s.",
                          status=504, code="timeout")
        except Exception as e:  # noqa: BLE001
            return _upstream_error(provider_name, e)
        if upstream_resp.status_code >= 400:
            content = upstream_resp.content
            ct = upstream_resp.headers.get("Content-Type", "application/json")
            upstream_resp.close()
            return Response(content, status=upstream_resp.status_code, content_type=ct)
        return _translated_stream_response(
            upstream_resp, outbound, inbound, provider_name, upstream_model, config
        )

    # Identity fast path: raw passthrough. The upstream is opened *eagerly*, as
    # on the translation path above, so a pre-stream error (429, 5xx, auth) comes
    # back as a real response with its real status instead of being wrapped in a
    # 200 text/event-stream carrying the provider's JSON error as its only
    # "chunk". Clients (and fusion's streaming-degradation branch, which tests
    # status_code < 400) cannot recognize the latter as a failure at all.
    try:
        upstream_resp = requests.post(url, headers=headers, json=body, stream=True, timeout=timeout)
    except requests.exceptions.Timeout:
        return _error(f"Upstream provider '{provider_name}' timed out after {timeout}s.",
                      status=504, code="timeout")
    except Exception as e:  # noqa: BLE001
        return _upstream_error(provider_name, e)
    if upstream_resp.status_code >= 400:
        content = upstream_resp.content
        status = upstream_resp.status_code
        ct = upstream_resp.headers.get("Content-Type", "application/json")
        upstream_resp.close()
        logger.warning(
            "[server:_proxy_streaming] provider=%s -> %d before any stream bytes",
            provider_name, status,
        )
        return Response(content, status=status, content_type=ct)

    @stream_with_context
    def generate(r=upstream_resp):
        tail = bytearray()
        try:
            with r:
                first = True
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        if first:
                            logger.info(
                                "  upstream %d  first chunk: %s",
                                r.status_code,
                                chunk[:200],
                            )
                            first = False
                        yield chunk
                        tail += chunk
                        if len(tail) > _STREAM_TAIL_BYTES:
                            del tail[:-_STREAM_TAIL_BYTES]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.error(
                "[server:_proxy_streaming] provider=%s timed out mid-stream", provider_name
            )
            _demote_on_mid_stream_failure(provider_name, upstream_model, e)
            for frame in _stream_error_frames("Upstream stream timed out.", upstream_model):
                yield frame
        except Exception as e:
            logger.error(
                "[server:_proxy_streaming] provider=%s: %s", provider_name, e
            )
            traceback.print_exc()
            _demote_on_mid_stream_failure(provider_name, upstream_model, e)
            msg = str(e).replace('"', "'")
            for frame in _stream_error_frames(f"Upstream error: {msg}", upstream_model):
                yield frame
        finally:
            if config is not None and upstream_model:
                _record_stream_usage(provider_name, upstream_model, bytes(tail), config)

    return Response(generate(), content_type="text/event-stream")


# ---------------------------------------------------------------------------
# Capability detection — what a request needs and whether a response delivered
# ---------------------------------------------------------------------------
# llmproxy can route around models that don't support a requested capability.
# Each capability has up to three pure detectors:
#   request-detector  : does this request need the capability?
#   strict-detector   : was the capability *mandatory* (so a 200 that ignores it
#                       is a genuine failure worth failing over)?  May be None.
#   response-validator: did a non-streaming 200 actually deliver it?  May be None.
# Capabilities without a response-validator rely on the upstream returning an
# HTTP error (which already triggers virtual-model failover) — there is no
# reliable 200-body signal that e.g. a non-vision model silently ignored an image.


def _request_has_tools(payload: dict) -> bool:
    """True when the request carries a non-empty ``tools`` array."""
    tools = payload.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def _tool_use_forced(payload: dict) -> bool:
    """True when the request both provides tools and *forces* a tool call.

    Per the OpenAI spec a tool call is mandatory when ``tool_choice`` is the
    string ``"required"`` or an object selecting a specific function.  Under
    ``"auto"``/``"none"``/absent the model may legitimately answer without a
    tool call, so those are never treated as forced.
    """
    if not _request_has_tools(payload):
        return False
    tc = payload.get("tool_choice")
    if tc == "required":
        return True
    return isinstance(tc, dict) and tc.get("type") == "function"


def _response_has_tool_call(body_bytes: bytes) -> bool:
    """Whether a non-streaming chat completion body contains a tool/function call.

    Safe default is ``True`` (i.e. "can't confirm a failure"): malformed JSON or
    an unexpected shape must never trigger a spurious failover that discards a
    possibly-valid 200.  Returns ``False`` only when the body is well-formed and
    definitively has no tool call.
    """
    try:
        data = json.loads(body_bytes)
    except Exception:
        return True
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list):
        return True
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message") or ch.get("delta") or {}
        if isinstance(msg, dict) and (msg.get("tool_calls") or msg.get("function_call")):
            return True
    return False


def _request_has_image(payload: dict) -> bool:
    """True when any message includes an image content part (vision request)."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and (
                part.get("type") == "image_url" or "image_url" in part
            ):
                return True
    return False


def _request_wants_reasoning(payload: dict) -> bool:
    """True when the request asks for reasoning (``reasoning_effort``/``reasoning``)."""
    return payload.get("reasoning_effort") is not None or payload.get("reasoning") is not None


def _request_wants_json(payload: dict) -> bool:
    """True when the request forces a JSON response via ``response_format``."""
    rf = payload.get("response_format")
    return isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema")


def _response_is_json(body_bytes: bytes) -> bool:
    """Whether the assistant message content of a 200 parses as JSON.

    Safe default ``True`` on any uncertainty (malformed/odd shape) so we never
    fail over a response we can't actually prove is non-JSON.
    """
    try:
        data = json.loads(body_bytes)
    except Exception:
        return True
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return True
    first = choices[0]
    if not isinstance(first, dict):
        return True
    content = (first.get("message") or {}).get("content")
    if not isinstance(content, str):
        return True
    try:
        json.loads(content)
        return True
    except Exception:
        return False


# capability -> (request_detector, strict_detector | None, response_validator | None)
_CAPABILITIES: dict[str, tuple] = {
    "tools": (_request_has_tools, _tool_use_forced, _response_has_tool_call),
    "vision": (_request_has_image, None, None),
    "reasoning": (_request_wants_reasoning, None, None),
    "json": (_request_wants_json, _request_wants_json, _response_is_json),
}


def _model_capabilities(config: dict) -> dict[str, set[str]]:
    """Return config['model_capabilities'] as a lowercased map key -> set of caps.

    Defensive against user-edited config: missing/None/non-dict → {}, and any
    malformed entry is logged once and skipped rather than raising.
    """
    raw = _merged_routing_config(config).get("model_capabilities")
    if not isinstance(raw, dict):
        if raw is not None:
            logger.warning(
                "config['model_capabilities'] must be a dict; got %s — ignoring.",
                type(raw).__name__,
            )
        return {}
    result: dict[str, set[str]] = {}
    for key, val in raw.items():
        if not isinstance(key, str) or not isinstance(val, list):
            logger.warning(
                "config['model_capabilities']: invalid entry %r: %r — skipping.", key, val
            )
            continue
        caps = {c.lower() for c in val if isinstance(c, str) and c.lower() in _CAPABILITIES}
        result[key.lower()] = caps
    return result


def _model_fact_keys(provider_name: str, upstream_id: str) -> tuple[str, ...]:
    """The id forms a per-model fact may be filed under, most specific first.

    Qualified and bare are how an override or a provider listing keys things;
    normalized is how the LEARNED layer keys what belongs to the weights rather
    than to one provider, so a single entry answers for every spelling.
    """
    forms = [f"{provider_name}/{upstream_id}".lower(), upstream_id.lower()]
    try:
        from .flagship import normalize_model_id
        forms.append(normalize_model_id(upstream_id))
    except Exception as e:  # noqa: BLE001 — a lookup must never fail a request
        print(f"[server:_model_fact_keys] {e}")
        traceback.print_exc()
    return tuple(dict.fromkeys(f for f in forms if f))


def _is_variant_id(upstream_id: str) -> bool:
    """Whether this id names a billing/routing variant of another model.

    Thin wrapper so a lookup can never fail a request on an import or a regex:
    an unknown answer here simply means "treat it as an ordinary id", which is
    the pre-existing behaviour.
    """
    try:
        from .flagship import has_variant_suffix
        return has_variant_suffix(upstream_id)
    except Exception as e:  # noqa: BLE001 — a lookup must never fail a request
        print(f"[server:_is_variant_id] {e}")
        traceback.print_exc()
        return False


def _lookup_model_fact(mapping: dict, provider_name: str, upstream_id: str):
    """Find a per-model fact by qualified id, bare id, then normalized model.

    The first two are how a hand-written override or a provider listing keys
    things. The third is how the LEARNED layer keys what belongs to the weights
    rather than to a provider: one entry under ``glm53flash`` answers for
    ``z-ai/glm-5.3-flash``, ``zai/glm-5.3-flash``, ``zai-org/glm-5.3-flash`` and
    the bare spelling alike. Without this last form those entries would be
    written and never read, since none of them is a literal id anyone uses.

    Tried last, so an explicit entry for this exact model on this exact provider
    always beats a fact inherited from the same weights elsewhere.
    """
    if not mapping:
        return None
    for form in _model_fact_keys(provider_name, upstream_id):
        hit = mapping.get(form)
        if hit is not None:
            return hit
    return None


def _lookup_capabilities(
    cap_map: dict[str, set[str]], provider_name: str, upstream_id: str
) -> set[str]:
    """Every capability known for this model, across all three id forms.

    Unlike ``_lookup_model_fact``, which stops at the first hit, this UNIONS the
    qualified, bare and normalized entries. A capability set is evidence rather
    than a setting: the per-provider listing says what this gateway documents,
    the normalized entry says what these weights are known to do anywhere, and
    neither retracts the other.

    First-match-wins was wrong here in a way that actively hurt. A gateway
    publishing a partial ``supported_parameters`` shadowed the joined entry, so
    ``_capability_state`` returned KNOWN_INCAPABLE — ranking the model BELOW an
    untagged one for a capability it demonstrably has.

    ONE EXCEPTION, for ids naming a billing or routing variant. When the id
    carries a variant suffix (``:free``, ``:batch``, ...) and this map holds an
    exact entry for it, that entry is returned alone rather than unioned with
    the normalized key. A provider listing both ``z-ai/glm-5.2`` and
    ``z-ai/glm-5.2:free`` is discriminating between two routing targets, not
    being terse about one, so the union would hand the free variant its paid
    sibling's tool support — which is how a model that cannot call tools
    reached the flagship tier and then 404'd at request time. Every
    non-variant id keeps the union, so the regression described above stays
    prevented: terseness still cannot retract what another provider asserted.
    """
    if not cap_map:
        return set()
    if _is_variant_id(upstream_id):
        # Name the exact forms rather than slicing _model_fact_keys, whose
        # ordering and de-duplication are not this function's to depend on.
        for form in (f"{provider_name}/{upstream_id}".lower(), upstream_id.lower()):
            hit = cap_map.get(form)
            if hit:
                return set(hit)
    out: set[str] = set()
    for form in _model_fact_keys(provider_name, upstream_id):
        hit = cap_map.get(form)
        if hit:
            out |= set(hit)
    return out


def _model_has_capability(provider_name: str, upstream_id: str, cap: str, cap_map: dict[str, set[str]]) -> bool:
    """Whether this model is known to support *cap*, across all three id forms.

    A capability this target has been observed refusing is never reported, so a
    model that 404'd on a tool call drops out of ``llmproxy/tools`` as well as
    out of the ordering.
    """
    if cap in _learned_capability_gaps(provider_name, upstream_id):
        return False
    return cap in _lookup_capabilities(cap_map, provider_name, upstream_id)


def _needed_capabilities(payload: dict) -> set[str]:
    """The set of capabilities this request needs, per the request-detectors."""
    return {cap for cap, (detect, _s, _v) in _CAPABILITIES.items() if detect(payload)}


# Capability metadata is sparse: the shipped sidecar tags a minority of the
# believed-free pool, and the untagged remainder includes some of the strongest
# tool-callers available. Scoring "no entry" the same as "entry that omits this
# capability" would bury those models behind weaker tagged ones on exactly the
# requests they are best at, so the three states are kept distinct.
_CAP_KNOWN_CAPABLE = 1
_CAP_UNKNOWN = 0
_CAP_KNOWN_INCAPABLE = -1


def _capability_state(
    provider_name: str, upstream_id: str, cap: str, cap_map: dict[str, set[str]]
) -> int:
    """Three-valued capability lookup for one model and one capability.

    Returns ``_CAP_KNOWN_CAPABLE`` when the model is tagged with *cap*,
    ``_CAP_KNOWN_INCAPABLE`` when it carries capability metadata that omits
    *cap*, and ``_CAP_UNKNOWN`` when it carries no metadata at all.

    The middle case is the point. ``_model_has_capability`` collapses "unknown"
    and "known to lack it" into one False, which is correct for the places that
    need a yes/no answer (advertising ``supported_parameters``, building the
    explicitly capability-scoped ``llmproxy/<cap>`` pools) but wrong for
    *ordering*, where an untagged model deserves to sit between a confirmed
    match and a confirmed mismatch rather than tied with the mismatch.
    """
    # An observed refusal outranks every listing. A provider saying it supports
    # tools and then refusing to route one is not ambiguous evidence: the
    # request already failed against this exact target.
    if cap in _learned_capability_gaps(provider_name, upstream_id):
        return _CAP_KNOWN_INCAPABLE
    caps = _lookup_capabilities(cap_map, provider_name, upstream_id)
    if not caps:
        return _CAP_UNKNOWN
    return _CAP_KNOWN_CAPABLE if cap in caps else _CAP_KNOWN_INCAPABLE


def _drop_known_incapable(
    candidates: list[tuple[str, dict, str]],
    needed: set[str],
    cap_map: dict[str, set[str]],
) -> tuple[list[tuple[str, dict, str]], int]:
    """Remove candidates KNOWN to lack a needed capability. Never empties the pool.

    Ordering alone cannot express a hard requirement: ``_order_by_capability``
    only sorts, so a model that cannot call tools stays selectable and any later
    pass that re-sorts — affinity, favourites — can put it back in front. A
    model that positively cannot do the job should not be *selected* at all, in
    any virtual pool, free or otherwise.

    What makes that safe is the three-valued state. Only ``KNOWN_INCAPABLE`` is
    dropped: there is evidence the model lacks the capability. ``UNKNOWN`` is
    kept, because capability metadata is sparse and untagged models include some
    of the strongest tool-callers available — dropping those is what would turn
    a thin pool into a hard 503.

    Returns ``(survivors, dropped_count)``. When EVERY candidate is known
    incapable the pool is returned unchanged: a request that is attempted and
    fails over is strictly better than a 503 with no upstream call made, and a
    capability map can be wrong. A no-op when *needed* is empty.
    """
    if not needed or not candidates:
        return candidates, 0
    survivors = [
        c for c in candidates
        if not any(
            _capability_state(c[0], c[2], cap, cap_map) == _CAP_KNOWN_INCAPABLE
            for cap in needed
        )
    ]
    if not survivors:
        return candidates, 0
    return survivors, len(candidates) - len(survivors)


def _apply_capability_gate(
    candidates: list[tuple[str, dict, str]],
    needed: set[str],
    cap_map: dict[str, set[str]],
    label: str,
) -> tuple[list[tuple[str, dict, str]], int]:
    """Drop known-incapable candidates, then order the survivors capable-first.

    The two passes belong together at every call site, so they are wrapped here
    rather than repeated. Logs the drop: a pool that silently shrank is hard to
    debug, and this is the pass most likely to be blamed for a 503 it did not
    cause.
    """
    kept, dropped = _drop_known_incapable(candidates, needed, cap_map)
    if dropped:
        logger.info(
            "  [%s] capability gate dropped %d candidate(s) known to lack %s",
            label, dropped, "+".join(sorted(needed)),
        )
    return _order_by_capability(kept, needed, cap_map), dropped


def _order_by_capability(
    candidates: list[tuple[str, dict, str]],
    needed: set[str],
    cap_map: dict[str, set[str]],
) -> list[tuple[str, dict, str]]:
    """Stable-sort candidates so those satisfying the most needed caps come first.

    Ranks on the summed three-valued state across *needed* (see
    ``_capability_state``), so the order is: models tagged for every needed
    capability, then untagged models, then models whose metadata says they lack
    one.  Python's ``sorted`` is stable, so candidates on equal footing keep the
    order the earlier passes gave them.

    Never drops candidates — incomplete capability metadata must not turn a
    request into a hard 503.  A no-op when *needed* is empty.
    """
    if not needed:
        return candidates

    def satisfied(c: tuple[str, dict, str]) -> int:
        pn, _cfg, uid = c
        return sum(_capability_state(pn, uid, cap, cap_map) for cap in needed)

    return sorted(candidates, key=satisfied, reverse=True)


# ---------------------------------------------------------------------------
# Context-window fit
# ---------------------------------------------------------------------------
# Until now nothing in routing knew how big a model's context window was:
# ``context_length`` was normalized for the /v1/models listing and thrown away.
# An agentic conversation that outgrew a candidate got a 400
# ``context_length_exceeded``, which is non-transient, so it failed straight over
# to the next candidate — chosen with no regard for context, and so 400-ing too.
# A long session walked the whole pool and ended on the last 400 or a 503,
# precisely when the work was most valuable.

# The ~4-chars/token estimate runs low on agent traffic: code, JSON tool schemas
# and diffs tokenize nearer 3 chars/token, and the estimate ignores per-message
# role framing. Inflate before comparing against an advertised window.
_CONTEXT_SAFETY_FACTOR: float = 1.30
# Output tokens reserved when a request does not cap itself, so a model that
# "just barely fits" the prompt does not 400 on the completion. Sized for a
# tool-calling agent's turn (a call plus a short rationale), not for prose.
_CONTEXT_OUTPUT_RESERVE: int = 4096


def _get_model_context(config: dict) -> dict[str, int]:
    """Return ``config['model_context']`` as a lowercased map of id -> positive int.

    Keys take either form used elsewhere in the config: a bare upstream id
    (``"llama-3.3-70b"``) or a qualified one (``"groq/llama-3.3-70b"``).
    Defensive against a hand-edited config: a missing, None or non-dict value
    yields ``{}``, and a malformed entry is logged once and skipped rather than
    raising.
    """
    raw = config.get("model_context")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "config['model_context'] must be a dict; got %s — ignoring.",
            type(raw).__name__,
        )
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        window = _coerce_context_length(value)
        if window is None:
            logger.warning(
                "config['model_context'][%r] must be a positive integer; got %r — ignoring.",
                key, value,
            )
            continue
        out[key.lower()] = window
    return out


def _model_context_window(
    provider_name: str,
    upstream_id: str,
    ctx_map: dict[str, int],
    discovered: dict[str, int],
) -> int | None:
    """One model's context window: config override first, then discovery.

    Config wins deliberately. Several OpenAI-compatible gateways report their
    *output* cap in ``context_length``, so ``model_context`` exists exactly to
    correct an upstream that misreports. Returns None when nothing is known,
    which callers must treat as neutral.
    """
    bare = ctx_map.get(upstream_id.lower())
    if bare is not None:
        return bare
    qualified = f"{provider_name}/{upstream_id}".lower()
    if qualified in ctx_map:
        return ctx_map[qualified]
    return discovered.get(qualified)


def _estimate_context_tokens(payload: dict) -> int:
    """Rough token estimate for everything that occupies the context window.

    Builds on ``_estimate_payload_tokens`` (message text) and adds what an
    agentic request actually spends its window on: the serialized ``tools``
    array, assistant ``tool_calls`` arguments, and ``tool``-role result bodies.

    Deliberately separate from ``_estimate_payload_tokens`` rather than an
    extension of it: that function drives reasoning-tier triage and its
    thresholds are calibrated against message text alone, so widening it in
    place would silently re-tier every request in the proxy.
    """
    base = _estimate_payload_tokens(payload)
    try:
        extra = 0
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            extra += len(json.dumps(tools, separators=(",", ":")))
        for msg in payload.get("messages", []):
            if not isinstance(msg, dict):
                continue
            for call in msg.get("tool_calls") or []:
                if isinstance(call, dict):
                    extra += len(json.dumps(call, separators=(",", ":")))
        return base + extra // 4
    except Exception as e:  # noqa: BLE001 — a size hint must never fail a request
        print(f"[server:_estimate_context_tokens] {e}")
        traceback.print_exc()
        return base


def _required_context_tokens(payload: dict, config: dict | None = None) -> int:
    """Tokens this request needs a candidate's window to hold, with headroom.

    The comparison is conservative in one direction only, because the costs are
    asymmetric: overestimating demotes a model that would have fit, which costs
    one suboptimal but *working* pick, while underestimating relays a request
    that 400s and walks the pool. An asymmetric margin is the right answer to
    asymmetric costs.
    """
    factor = _config_float("context_safety_factor", _CONTEXT_SAFETY_FACTOR, config)
    reserve = _config_int("context_output_reserve", _CONTEXT_OUTPUT_RESERVE, config)
    budget = payload.get("max_completion_tokens") or payload.get("max_tokens")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        budget = 0
    return int(_estimate_context_tokens(payload) * max(1.0, factor)) + max(budget, reserve)


def _order_by_context_fit(
    candidates: list[tuple[str, dict, str]],
    payload: dict,
    ctx_map: dict[str, int],
    discovered: dict[str, int],
    config: dict | None = None,
) -> list[tuple[str, dict, str]]:
    """Stable-sort candidates so models *known* to be too small sort last.

    Structurally identical to ``_order_by_capability``: it reorders and never
    drops, because a model with a wrong or missing ``context_length`` must not be
    able to turn a request into a hard 503. Three-valued rather than a score — a
    candidate is demoted only when its window is *known* and smaller than the
    request needs, and an unknown window is neutral and keeps its incoming
    position.

    Returns *candidates* unchanged (the same object) when nothing is known to
    overflow, which is the common case. That identity is what lets the pass run
    last, after favorites, without disturbing any other pass's ranking.
    """
    if not candidates or len(candidates) < 2:
        return candidates
    needed = _required_context_tokens(payload, config)

    def too_small(c: tuple[str, dict, str]) -> bool:
        window = _model_context_window(c[0], c[2], ctx_map, discovered)
        return window is not None and window < needed

    if not any(too_small(c) for c in candidates):
        return candidates
    return sorted(candidates, key=too_small)


# Token thresholds for mapping a request's estimated input size to a reasoning
# tier when routing the GENERAL virtuals (llmproxy__free / llmproxy__local).
# Small, quick prompts prefer fast (exploratory) models; long or deliberately
# "thinking" requests prefer deep models. Configurable here rather than in JSON
# because they are routing heuristics, not per-deployment policy.
_TIER_SMALL_MAX_TOKENS: int = 1500
_TIER_MEDIUM_MAX_TOKENS: int = 8000


def _estimate_payload_tokens(payload: dict) -> int:
    """Rough token estimate (~4 chars/token) over a canonical request's text."""
    chars = 0
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            chars += sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
    return chars // 4


def _wants_thinking(payload: dict) -> bool:
    """True when the request explicitly asks for extra reasoning effort.

    Recognizes the OpenAI-style ``reasoning_effort`` ("medium"/"high") and a
    truthy ``reasoning`` field. A request that wants thinking is routed toward
    deep-tier models regardless of its size.
    """
    eff = payload.get("reasoning_effort")
    if isinstance(eff, str) and eff.lower() in ("medium", "high"):
        return True
    return bool(payload.get("reasoning"))


# Routing decision sources. The tier sources explain *which reasoning tier* a
# request was aimed at; the route sources explain *which ordering passes* moved
# the pool before a candidate was picked. Both are reported on the response so a
# pick can be explained after the fact instead of reverse-engineered from logs.
ROUTE_SOURCE_CAPACITY = "capacity"
ROUTE_SOURCE_LOADBALANCED = "loadbalanced"
ROUTE_SOURCE_CYCLING = "cycling"
ROUTE_SOURCE_FLAGSHIP_RANK = "flagship_rank"
ROUTE_SOURCE_REQUEST_FIT = "request_fit"
ROUTE_SOURCE_CAPABILITY = "capability"
ROUTE_SOURCE_FAVORITE = "favorite"
ROUTE_SOURCE_AFFINITY = "cache_affinity"
ROUTE_SOURCE_CONTEXT_FIT = "context_fit"
ROUTE_SOURCE_OVERSIZE = "oversize"
ROUTE_SOURCE_FAILOVER = "failover"

TIER_SOURCE_EXPLICIT = "explicit_reasoning_effort"
TIER_SOURCE_PROMPT_SIZE = "prompt_size"
TIER_SOURCE_TOOL_SIGNALS = "tool_signals"


def _tool_signal_routing_enabled(config: dict | None = None) -> bool:
    """Whether tool-result signals may adjust the prompt-size tier.

    Operators who want strictly size-based triage can set
    ``server.tool_signal_routing`` to false; the default is on.
    """
    try:
        cfg = config if config is not None else load_config()
        return bool(cfg.get("server", {}).get("tool_signal_routing", True))
    except Exception:  # noqa: BLE001 — routing must never fail on config shape
        return True


# — configuration accessors for the routing and streaming hardening knobs —
#
# All of these read through ``server.*`` with a default that reproduces the
# behavior llmproxy had before the knob existed, so an untouched config is
# bit-for-bit unchanged. They follow ``_tool_signal_routing_enabled`` above:
# tolerant of a hand-edited config, and never able to fail a request on a
# malformed value.

# Overall wall-clock budget for one virtual-model request's candidate walk.
# ``_VIRTUAL_CANDIDATE_TIMEOUT`` is per *candidate*, so a pool of N slow
# upstreams can currently burn N x 60s before the client sees anything. 0
# disables the deadline, which is the default and today's behavior.
_DEFAULT_CYCLE_DEADLINE_S: float = 0.0
# Below this much remaining budget, starting another candidate is pointless — a
# two-second window buys a TCP connect and nothing else — so the loop stops.
_MIN_CANDIDATE_TIMEOUT_S: float = 5.0
# Ceilings on the pre-commit buffer (see ``_open_stream_window``). Both are
# small: the window exists to see past a role preamble, not to buffer an answer.
_DEFAULT_PRECOMMIT_MAX_BYTES: int = 8192
_DEFAULT_PRECOMMIT_MAX_SECONDS: float = 2.0
# One patience setting for every virtual-model pool: how long a candidate may go
# without producing bytes before it is abandoned and cooled. 0 disables it, which
# is the default and today's behavior (the ordinary request/stream timeouts still
# apply). See ``_virtual_timeout``.
_DEFAULT_VIRTUAL_TIMEOUT_S: float = 0.0
# Per-request audit records. "off" is the default, matching every other knob
# here: an untouched config behaves exactly as it did before the knob existed.
_REQUEST_LOG_MODES: tuple[str, ...] = ("off", "metadata", "full")
_DEFAULT_REQUEST_LOG: str = "off"
# 0 = no cap, which is what "full" means. A cap is offered because a record
# holds the whole body in memory until the response completes, and a streamed
# answer has no size known in advance.
_DEFAULT_REQUEST_LOG_MAX_BODY: int = 0


def _config_float(key: str, default: float, config: dict | None = None) -> float:
    """Read ``server.<key>`` as a float, falling back to *default* on any problem."""
    try:
        cfg = config if config is not None else load_config()
        raw = cfg.get("server", {}).get(key, default)
        if isinstance(raw, bool) or raw is None:
            return default
        return float(raw)
    except Exception:  # noqa: BLE001 — routing must never fail on config shape
        return default


def _config_int(key: str, default: int, config: dict | None = None) -> int:
    """Read ``server.<key>`` as an int, falling back to *default* on any problem."""
    try:
        cfg = config if config is not None else load_config()
        raw = cfg.get("server", {}).get(key, default)
        if isinstance(raw, bool) or raw is None:
            return default
        return int(raw)
    except Exception:  # noqa: BLE001
        return default


def _config_bool(key: str, default: bool, config: dict | None = None) -> bool:
    """Read ``server.<key>`` as a bool, falling back to *default* on any problem."""
    try:
        cfg = config if config is not None else load_config()
        return bool(cfg.get("server", {}).get(key, default))
    except Exception:  # noqa: BLE001
        return default


def _request_log_mode(config: dict | None = None) -> str:
    """How much of each request to record: ``off``, ``metadata`` or ``full``.

    ``off`` (the default) emits nothing, leaving only the human ``→``/``←``
    lines. ``metadata`` emits one JSON object per request carrying everything
    llmproxy knows *about* the request — which model was asked for, which
    candidate answered, why it was ranked first, status, tokens, cost, timing —
    and no message content. ``full`` adds the request and response bodies.

    An unrecognised value reads as ``off`` rather than as an error: this decides
    whether user content is written down, so a typo must fail closed.
    """
    try:
        cfg = config if config is not None else load_config()
        raw = cfg.get("server", {}).get("request_log", _DEFAULT_REQUEST_LOG)
        if isinstance(raw, bool):
            # `true` is the obvious thing to write when you just want records.
            return "full" if raw else "off"
        mode = str(raw).strip().lower()
        return mode if mode in _REQUEST_LOG_MODES else _DEFAULT_REQUEST_LOG
    except Exception:  # noqa: BLE001 — auditing must never fail a request
        return _DEFAULT_REQUEST_LOG


def _virtual_timeout(config: dict | None = None) -> float | None:
    """Seconds of silence a virtual-model candidate is allowed, or None when off.

    This is the one patience setting that covers every virtual pool —
    ``llmproxy/free``, the reasoning tiers, flagship, loadbalanced, the
    per-provider slices — rather than a per-endpoint knob, because "how long am I
    willing to wait for an answer" is a property of the caller, not of the pool.

    It is an **idle** bound, not a total one: it limits how long a candidate may
    go without sending anything, at every stage of a request. Before a stream
    commits that means connect and first byte; after it commits it means the gap
    between chunks, which is the only bound that can catch a stream that starts
    normally and then stops. A long but steadily-producing generation is never
    cut off, however long it runs, which a total budget would get wrong.

    Default 0 (disabled), so an untouched config keeps the existing
    ``request_timeout``/``stream_timeout`` behavior exactly. A negative value is
    treated as disabled rather than as an instant timeout, since that is the
    harmless reading of a typo.
    """
    budget = _config_float("virtual_timeout_seconds", _DEFAULT_VIRTUAL_TIMEOUT_S, config)
    return budget if budget > 0 else None


def _cycle_deadline(config: dict | None = None) -> float | None:
    """Monotonic instant this request's candidate walk must stop by, or None.

    Returns ``time.monotonic() + budget`` so callers compare against a fixed
    instant instead of re-reading config inside the loop. A budget of zero or
    less disables the deadline entirely, which is the default.
    """
    budget = _config_float("cycle_deadline_seconds", _DEFAULT_CYCLE_DEADLINE_S, config)
    if budget <= 0:
        return None
    return time.monotonic() + budget


def _remaining_budget(deadline: float | None) -> float | None:
    """Seconds left before *deadline*, or None when no deadline is set."""
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _timeout_for_candidate(deadline: float | None, candidate_timeout: float) -> float | None:
    """Per-candidate timeout shrunk to the remaining budget.

    Returns *candidate_timeout* unchanged when no deadline is set, and None when
    the budget is gone or too small to be worth spending (see
    ``_MIN_CANDIDATE_TIMEOUT_S``) — which the cycling loops read as "stop here".
    """
    remaining = _remaining_budget(deadline)
    if remaining is None:
        return candidate_timeout
    if remaining < _MIN_CANDIDATE_TIMEOUT_S:
        return None
    return min(candidate_timeout, remaining)


def _free_tier_cache_affinity_enabled(config: dict | None = None) -> bool:
    """Whether prompt-cache affinity also pins *model* choice in free-tier pools.

    Off by default, and that default is deliberate rather than accidental: the
    free tier's ordering exists to spread load across quotas, and pinning a
    conversation to one model spends one model's allowance instead of the
    pool's.

    It is the wrong default for a single-user coding agent. There, consecutive
    turns landing on different models means different tool-calling conventions
    and different instruction-following inside one task, and the load being
    spread is one person's. Turning this on makes a conversation stick to one
    model for as long as that model keeps answering, and failover is unaffected
    because this only reorders.
    """
    return _config_bool("free_tier_cache_affinity", False, config)


def _precommit_window_enabled(config: dict | None = None) -> bool:
    """Whether the pre-commit window widens past the first non-empty chunk.

    Off by default, because turning it on can add up to
    ``stream_precommit_max_seconds`` to time-to-first-token for a healthy but
    slow upstream. On, llmproxy waits for a chunk that actually carries output
    before committing, which converts the common free-tier failure "accept the
    request, emit a role preamble, then die" from an unrecoverable mid-stream
    corruption into a clean, invisible failover.
    """
    return _config_bool("stream_commit_on_content", False, config)


def _precommit_max_seconds(deadline: float | None, config: dict | None = None) -> float:
    """Seconds the pre-commit window may buffer, clamped to the cycle budget.

    Without the clamp the two budgets could disagree: a request whose wall-clock
    deadline has nearly expired would still sit buffering for the full window.
    """
    configured = _config_float(
        "stream_precommit_max_seconds", _DEFAULT_PRECOMMIT_MAX_SECONDS, config
    )
    remaining = _remaining_budget(deadline)
    if remaining is None:
        return max(0.0, configured)
    return max(0.0, min(configured, remaining))


# Ceiling on the whole-response buffer used by ``server.stream_buffer_full``.
# Past this the response is committed and the remainder streamed incrementally:
# a reply this large is one the client wants to start seeing, and an unbounded
# buffer is a memory footgun on a threaded server.
_DEFAULT_STREAM_BUFFER_MAX_BYTES: int = 8 * 1024 * 1024


def _stream_buffer_full_enabled(config: dict | None = None) -> bool:
    """Whether a streamed virtual-model request is buffered whole before relay.

    Off by default. When on, ``_proxy_cycling_streaming`` reads each candidate's
    entire response before sending the client a byte, which is the only way to
    get genuine end-to-end failover on a streamed request: a provider that dies
    four fifths of the way through a generation can then be failed over exactly
    like one that returned a 500, because nothing has been committed yet.

    The cost is real and should be stated plainly rather than buried:
    time-to-first-token becomes time-to-*last*-token. For a tool-calling agent
    that cannot act on half a tool call anyway this is often the right trade; for
    an interactive chat UI it is not.
    """
    return _config_bool("stream_buffer_full", False, config)


def _target_reasoning_tier_explained(payload: dict) -> tuple[str, str]:
    """Choose the reasoning tier for a request and say why.

    Explicit thinking requests go to ``deep``; otherwise the estimated input
    size buckets the request into ``exploratory`` (small/fast), ``standard``, or
    ``deep`` (large).

    Prompt size alone misreads agentic traffic, where a short request can carry
    a long, failing task. So when the message list contains tool results, the
    size-derived tier is nudged one step by ``signals.tier_adjustment`` — up when
    the agent is erroring or spinning, down when it has just finished cleanly.
    The adjustment is bounded to a single step in either direction, which keeps
    it a correction to the size heuristic rather than a replacement for it.

    This drives the first-pick order for the general virtuals; failover still
    walks the rest of the candidates, so the choice is only a preference, never
    a restriction.
    """
    if _wants_thinking(payload):
        return "deep", TIER_SOURCE_EXPLICIT

    tokens = _estimate_payload_tokens(payload)
    if tokens <= _TIER_SMALL_MAX_TOKENS:
        base = "exploratory"
    elif tokens <= _TIER_MEDIUM_MAX_TOKENS:
        base = "standard"
    else:
        base = "deep"

    if not _tool_signal_routing_enabled():
        return base, TIER_SOURCE_PROMPT_SIZE

    try:
        delta, signal_source = tier_adjustment(payload)
    except Exception:  # noqa: BLE001 — a heuristic must never fail a request
        logger.debug("tool-signal tiering failed; falling back to prompt size", exc_info=True)
        return base, TIER_SOURCE_PROMPT_SIZE

    if not delta or signal_source == SOURCE_NEUTRAL:
        return base, TIER_SOURCE_PROMPT_SIZE

    # Clamp to the strongest *inferable* tier, not to the end of the tuple.
    # Overlay tiers such as flagship sit above deep and are opt-in by name only
    # (llmproxy/flagship), so no prompt size or tool signal may drift into them.
    idx = _REASONING_LEVELS.index(base)
    shifted = min(_MAX_INFERRED_LEVEL_INDEX, max(0, idx + delta))
    if shifted == idx:
        return base, TIER_SOURCE_PROMPT_SIZE
    return _REASONING_LEVELS[shifted], f"{TIER_SOURCE_TOOL_SIGNALS}:{signal_source}"


def _target_reasoning_tier(payload: dict) -> str:
    """Reasoning tier for *payload* — see ``_target_reasoning_tier_explained``."""
    return _target_reasoning_tier_explained(payload)[0]


def _order_by_request_fit(
    candidates: list[tuple[str, dict, str]],
    payload: dict,
    reasoning_map: dict[str, str],
) -> list[tuple[str, dict, str]]:
    """Stable-sort *candidates* so the best fit for *payload* comes first.

    Triages within a single tier (free or local): the candidate pool is already
    constrained to its tier by the selector, and this only **reorders** it — it
    never adds, drops, or substitutes a candidate, so failover behavior and tier
    containment are preserved. Sorts by a two-part key:

    1. **tier distance** — distance from the candidate's ``model_reasoning`` tier
       to the request's target tier along exploratory(0) < standard(1) < deep(2):
       an exact match sorts first, adjacent next, far last; untagged models sort
       neutral (1.5) so incomplete metadata never buries a usable model.
    2. **size fit** — within an equal-tier band, prefer the right-*sized* model
       for the job: a deep/thinking request prefers the **largest** model, a small
       (exploratory) request prefers the **smallest**, and a standard request is
       neutral (base order preserved). This is what lets even a constrained
       sub-virtual like ``deep/free`` pick the right-sized model from what's
       available.

    Stable, so the base ordering (capacity headroom for /free, random rotation
    for /local) is preserved within each fit band. A no-op on an empty pool.
    """
    if not candidates:
        return candidates
    target_tier = _target_reasoning_tier(payload)
    order = {lvl: i for i, lvl in enumerate(_REASONING_LEVELS)}
    target_idx = order.get(target_tier, 1)
    # Within an equal-tier band, bias toward the size the request warrants:
    # +1 prefer larger params, -1 prefer smaller, 0 neutral (keep base order).
    if target_idx >= order.get("deep", 2) or _wants_thinking(payload):
        size_pref = 1
    elif target_idx <= order.get("exploratory", 0):
        size_pref = -1
    else:
        size_pref = 0

    def rank(c: tuple[str, dict, str]) -> tuple[float, float]:
        pn, _cfg, uid = c
        lvl = reasoning_map.get(uid.lower()) or reasoning_map.get(f"{pn}/{uid}".lower())
        if lvl is None or lvl not in order:
            tier_d = 1.5  # untagged: neutral — after exact/adjacent, before far
        else:
            tier_d = float(abs(order[lvl] - target_idx))
        # -size_pref so prefer-large (+1) sorts bigger params first and
        # prefer-small (-1) sorts smaller params first; neutral (0) is a no-op.
        size_d = -size_pref * _param_count(uid)
        return (tier_d, size_d)

    return sorted(candidates, key=rank)


def _demote_oversize_candidates(
    candidates: list[tuple[str, dict, str]],
    payload: dict,
) -> tuple[list[tuple[str, dict, str]], int]:
    """Move candidates that have refused a request this large to the back.

    Returns ``(ordered, demoted_count)``. Demote rather than drop, matching the
    discipline the saturation path already keeps: a pool where every candidate
    has 413'd still serves, and the client gets the upstream's real 413 instead
    of a 503 this proxy invented. It is also what makes the watermark
    self-correcting, since a demoted candidate reached as a last resort can
    still prove the limit wrong by succeeding.

    Stable within each group, so every ordering decision made before this one —
    benchmark rank, capacity, request fit — survives among the candidates that
    are still plausible.

    Costs nothing until a 413 has actually happened: with an empty registry it
    returns immediately, without serializing the payload.
    """
    if not candidates:
        return candidates, 0
    with _oversize_lock:
        if not _oversize_registry:
            return candidates, 0
    size = _payload_size_bytes(payload)
    if size <= 0:
        return candidates, 0
    fits = [c for c in candidates if not _is_oversize_for(c[0], c[2], size)]
    too_big = [c for c in candidates if _is_oversize_for(c[0], c[2], size)]
    if not too_big:
        return candidates, 0
    return fits + too_big, len(too_big)


def _capability_failed(payload: dict, body_bytes: bytes) -> bool:
    """True when a non-streaming 200 failed to deliver a *forced* capability.

    Only capabilities whose strict-detector fires and that have a response
    validator can trigger this (today: tools, json).  Capabilities without a
    validator (vision, reasoning) rely on HTTP-error failover instead.
    """
    for _cap, (_detect, strict, validate) in _CAPABILITIES.items():
        if strict is None or validate is None:
            continue
        if strict(payload) and not validate(body_bytes):
            return True
    return False


def _is_transient_status(status: int) -> bool:
    """True for statuses worth retrying on the *same* candidate.

    HTTP 429 (rate limited) and any 5xx are transient — including the 502/504
    that ``_proxy_request`` synthesizes for connection errors and timeouts.
    Other 4xx (bad request, auth, not-found) won't improve on retry, so the
    cycling loop fails straight over to the next candidate instead.
    """
    return status == 429 or status >= 500


def _choice_yields_output(choice: dict) -> bool:
    """True when a completion choice carries something usable for the client.

    "Usable" means visible text content, a tool/function call, or a refusal —
    anything the caller can act on.  A choice whose only signal is reasoning
    ("thinking") tokens with empty content is *not* usable: the caller asked for
    an answer, not the model's scratch work.  This is what lets the waterfall
    fail over a model that spends a small ``max_tokens`` budget entirely on
    reasoning and returns an empty final message, so a lighter model that can
    answer inside the budget serves the request instead.
    """
    if not isinstance(choice, dict):
        return False
    message = choice.get("message")
    if not isinstance(message, dict):
        # A non-dict, non-null message is an unexpected shape; treat it as usable
        # rather than risk dropping a real answer we simply don't recognize.
        return message is not None
    if message.get("tool_calls") or message.get("function_call"):
        return True
    if message.get("refusal"):
        return True
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        # Multimodal content parts: usable if any part has non-empty text.
        for part in content:
            if isinstance(part, str) and part.strip():
                return True
            if isinstance(part, dict) and (part.get("text") or "").strip():
                return True
        return False
    # content is None or an unexpected type — no visible text.
    return False


def _is_budget_truncated_empty(body_bytes: bytes) -> bool:
    """True when a 200 delivered no visible output *only* because it ran out of
    token budget: every choice is empty (no text, no tool call) and at least one
    was cut off with ``finish_reason`` "length".

    This is the signature of a reasoning model that spent its whole ``max_tokens``
    budget on thinking and had nothing left for the answer. Retrying the same
    model with a larger budget can recover a real reply, so the cycling loop
    escalates the budget before failing over (see ``_escalate_budget_if_starved``).
    A body that already carries usable output, or that was truncated for any other
    reason, is left alone.
    """
    try:
        data = json.loads(body_bytes)
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        return False
    saw_length = False
    for choice in choices:
        if _choice_yields_output(choice):
            return False
        finish = choice.get("finish_reason") if isinstance(choice, dict) else None
        if finish in ("length", "max_tokens"):
            saw_length = True
    return saw_length


def _budget_escalation_enabled(config: dict | None = None) -> bool:
    """Whether a budget-starved 200 is retried with more room. Default on."""
    return _config_bool("budget_escalation", True, config)


def _budget_bump_factor(config: dict | None = None) -> int:
    """Multiplier applied to the token budget on each retry."""
    return _config_int("budget_escalation_factor", _BUDGET_BUMP_FACTOR, config)


def _budget_bump_ceiling(config: dict | None = None) -> int:
    """Hard upper bound on an escalated budget, before the per-model clamp."""
    return _config_int("budget_escalation_ceiling", _BUDGET_BUMP_CEILING, config)


def _budget_bump_max_retries(config: dict | None = None) -> int:
    """How many times one candidate may be retried with a larger budget."""
    return _config_int("budget_escalation_max_retries",
                       _BUDGET_BUMP_MAX_RETRIES, config)


def _budget_ceiling_for(provider_name: str | None, upstream_id: str | None,
                        config: dict | None = None) -> int:
    """The escalation ceiling for one model: configured, clamped to its window.

    The configured ceiling alone was safe while it was 4096, which is below
    almost every model's output cap. It is 65535 now, which is above what most
    models will accept, and asking for more than a model can give comes back as
    a 400 — degrading safely, since 4xx walks to the next candidate, but
    spending a round trip and losing the very answer the escalation exists to
    recover.

    So a KNOWN context window caps the bump. An unknown one is neutral and the
    configured ceiling stands, matching how ``_order_by_context_fit`` treats a
    missing ``context_length``: absent metadata must never make things worse.

    The window is the whole context, prompt included, so capping the OUTPUT
    budget at it is deliberately loose — it is a backstop against asking for
    obvious nonsense, not a context-fit calculation. ``_order_by_context_fit``
    is what actually reasons about fit.
    """
    ceiling = _budget_bump_ceiling(config)
    if not provider_name or not upstream_id:
        return ceiling
    try:
        cfg = config if config is not None else load_config()
        window = _model_context_window(
            provider_name, upstream_id,
            _get_model_context(cfg), _get_model_context_snapshot(),
        )
    except Exception as e:  # noqa: BLE001 — a clamp must never fail a retry
        print(f"[server:_budget_ceiling_for] {e}")
        traceback.print_exc()
        return ceiling
    if window and window > 0:
        return min(ceiling, window)
    return ceiling


def _bumped_budget(payload: dict, provider_name: str | None = None,
                   config: dict | None = None) -> dict | None:
    """Return a copy of *payload* with its token budget multiplied, or ``None``.

    Recognizes the OpenAI ``max_completion_tokens`` and legacy ``max_tokens``
    fields. Returns ``None`` when neither is set to a positive int (nothing to
    bump — an uncapped request would never truncate) or the budget is already at
    the ceiling (further bumps refused so cost stays bounded).

    The ceiling comes from ``server.budget_escalation_ceiling``, clamped to the
    model's known context window — see ``_budget_ceiling_for``. ``provider_name``
    is optional so existing callers and tests keep working unclamped.
    """
    ceiling = _budget_ceiling_for(provider_name, payload.get("model"), config)
    factor = _budget_bump_factor(config)
    for field in ("max_completion_tokens", "max_tokens"):
        current = payload.get(field)
        if isinstance(current, int) and not isinstance(current, bool) and current > 0:
            if current >= ceiling:
                return None
            bumped = min(current * factor, ceiling)
            # Covers a factor of 1 or less, which would otherwise spin through
            # every retry making no progress.
            if bumped <= current:
                return None
            return {**payload, field: bumped}
    return None


def _response_unusable(body_bytes: bytes) -> bool:
    """True when a non-streaming HTTP 200 isn't actually a usable completion.

    Some upstreams answer ``200 OK`` while the body carries an error object, an
    empty result (no ``choices``), or a choice with no visible output.  Treating
    these as failures lets the cycling loop fail over instead of handing the
    client a dead response.

    A body is usable when at least one choice yields output — visible text, a
    tool/function call, or a refusal (see ``_choice_yields_output``).  An empty
    ``content`` string alone is *not* usable unless it is paired with a tool
    call; this is what a reasoning model returns when a tight ``max_tokens``
    budget is consumed entirely by thinking, and failing over lets a lighter
    candidate answer.  A body that isn't JSON at all is treated as unusable,
    since every cycled endpoint speaks JSON chat/completions.
    """
    try:
        data = json.loads(body_bytes)
    except (ValueError, TypeError):
        return True
    if not isinstance(data, dict):
        return True
    if data.get("error"):
        return True
    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        return True
    if not any(_choice_yields_output(c) for c in choices):
        return True
    return False


def _peek_stream(resp) -> tuple[bytes | None, bytes, "Iterator[bytes]"]:
    """Read the first non-empty chunk of a streamed response without losing it.

    Returns ``(error_body, prefix, rest)`` where:

    * ``error_body`` is the upstream bytes when the opening of the stream is an
      SSE error event (a ``data:`` payload whose JSON carries an ``error``),
      else ``None``.
    * ``prefix`` is the first non-empty chunk already pulled off the wire.
    * ``rest`` is an iterator over the remaining chunks.

    Streaming the buffered ``prefix`` first means the first token is never
    dropped, while the peek lets the caller fail over when a provider returns
    ``200`` and then immediately errors inside the stream.
    """
    chunks = resp.iter_content(chunk_size=None)
    prefix = b""
    for chunk in chunks:
        if chunk:
            prefix = chunk
            break
    error_body = prefix if (prefix and _sse_prefix_is_error(prefix)) else None
    return error_body, prefix, chunks


def _delta_yields_output(delta) -> bool:
    """True when a streamed delta carries something the client can act on.

    The streaming analogue of ``_choice_yields_output``: visible text, a
    tool-call or function-call fragment, or a refusal. A role-only preamble
    (``{"role": "assistant"}``) and a reasoning-only delta are deliberately NOT
    output — they are exactly what a first-chunk peek mistakes for a working
    generation, which is the whole reason the window below exists.
    """
    if not isinstance(delta, dict):
        return False
    if delta.get("tool_calls") or delta.get("function_call"):
        return True
    if delta.get("refusal"):
        return True
    content = delta.get("content")
    if isinstance(content, str):
        return bool(content)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str) and part:
                return True
            if isinstance(part, dict) and (part.get("text") or ""):
                return True
    return False


def _sse_window_has_output(buffered: list[bytes], outbound) -> bool:
    """True when the buffered SSE bytes already contain an output-bearing delta.

    Parses with the provider's own ``outbound.parse_stream``, so this is correct
    for native Anthropic and Gemini event streams as well as the OpenAI case
    (whose adapter inherits the canonical ``data: {json}`` parser).

    The parse runs over a throwaway iterator built from a copy of the buffer and
    its result is discarded; on commit the caller replays the buffer from the
    start through a *fresh* ``parse_stream``, so no adapter state machine is ever
    left half-advanced. Do not "optimize" that replay away.

    Defensive by design: any parse failure reads as "no output yet" rather than
    as a stream failure, so a dialect quirk can only cost a little buffering.
    """
    try:
        for chunk in outbound.parse_stream(iter(list(buffered))):
            if chunk is None:  # [DONE] sentinel
                continue
            if not isinstance(chunk, dict):
                continue
            for choice in chunk.get("choices") or []:
                if isinstance(choice, dict) and _delta_yields_output(choice.get("delta")):
                    return True
    except Exception as e:  # noqa: BLE001
        print(f"[server:_sse_window_has_output] {e}")
        traceback.print_exc()
    return False


def _open_stream_window(
    resp,
    outbound,
    *,
    max_bytes: int,
    max_seconds: float,
) -> tuple[bytes | None, list[bytes], "Iterator[bytes]", str]:
    """Buffer the opening of a streamed response until it is safe to commit.

    Generalizes ``_peek_stream``. Pulls chunks until the first of:

    * an SSE error event appears in the buffer      -> ``"error"`` (fail over)
    * an output-bearing delta appears               -> ``"content"`` (commit)
    * ``max_bytes`` or ``max_seconds`` is exhausted -> ``"budget"`` (commit)
    * the upstream ends the stream                  -> ``"content"`` or ``"empty"``

    Returns ``(error_body, buffered, rest, reason)``. ``buffered`` holds every
    chunk pulled, in order, for verbatim replay: no byte is ever dropped, so the
    client still receives the first token.

    A stream that *ends* inside the window having produced no output-bearing
    delta returns ``"empty"``, which the caller fails over. That is the streaming
    counterpart of ``_response_unusable``'s empty-completion check, and it is
    precisely the case a first-chunk peek cannot see: a provider that accepts the
    request, emits a role preamble, and then closes.
    """
    chunks = resp.iter_content(chunk_size=None)
    buffered: list[bytes] = []
    total = 0
    started = time.monotonic()
    for chunk in chunks:
        if not chunk:
            continue
        buffered.append(chunk)
        total += len(chunk)
        joined = b"".join(buffered)
        if _sse_prefix_is_error(joined):
            return joined, buffered, chunks, "error"
        if _sse_window_has_output(buffered, outbound):
            return None, buffered, chunks, "content"
        if total >= max_bytes or (time.monotonic() - started) >= max_seconds:
            return None, buffered, chunks, "budget"
    # The upstream closed while we were still inside the window.
    if not buffered:
        return None, buffered, chunks, "empty"
    joined = b"".join(buffered)
    if _sse_prefix_is_error(joined):
        return joined, buffered, chunks, "error"
    if _sse_window_has_output(buffered, outbound):
        return None, buffered, chunks, "content"
    return None, buffered, chunks, "empty"


class _ReplayUpstream:
    """A already-consumed upstream stream, re-served from memory.

    ``server.stream_buffer_full`` reads a candidate's whole response before
    committing, which means the real ``requests.Response`` is exhausted by the
    time the relay code runs. Both relay paths (identity passthrough and the
    dialect translator) expect an object they can iterate and close, so the
    buffered bytes are handed back wearing the same shape. Only the members
    those paths actually touch are implemented.
    """

    def __init__(self, status_code: int, chunks: list[bytes]) -> None:
        self.status_code = status_code
        self.headers: dict = {"Content-Type": "text/event-stream"}
        self._chunks = chunks

    def iter_content(self, chunk_size=None):  # noqa: ANN001 — requests' signature
        yield from self._chunks

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:  # noqa: ANN002
        return False


def _drain_stream(
    rest, buffered: list[bytes], max_bytes: int
) -> tuple[bool, str]:
    """Read the remainder of a stream into *buffered*.

    Returns ``(ok, detail)``. ``ok`` is False when the upstream died partway,
    which — because nothing has been sent to the client yet — the caller can
    still treat as an ordinary candidate failure and fail over from.

    Stops early once *max_bytes* is buffered. A response that large is being
    relayed to a client that asked for a stream, so the right move is to commit
    what we have and hand the rest over incrementally rather than grow the
    buffer without bound.
    """
    total = sum(len(c) for c in buffered)
    try:
        for chunk in rest:
            if not chunk:
                continue
            buffered.append(chunk)
            total += len(chunk)
            if total >= max_bytes:
                return True, "truncated"
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        return False, f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        print(f"[server:_drain_stream] {e}")
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"
    return True, "complete"


def _sse_prefix_is_error(prefix: bytes) -> bool:
    """True when the opening SSE bytes encode a JSON object carrying an error."""
    for line in prefix.split(b"\n"):
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[len(b"data:"):].strip()
        if not line or line == b"[DONE]":
            continue
        try:
            data = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict) and data.get("error"):
            return True
    return False


# ---------------------------------------------------------------------------
# Virtual models — shared cycling logic + per-model candidate selectors
# ---------------------------------------------------------------------------

def _candidate_max_attempts(idx: int, total: int) -> int:
    """How many times to try one candidate before failing over.

    While alternatives remain (not the last candidate) a transient failure
    (429/5xx/timeout) fails over *immediately* — one attempt, no backoff — so a
    rate-limited or flaky upstream never stalls the pipeline when another (often
    free or local) model could answer now. The last candidate, having no
    fallback, gets the full ``_VIRTUAL_MAX_RETRIES`` same-candidate retries.
    """
    return (_VIRTUAL_MAX_RETRIES + 1) if idx == total - 1 else 1


def _record_quota_saturation(provider_name: str, provider_cfg: dict, upstream_model: str, retry_after) -> None:
    """Cool a candidate (and, for allowance-backed providers, its circuit) on a quota error.

    Marks the specific account/model saturated so it drops to the back of the
    pool until its reset. When the provider carries a shared ``free_allowance``
    the whole provider/account circuit is opened too, so concurrent in-flight
    requests stop hammering an allowance that's already depleted.
    """
    account_id = provider_account_id(provider_cfg)
    _mark_saturated(_usage_key(provider_name, upstream_model, account_id), retry_after)
    if _provider_free_allowance(provider_cfg):
        _mark_provider_circuit(provider_name, account_id, retry_after)


def _call_with_account_failover(
    endpoint: str,
    provider_name: str,
    provider_cfg: dict,
    payload: dict,
    timeout: int,
    *,
    forwarded_headers: dict | None = None,
) -> tuple["Response | None", str | None]:
    """Call one model, rotating across the provider's accounts on quota errors.

    Tries the provider's accounts fresh-first (cooling ones last), cooling any
    that return a 402/429 or quota-shaped body so the rotation is sticky. Returns
    early on the first usable response or on a non-quota hard error (which another
    account would hit the same way). Returns ``(resp, account_id)`` of the winning
    — or last — attempt. Usage is NOT recorded here; the caller records it under
    the returned account_id. This is fusion's per-model equivalent of the plain
    cycling engine's account rotation, used by the panel, judge, and synthesizer.
    """
    upstream_model = payload.get("model")
    accounts = provider_accounts(provider_cfg)
    fresh = [a for a in accounts if not _is_candidate_saturated(provider_name, upstream_model, a.id)]
    cooling = [a for a in accounts if _is_candidate_saturated(provider_name, upstream_model, a.id)]
    last_resp: Response | None = None
    last_acct: str | None = None
    for acct in fresh + cooling:
        bound = account_bound_cfg(provider_cfg, acct)
        resp = _proxy_request(endpoint, provider_name, bound, payload, timeout,
                              forwarded_headers=forwarded_headers)
        last_resp, last_acct = resp, acct.id
        if resp.status_code < 400:
            if not _response_unusable(resp.get_data()):
                return resp, acct.id
            if _is_quota_error(200, resp.get_data()):
                _record_quota_saturation(provider_name, bound, upstream_model, None)
                continue
            return resp, acct.id  # unusable but not quota — let the caller decide
        if _is_quota_error(resp.status_code, resp.get_data()):
            _record_quota_saturation(provider_name, bound, upstream_model, resp.headers.get("Retry-After"))
            continue
        return resp, acct.id  # non-quota hard error — another account won't help
    return last_resp, last_acct


def _bind_freshest_account(provider_name: str, provider_cfg: dict, upstream_model: str) -> tuple[dict, str | None]:
    """Return ``(cfg, account_id)`` bound to the freshest non-cooling account.

    A single-credential provider returns its cfg unchanged. Used by the streaming
    synthesizer, which cannot rotate mid-stream, to at least avoid a credential
    already known to be rate-limited.
    """
    accounts = provider_accounts(provider_cfg)
    if len(accounts) <= 1:
        return provider_cfg, provider_account_id(provider_cfg)
    fresh = [a for a in accounts if not _is_candidate_saturated(provider_name, upstream_model, a.id)]
    chosen = (fresh or accounts)[0]
    return account_bound_cfg(provider_cfg, chosen), chosen.id


def _escalate_budget_if_starved(
    endpoint: str,
    provider_name: str,
    provider_cfg: dict,
    upstream_payload: dict,
    resp: "Response",
    timeout: int,
    label: str,
    *,
    deadline: float | None = None,
) -> "Response":
    """Retry a budget-starved 200 on the *same* candidate with a larger budget.

    When *resp* is an empty completion truncated on ``max_tokens`` (see
    ``_is_budget_truncated_empty``), the model spent its whole budget thinking and
    had nothing left to say. Rather than failing over — the strongest model is
    usually the one that reasons this hard — give it more room: multiply the
    budget and retry, up to ``server.budget_escalation_max_retries`` times or
    until the budget hits the ceiling (see ``_budget_ceiling_for``). Returns the
    first usable response, or the last attempt (which the caller's normal
    failover path then handles — a still-empty body is caught by
    ``_response_unusable`` and walks to the next candidate, so this escalates and
    then fails rather than handing back an empty 200).

    ``server.budget_escalation: false`` skips the whole thing, returning the
    empty body immediately for ordinary failover to handle.

    ``deadline`` is the caller's wall-clock budget. This is the one place that
    spends *several* full timeouts on a single candidate, so without the check a
    240-second cycle budget could still be overrun by two more 60-second calls.
    Keyword-only with a None default, so callers that do not set a deadline (and
    every existing test) behave exactly as before.
    """
    if not _budget_escalation_enabled():
        return resp
    payload = upstream_payload
    for _ in range(max(0, _budget_bump_max_retries())):
        if resp.status_code >= 400 or not _is_budget_truncated_empty(resp.get_data()):
            return resp
        bumped = _bumped_budget(payload, provider_name)
        if bumped is None:
            return resp
        attempt_timeout = _timeout_for_candidate(deadline, timeout)
        if attempt_timeout is None:
            logger.warning(
                "  [%s] %s/%s was budget-starved but the cycle deadline leaves no "
                "room to retry it; returning the empty body",
                label, provider_name, upstream_payload.get("model"),
            )
            return resp
        new_budget = bumped.get("max_completion_tokens") or bumped.get("max_tokens")
        logger.warning(
            "  [%s] %s/%s returned an empty, budget-truncated body; "
            "retrying with a larger token budget (%s)",
            label, provider_name, upstream_payload.get("model"), new_budget,
        )
        payload = bumped
        resp = _proxy_request(endpoint, provider_name, provider_cfg, payload, attempt_timeout)
    return resp


ROUTE_HEADER_SELECTED_MODEL = "X-LLMProxy-Selected-Model"
ROUTE_HEADER_ROUTE_REASON = "X-LLMProxy-Route-Reason"
# Every provenance header shares this prefix, which is what lets a response
# rebuild copy them forward without having to know their individual names.
ROUTE_HEADER_PREFIX = "x-llmproxy-"


def _route_reason_with_attempt(route_reason: str, attempt_index: int) -> str:
    """Append the failover marker when *attempt_index* is not the ranked pick.

    ``attempt_index`` is 0 for the first-ranked candidate; anything higher means
    the ranked pick failed and this is a failover, which is appended to the
    reason so the provenance alone distinguishes "chosen" from "settled for".
    """
    if attempt_index > 0:
        return f"{route_reason},{ROUTE_SOURCE_FAILOVER}#{attempt_index}"
    return route_reason


def _note_selected_model(
    provider_name: str,
    upstream_model: str,
    *,
    route_reason: str | None = None,
    attempt_index: int = 0,
) -> None:
    """Record which candidate served this request, for the after_request stamp.

    Callers record at the moment of selection; ``_stamp_route_provenance`` then
    applies the headers to whatever ``Response`` finally leaves the app. The two
    halves together are what make the guarantee hold. A response object is
    rebuilt several times downstream of selection — dialect rendering, error
    wrapping, cache replay — and any rebuild that does not deliberately copy
    headers drops them silently, which is exactly how the route headers came to
    be missing from /v1/messages and /v1/responses. Stamping at the single exit
    point all replies pass through means a future rebuild cannot reintroduce
    that.

    A no-op outside a request context, so the cycling engines remain directly
    callable from unit tests.
    """
    if not has_request_context():
        return
    try:
        g.llmproxy_selected_model = f"{provider_name}/{upstream_model}"
        if route_reason is not None:
            g.llmproxy_route_reason = _route_reason_with_attempt(route_reason, attempt_index)
    except Exception as e:  # noqa: BLE001 — never fail a request over provenance
        print(f"[server:_note_selected_model] {e}")
        traceback.print_exc()


def _stamp_route_provenance(resp: "Response") -> "Response":
    """Apply the recorded route provenance to *resp* if it is not already there.

    Called from the ``after_request`` hook, so it sees the final response object
    however many times it was rebuilt on the way. Values already present win:
    ``_stamp_route_headers`` stamps at the point of selection so a response
    carries them the moment it is built, and this pass only fills the gaps.

    Flask finalizes the response — running ``after_request`` — before the WSGI
    server iterates the body, so streamed replies get their headers before any
    bytes flow.
    """
    try:
        selected = g.get("llmproxy_selected_model")
        if selected and ROUTE_HEADER_SELECTED_MODEL not in resp.headers:
            resp.headers[ROUTE_HEADER_SELECTED_MODEL] = selected
        reason = g.get("llmproxy_route_reason")
        if reason and ROUTE_HEADER_ROUTE_REASON not in resp.headers:
            resp.headers[ROUTE_HEADER_ROUTE_REASON] = reason
    except Exception as e:  # noqa: BLE001 — never fail a served response over a header
        print(f"[server:_stamp_route_provenance] {e}")
        traceback.print_exc()
    return resp


def _carry_route_headers(src: "Response", dst: "Response") -> "Response":
    """Copy the ``X-LLMProxy-*`` headers from *src* onto *dst*.

    For the places that deliberately rebuild a response and must not lose the
    provenance stamped upstream of them. Only this prefix is copied: the rebuilt
    body has its own length and content type, so carrying everything across
    would hand the client a wrong ``Content-Length``.
    """
    try:
        for key, value in src.headers.items():
            if key.lower().startswith(ROUTE_HEADER_PREFIX):
                dst.headers[key] = value
    except Exception as e:  # noqa: BLE001 — never fail a served response over a header
        print(f"[server:_carry_route_headers] {e}")
        traceback.print_exc()
    return dst


def _report_route_enabled(config: dict | None = None) -> bool:
    """Whether to attach the additive ``llmproxy_route`` body block.

    On by default: strict OpenAI clients ignore unknown top-level keys, which is
    the same bet ``llmproxy_fusion`` already makes. The switch exists for a
    client that validates its response schema strictly enough to reject one.
    Turning it off leaves the response headers in place.
    """
    return _config_bool("report_route", True, config)


def _build_route_report(
    virtual_model: str,
    provider_name: str,
    upstream_model: str,
    route_reason: str | None,
    attempt_index: int,
) -> dict:
    """Assemble the ``llmproxy_route`` block for the candidate that answered."""
    return _route_report.build_route_report(
        virtual=virtual_model,
        provider=provider_name,
        model=upstream_model,
        route_reason=_route_reason_with_attempt(route_reason, attempt_index)
        if route_reason is not None else None,
        attempt_index=attempt_index,
    )


def _stamp_route_headers(
    resp: "Response",
    route_reason: str | None,
    provider_name: str,
    upstream_model: str,
    attempt_index: int,
    *,
    virtual_model: str | None = None,
    config: dict | None = None,
) -> "Response":
    """Record on *resp* which candidate served it and why it was ranked first.

    ``attempt_index`` is 0 for the first-ranked candidate; anything higher means
    the ranked pick failed and this is a failover, which is appended to the
    reason so a header alone distinguishes "chosen" from "settled for".

    Also records the selection on the request context, so the after_request
    stamp can restore the headers if a later rebuild drops them, and — when the
    request named a virtual model — attaches the same facts to the JSON body,
    since most SDK clients never expose response headers to their callers.
    """
    _note_selected_model(
        provider_name, upstream_model,
        route_reason=route_reason, attempt_index=attempt_index,
    )
    if route_reason is None:
        return resp
    reason = _route_reason_with_attempt(route_reason, attempt_index)
    try:
        resp.headers[ROUTE_HEADER_ROUTE_REASON] = reason
        resp.headers[ROUTE_HEADER_SELECTED_MODEL] = f"{provider_name}/{upstream_model}"
    except Exception:  # noqa: BLE001 — never fail a served response over a header
        logger.debug("could not stamp route headers", exc_info=True)
    if virtual_model is None or not _report_route_enabled(config):
        return resp
    # Only JSON bodies: a streamed response carries its provenance in a leading
    # SSE frame instead, and an error body is left exactly as the upstream sent
    # it so the client still sees the real diagnostic.
    try:
        if not (200 <= resp.status_code < 300):
            return resp
        if not (resp.content_type or "").startswith("application/json"):
            return resp
        report = _build_route_report(
            virtual_model, provider_name, upstream_model, route_reason, attempt_index,
        )
        injected = _route_report.inject_route_report(resp.get_data(), report)
        return _carry_route_headers(resp, Response(
            injected, status=resp.status_code, content_type=resp.content_type,
        ))
    except Exception as e:  # noqa: BLE001 — never fail a served response over provenance
        print(f"[server:_stamp_route_headers] {e}")
        traceback.print_exc()
        return resp


def _proxy_cycling_non_streaming(
    endpoint: str,
    label: str,
    candidates: list[tuple[str, dict, str]],
    payload: dict,
    timeout: int,
    on_success: Callable[..., None] | None = None,
    route_reason: str | None = None,
    virtual_model: str | None = None,
    config: dict | None = None,
) -> Response:
    """Try each candidate in order, returning the first success.

    Failover is triggered by any of: an HTTP error, a 200 that fails to deliver
    a *forced* capability (e.g. ``tool_choice`` forced a tool call but the body
    has none), or a 200 whose body is unusable (an error object or no
    ``choices``).  A *transient* failure (HTTP 429/5xx, which also covers the
    502/504 ``_proxy_request`` synthesizes for connection errors and timeouts)
    fails over to the next candidate immediately while alternatives remain, and
    is only retried on the same candidate (up to ``_VIRTUAL_MAX_RETRIES`` times
    with a short backoff) when it is the last candidate — see
    ``_candidate_max_attempts``.  When every candidate is exhausted the last
    response is returned so the client still receives the real upstream body
    rather than a synthesized error.

    Before failing over a 200, a candidate that answered with an empty completion
    truncated on ``max_tokens`` (a reasoning model that spent the whole budget
    thinking) is retried on the *same* candidate with a larger token budget — see
    ``_escalate_budget_if_starved`` — so the strongest model still answers instead
    of being skipped for an avoidable empty body.

    ``on_success`` is invoked as ``on_success(provider, model, body)`` with the
    successful response bytes so the caller can record token + cost usage.
    """
    candidate_timeout = min(timeout, _virtual_timeout(config) or _VIRTUAL_CANDIDATE_TIMEOUT)
    deadline = _cycle_deadline(config)
    total = len(candidates)
    last: Response | None = None
    _attempted: list[tuple[str, str, int | None]] = []
    for idx, (provider_name, provider_cfg, upstream_model) in enumerate(candidates):
        # The first candidate always gets its attempt: an already-tight budget
        # must never produce a 503 with zero upstream calls made.
        attempt_timeout = candidate_timeout if idx == 0 else _timeout_for_candidate(
            deadline, candidate_timeout
        )
        if attempt_timeout is None:
            logger.warning(
                "  [%s] cycle deadline reached after %d candidate(s); returning the last error",
                label, idx,
            )
            break
        account_id = provider_account_id(provider_cfg)
        upstream_payload = {**payload, "model": upstream_model}
        # Record the candidate before trying it, not only on success. When every
        # candidate fails, the reply that says so is exactly the one where
        # knowing the last model tried is most useful, and it has no other way
        # to say. A success overwrites this with the same value.
        _note_selected_model(
            provider_name, upstream_model, route_reason=route_reason, attempt_index=idx,
        )
        max_attempts = _candidate_max_attempts(idx, total)
        for attempt in range(max_attempts):
            logger.info("  [%s] trying %s/%s", label, provider_name, upstream_model)
            _started = time.monotonic()
            resp = _proxy_request(endpoint, provider_name, provider_cfg, upstream_payload, attempt_timeout)
            _elapsed_ms = (time.monotonic() - _started) * 1000.0
            if resp.status_code < 400 or not _is_transient_status(resp.status_code):
                break
            if attempt < max_attempts - 1:
                # Same-candidate retries only happen on the last candidate, so an
                # exhausted budget here means there is nowhere left to go anyway;
                # spending the backoff would be pure added latency.
                retry_timeout = _timeout_for_candidate(deadline, candidate_timeout)
                if retry_timeout is None:
                    logger.warning(
                        "  [%s] %s/%s returned %d but the cycle deadline leaves no room to retry",
                        label, provider_name, upstream_model, resp.status_code,
                    )
                    break
                logger.warning(
                    "  [%s] %s/%s returned %d, retrying (%d/%d)",
                    label, provider_name, upstream_model, resp.status_code,
                    attempt + 1, max_attempts - 1,
                )
                attempt_timeout = retry_timeout
                time.sleep(_VIRTUAL_RETRY_BACKOFF)
        # A 200 that emitted no visible content only because it ran out of token
        # budget gets a larger budget on this same candidate before we fail over.
        if resp.status_code < 400:
            resp = _escalate_budget_if_starved(
                endpoint, provider_name, provider_cfg, upstream_payload,
                resp, attempt_timeout, label, deadline=deadline,
            )
        if resp.status_code < 400:
            body = resp.get_data()
            if _capability_failed(payload, body):
                # Not a provider fault: the model answered, it just lacks the
                # capability this request forced. Ordering by capability is what
                # fixes that, not demoting the provider's health.
                _record_outcome(provider_name, upstream_model, True,
                                latency_ms=_elapsed_ms, account_id=account_id)
                logger.warning(
                    "  [%s] %s/%s returned 200 but did not honor a forced capability, trying next",
                    label, provider_name, upstream_model,
                )
                last = resp
                continue
            if _response_unusable(body):
                # Some providers report quota exhaustion as a 200 with an error
                # body — cool it so the rotation is sticky across requests.
                if _is_quota_error(200, body):
                    _record_quota_saturation(provider_name, provider_cfg, upstream_model, None)
                _record_outcome(provider_name, upstream_model, False,
                                latency_ms=_elapsed_ms, account_id=account_id)
                logger.warning(
                    "  [%s] %s/%s returned 200 with an unusable body (error/empty), trying next",
                    label, provider_name, upstream_model,
                )
                last = resp
                continue
            _record_outcome(provider_name, upstream_model, True,
                            latency_ms=_elapsed_ms, account_id=account_id)
            # A success at or above a recorded size limit disproves it.
            _note_accepted_size(provider_name, upstream_model,
                                _payload_size_bytes(payload))
            if on_success is not None:
                on_success(provider_name, upstream_model, body, account_id)
            return _stamp_route_headers(
                resp, route_reason, provider_name, upstream_model, idx,
                virtual_model=virtual_model, config=config,
            )
        # A timeout is cooled exactly like a 429. Both mean the same thing to the
        # next request — this candidate is not currently answering — and without
        # a cooldown a timing-out model keeps its place in the order and is
        # picked first again, costing another full timeout every time. Health
        # alone does not cover it: it needs several samples to move, so a model
        # that has started timing out stays ranked first for the several requests
        # it takes to notice.
        if resp.status_code == 413:
            # Not cooled and not counted against health: the candidate is fine,
            # this request was simply too big for it. Remember the size so the
            # ordering can route requests at least that large around it, and
            # leave everything smaller untouched.
            _oversize = _payload_size_bytes(payload)
            _record_oversize(provider_name, upstream_model, _oversize)
            logger.warning(
                "  [%s] %s/%s rejected a %d-byte request as too large; "
                "requests at least that size will route around it",
                label, provider_name, upstream_model, _oversize,
            )
        if _is_timeout_response(resp):
            logger.warning(
                "  [%s] %s/%s timed out after %ss — cooling it like a 429",
                label, provider_name, upstream_model, attempt_timeout,
            )
            _record_quota_saturation(provider_name, provider_cfg, upstream_model, None)
        elif _is_quota_error(resp.status_code, resp.get_data()):
            _record_quota_saturation(
                provider_name, provider_cfg, upstream_model, resp.headers.get("Retry-After")
            )
        if _is_upstream_failure(status=resp.status_code):
            _record_outcome(provider_name, upstream_model, False,
                            latency_ms=_elapsed_ms, account_id=account_id)
        _body = resp.get_data()
        _cap = _note_capability_rejection(
            provider_name, upstream_model, resp.status_code, _body
        )
        _note_candidate_failure(
            _attempted, provider_name, upstream_model,
            status=resp.status_code,
            kind=_classify_failure(resp.status_code, _body, capability=_cap,
                                   timed_out=_is_timeout_response(resp)),
            detail=_body,
            virtual_model=virtual_model,
            duration_ms=_elapsed_ms,
        )
        logger.warning(
            "  [%s] %s/%s returned %d, trying next", label, provider_name, upstream_model, resp.status_code
        )
        last = resp

    if last is None:
        return _error(f"No '{label}' models available.", status=503)
    if last.status_code < 400:
        # A 200 that failed a content check (a forced tool call that never came,
        # an unusable body). There is no upstream error status to sanitize, and
        # handing the client the real body is deliberate — see the docstring.
        return last
    status = _exhausted_pool_status(_attempted)
    if status == last.status_code:
        # Unanimous and meaningful to the client — relay it untouched, headers
        # and all, so a Retry-After on a 429 survives.
        return last
    logger.warning(
        "  [%s] every candidate failed (%s); returning %d rather than relaying %d",
        label, ", ".join(f"{pn}/{um}={st}" for pn, um, st in _attempted),
        status, last.status_code,
    )
    return _error_body_response(
        _exhausted_pool_body(label, _attempted, _failure_detail(last.get_data())), status
    )


def _proxy_cycling_streaming(
    endpoint: str,
    label: str,
    candidates: list[tuple[str, dict, str]],
    payload: dict,
    timeout: int,
    on_success: Callable[..., None] | None = None,
    config: dict | None = None,
    *,
    inbound=None,
    route_reason: str | None = None,
    virtual_model: str | None = None,
) -> Response:
    """
    Try each candidate in order.  Checks the HTTP status code — and peeks at the
    first streamed chunk — before committing to stream the response, so failed
    upstreams (including a 200 that immediately errors inside the stream) are
    skipped transparently.  A *transient* failure (HTTP 429/5xx, timeout, or
    connection error) fails over to the next candidate immediately while
    alternatives remain, and is retried on the same candidate (up to
    ``_VIRTUAL_MAX_RETRIES`` times with a short backoff) only when it is the last
    candidate — see ``_candidate_max_attempts``.  When all candidates fail the
    last upstream error body is returned so clients receive the same diagnostic
    information as the non-streaming path.

    Each candidate's ``protocol`` selects its outbound adapter; ``inbound`` (the
    client dialect, default openai) renders the canonical stream. When both are
    identities the raw passthrough below is used unchanged.

    ``on_success`` is invoked (pre-stream) as ``on_success(provider, model)`` to
    count the request for load balancing; token + cost totals are recorded
    post-stream.
    """
    virtual_timeout = _virtual_timeout(config)
    candidate_timeout = min(timeout, virtual_timeout or _VIRTUAL_CANDIDATE_TIMEOUT)
    deadline = _cycle_deadline(config)
    inbound = inbound or get_inbound("openai")
    total = len(candidates)
    last_error: tuple[bytes, int, str] | None = None
    _attempted: list[tuple[str, str, int | None]] = []
    _last_detail: str = ""

    for idx, (provider_name, provider_cfg, upstream_model) in enumerate(candidates):
        # As on the non-streaming path, candidate 0 always gets its attempt.
        # Note the deadline bounds time-to-*commit*, not stream duration: once a
        # stream is committed below, the generator runs for as long as the model
        # talks. A twenty-minute legitimate generation must not be killed by a
        # routing budget.
        attempt_timeout = candidate_timeout if idx == 0 else _timeout_for_candidate(
            deadline, candidate_timeout
        )
        if attempt_timeout is None:
            logger.warning(
                "  [%s] cycle deadline reached after %d candidate(s); returning the last error",
                label, idx,
            )
            break
        account_id = provider_account_id(provider_cfg)
        upstream_payload = {**payload, "model": upstream_model}
        # Record the candidate before trying it, not only on success. When every
        # candidate fails, the reply that says so is exactly the one where
        # knowing the last model tried is most useful, and it has no other way
        # to say. A success overwrites this with the same value.
        _note_selected_model(
            provider_name, upstream_model, route_reason=route_reason, attempt_index=idx,
        )
        max_attempts = _candidate_max_attempts(idx, total)
        base_url = provider_base_url(provider_cfg)
        outbound = get_outbound(provider_cfg.get("protocol"))
        url, headers, body = outbound.build_request(
            endpoint, base_url, provider_cfg, upstream_payload,
            stream=True, forwarded_headers=_forwarded_client_headers(),
        )
        # Shrink the read timeout too. It governs how long we wait for bytes
        # *before committing*; left at the full stream_timeout, one silent
        # upstream could eat the entire deadline inside a single post().
        read_timeout = timeout if deadline is None else min(timeout, max(attempt_timeout, 1.0))
        if virtual_timeout is not None:
            # This is a socket-level read timeout, so it stays in force for the
            # life of the connection — including inside iter_content, after the
            # stream has committed. That is deliberate and is the only thing that
            # can catch a stream which starts normally and then goes quiet: by
            # then the bytes belong to the client and no failover is possible, so
            # a bound on the gap between chunks is all that is left. It measures
            # silence, never total duration, so a long steady generation is safe.
            read_timeout = min(read_timeout, virtual_timeout)

        # Open the upstream. Transient failures fail over to the next candidate
        # immediately unless this is the last one (then same-candidate retries).
        resp = None
        _open_started = time.monotonic()
        for attempt in range(max_attempts):
            logger.info("  [%s] trying %s/%s  [streaming]", label, provider_name, upstream_model)
            try:
                resp = requests.post(url, headers=headers, json=body, stream=True,
                                     timeout=(attempt_timeout, read_timeout))
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_attempts - 1 and _timeout_for_candidate(
                    deadline, candidate_timeout
                ) is not None:
                    logger.warning("  [%s] %s/%s connect error: %s, retrying (%d/%d)",
                                   label, provider_name, upstream_model, e, attempt + 1, max_attempts - 1)
                    time.sleep(_VIRTUAL_RETRY_BACKOFF)
                    continue
                logger.warning("  [%s] %s/%s error: %s, trying next", label, provider_name, upstream_model, e)
                _record_outcome(provider_name, upstream_model, False, account_id=account_id)
                _cool_on_timeout(e, label, provider_name, provider_cfg, upstream_model)
                _last_detail = _note_candidate_failure(
                    _attempted, provider_name, upstream_model,
                    kind=_exception_failure_kind(e), detail=str(e),
                    virtual_model=virtual_model,
                    duration_ms=(time.monotonic() - _open_started) * 1000.0,
                )
                resp = None
                break
            except Exception as e:
                if _is_upstream_failure(e):
                    _record_outcome(provider_name, upstream_model, False, account_id=account_id)
                logger.warning("  [%s] %s/%s error: %s, trying next", label, provider_name, upstream_model, e)
                _last_detail = _note_candidate_failure(
                    _attempted, provider_name, upstream_model,
                    kind=_exception_failure_kind(e), detail=str(e),
                    virtual_model=virtual_model,
                    duration_ms=(time.monotonic() - _open_started) * 1000.0,
                )
                resp = None
                break
            if resp.status_code < 400 or not _is_transient_status(resp.status_code):
                break
            if attempt < max_attempts - 1 and _timeout_for_candidate(
                deadline, candidate_timeout
            ) is not None:
                logger.warning("  [%s] %s/%s -> %d, retrying (%d/%d)",
                               label, provider_name, upstream_model, resp.status_code, attempt + 1, max_attempts - 1)
                resp.close()
                time.sleep(_VIRTUAL_RETRY_BACKOFF)

        if resp is None:
            continue
        if resp.status_code >= 400:
            if resp.status_code == 413:
                # As on the non-streaming path: remember the size rather than
                # cooling the candidate, so only requests at least this large
                # route around it. See the oversize registry.
                _oversize = _payload_size_bytes(payload)
                _record_oversize(provider_name, upstream_model, _oversize)
                logger.warning(
                    "  [%s] %s/%s rejected a %d-byte request as too large; "
                    "requests at least that size will route around it",
                    label, provider_name, upstream_model, _oversize,
                )
            if _is_quota_error(resp.status_code, resp.content):
                _record_quota_saturation(
                    provider_name, provider_cfg, upstream_model, resp.headers.get("Retry-After")
                )
            last_error = (
                resp.content,
                resp.status_code,
                resp.headers.get("Content-Type", "application/json"),
            )
            if _is_upstream_failure(status=resp.status_code):
                _record_outcome(provider_name, upstream_model, False, account_id=account_id)
            _body = resp.content
            _cap = _note_capability_rejection(
                provider_name, upstream_model, resp.status_code, _body
            )
            _last_detail = _note_candidate_failure(
                _attempted, provider_name, upstream_model,
                status=resp.status_code,
                kind=_classify_failure(resp.status_code, _body, capability=_cap),
                detail=_body,
                virtual_model=virtual_model,
                duration_ms=(time.monotonic() - _open_started) * 1000.0,
            )
            resp.close()
            logger.warning(
                "  [%s] %s/%s -> %d, trying next", label, provider_name, upstream_model, resp.status_code
            )
            continue

        try:
            # Inspect the opening of the stream so a 200 that immediately emits
            # an SSE error event fails over like an HTTP error instead of being
            # handed to the client.  Everything read here is replayed verbatim,
            # so the first token is never dropped.
            #
            # With ``server.stream_commit_on_content`` the inspection widens from
            # the first non-empty chunk to the first chunk carrying real output
            # (see ``_open_stream_window``): many providers emit a role preamble
            # before anything else, so surviving a one-chunk peek does not mean
            # the generation works.
            if _precommit_window_enabled(config):
                error_body, buffered, rest, window_reason = _open_stream_window(
                    resp, outbound,
                    max_bytes=_config_int(
                        "stream_precommit_max_bytes", _DEFAULT_PRECOMMIT_MAX_BYTES, config
                    ),
                    max_seconds=_precommit_max_seconds(deadline, config),
                )
            else:
                error_body, prefix, rest = _peek_stream(resp)
                buffered = [prefix] if prefix else []
                window_reason = "peek"
            if error_body is not None:
                # A stream that opens with a quota error cools the candidate too.
                if _is_quota_error(None, error_body):
                    _record_quota_saturation(provider_name, provider_cfg, upstream_model, None)
                last_error = (error_body, 502, "text/event-stream")
                _record_outcome(provider_name, upstream_model, False, account_id=account_id)
                _last_detail = _note_candidate_failure(
                    _attempted, provider_name, upstream_model,
                    status=200,
                    kind=("quota" if _is_quota_error(None, error_body) else "stream"),
                    detail=error_body, virtual_model=virtual_model,
                    duration_ms=(time.monotonic() - _open_started) * 1000.0,
                )
                resp.close()
                logger.warning(
                    "  [%s] %s/%s -> 200 then stream error, trying next",
                    label, provider_name, upstream_model,
                )
                continue
            if window_reason == "empty":
                # The upstream accepted the request, said nothing usable, and
                # closed. Non-streaming calls this an unusable body and fails
                # over; streaming could not see it until the window existed.
                last_error = (
                    b'data: {"error":{"message":"Upstream stream produced no output."}}\n\n',
                    502,
                    "text/event-stream",
                )
                _record_outcome(provider_name, upstream_model, False, account_id=account_id)
                _last_detail = _note_candidate_failure(
                    _attempted, provider_name, upstream_model,
                    status=200, kind="stream",
                    detail="Upstream stream produced no output.",
                    virtual_model=virtual_model,
                    duration_ms=(time.monotonic() - _open_started) * 1000.0,
                )
                resp.close()
                logger.warning(
                    "  [%s] %s/%s -> 200 but the stream produced no output, trying next",
                    label, provider_name, upstream_model,
                )
                continue
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            _record_outcome(provider_name, upstream_model, False, account_id=account_id)
            _cool_on_timeout(e, label, provider_name, provider_cfg, upstream_model)
            _last_detail = _note_candidate_failure(
                _attempted, provider_name, upstream_model,
                kind=_exception_failure_kind(e), detail=str(e),
                virtual_model=virtual_model,
                duration_ms=(time.monotonic() - _open_started) * 1000.0,
            )
            logger.warning("  [%s] %s/%s error mid-peek: %s, trying next", label, provider_name, upstream_model, e)
            resp.close()
            continue

        # Optional whole-response buffering. This is the only configuration in
        # which a streamed request gets *genuine* end-to-end failover: a provider
        # that dies four fifths of the way through a generation has still sent
        # the client nothing, so it can be failed over exactly like a 500. The
        # price is that time-to-first-token becomes time-to-last-token, which is
        # why it is opt-in rather than the default.
        if _stream_buffer_full_enabled(config):
            ok, detail = _drain_stream(
                rest, buffered,
                _config_int(
                    "stream_buffer_max_bytes", _DEFAULT_STREAM_BUFFER_MAX_BYTES, config
                ),
            )
            resp.close()
            whole = b"".join(buffered)
            if not ok:
                last_error = (
                    b'data: {"error":{"message":"Upstream stream failed before completion."}}\n\n',
                    502,
                    "text/event-stream",
                )
                _record_outcome(provider_name, upstream_model, False, account_id=account_id)
                _last_detail = _note_candidate_failure(
                    _attempted, provider_name, upstream_model,
                    status=200, kind="stream",
                    detail=f"Upstream stream failed before completion ({detail}).",
                    virtual_model=virtual_model,
                    duration_ms=(time.monotonic() - _open_started) * 1000.0,
                )
                logger.warning(
                    "  [%s] %s/%s died mid-stream (%s); buffering let us fail over, trying next",
                    label, provider_name, upstream_model, detail,
                )
                continue
            if _sse_prefix_is_error(whole) or not _sse_window_has_output(buffered, outbound):
                if _is_quota_error(None, whole):
                    _record_quota_saturation(provider_name, provider_cfg, upstream_model, None)
                last_error = (
                    whole or b'data: {"error":{"message":"Upstream stream produced no output."}}\n\n',
                    502,
                    "text/event-stream",
                )
                _record_outcome(provider_name, upstream_model, False, account_id=account_id)
                _last_detail = _note_candidate_failure(
                    _attempted, provider_name, upstream_model,
                    status=200,
                    kind=("quota" if _is_quota_error(None, whole) else "stream"),
                    detail=whole or "Upstream stream produced no output.",
                    virtual_model=virtual_model,
                    duration_ms=(time.monotonic() - _open_started) * 1000.0,
                )
                logger.warning(
                    "  [%s] %s/%s buffered stream was unusable, trying next",
                    label, provider_name, upstream_model,
                )
                continue
            # Sound and complete: re-serve it from memory.
            resp = _ReplayUpstream(200, list(buffered))
            rest = iter(())
            window_reason = f"buffered:{detail}"

        # Reactive capability detection (forced-tool/json 200-body checks) is
        # intentionally NOT applied beyond the pre-commit window: validating
        # delta.tool_calls would require buffering the whole SSE stream, which is
        # what ``server.stream_buffer_full`` is for.  Proactive capability
        # ordering still steers streaming requests to capable models.
        #
        # The stream survived the pre-commit window. That is the last moment this
        # path can still fail over: from here the bytes belong to the client.
        # Health is credited here rather than at connect, so "healthy" means the
        # candidate produced output (or exhausted the window) rather than merely
        # having returned a byte. A failure *after* this point is caught by the
        # generator below and demoted through ``_demote_on_mid_stream_failure``.
        _record_outcome(provider_name, upstream_model, True, account_id=account_id)
        # A stream that survived the pre-commit window was accepted, so a size
        # limit recorded for this target at or below this request is wrong.
        _note_accepted_size(provider_name, upstream_model, _payload_size_bytes(payload))
        if on_success is not None:
            on_success(provider_name, upstream_model, None, account_id)
        logger.info(
            "  [%s] %s/%s committed (%s)", label, provider_name, upstream_model, window_reason
        )

        # Translation path: pipe the native stream through the adapters.
        if not (outbound.is_identity and inbound.is_identity):
            return _stamp_route_headers(
                _translated_stream_response(
                    resp, outbound, inbound, provider_name, upstream_model, config,
                    prefix=b"".join(buffered), account_id=account_id,
                ),
                route_reason, provider_name, upstream_model, idx,
            )

        captured_resp = resp
        captured_provider = provider_name
        captured_model = upstream_model
        captured_prefix = list(buffered)
        captured_rest = rest
        # A stream has no body to inject into, and rewriting the upstream's own
        # chunks would forfeit the byte-for-byte relay the buffered prefix
        # depends on. So the provenance rides one synthetic frame in front.
        captured_lead = (
            _route_report.route_chunk(
                _build_route_report(
                    virtual_model, provider_name, upstream_model, route_reason, idx,
                ),
                upstream_model,
            )
            if virtual_model is not None and _report_route_enabled(config)
            else b""
        )

        @stream_with_context
        def generate(r=captured_resp, pn=captured_provider, um=captured_model,
                     pfx=captured_prefix, rst=captured_rest, acct=account_id,
                     lead=captured_lead):
            tail = bytearray()
            try:
                if lead:
                    yield lead
                with r:
                    first = True
                    for chunk in itertools.chain(pfx, rst):
                        if chunk:
                            if first:
                                logger.info("  upstream %d  first chunk: %s", r.status_code, chunk[:200])
                                first = False
                            yield chunk
                            tail += chunk
                            if len(tail) > _STREAM_TAIL_BYTES:
                                del tail[:-_STREAM_TAIL_BYTES]
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                logger.error("[%s] provider=%s timed out mid-stream", label, pn)
                _demote_on_mid_stream_failure(pn, um, e, acct)
                for frame in _stream_error_frames("Upstream stream timed out.", um):
                    yield frame
            except Exception as e:
                logger.error("[%s] provider=%s mid-stream error: %s", label, pn, e)
                traceback.print_exc()
                _demote_on_mid_stream_failure(pn, um, e, acct)
                msg = str(e).replace('"', "'")
                for frame in _stream_error_frames(f"Upstream error: {msg}", um):
                    yield frame
            finally:
                _record_stream_usage(pn, um, bytes(tail), config, account_id=acct)

        return _stamp_route_headers(
            Response(generate(), content_type="text/event-stream"),
            route_reason, provider_name, upstream_model, idx,
        )

    if last_error:
        body, status, ct = last_error
        # Same sanitising as the non-streaming path: a relayed 404 tells an
        # OpenAI-compatible client the model does not exist, which is terminal,
        # so a pool outage silently disables the client's own retry. See
        # _exhausted_pool_status. Synthesized SSE errors (status 502, already
        # ours) carry no upstream status and pass straight through.
        if _attempted:
            sanitized = _exhausted_pool_status(_attempted)
            if sanitized != status:
                logger.warning(
                    "  [%s] every candidate failed (%s); returning %d rather than relaying %d",
                    label, ", ".join(f"{pn}/{um}={st}" for pn, um, st in _attempted),
                    sanitized, status,
                )
                return _error_body_response(
                    _exhausted_pool_body(label, _attempted, _failure_detail(body)), sanitized
                )
        return Response(body, status=status, content_type=ct)
    if _attempted:
        # Every candidate died before producing a response body at all — the
        # classic all-timeouts walk. There is no upstream reply to relay, but
        # candidates WERE tried, so this is an exhausted pool rather than an
        # empty one and must say so with the same roll-call: a bare 503 here
        # told the caller nothing about which models it had just spent minutes
        # on, and read as "nothing to try" when five things had been tried.
        logger.warning(
            "  [%s] every candidate failed without answering (%s); returning 502",
            label, ", ".join(
                f"{pn}/{um}={'-' if st is None else st}" for pn, um, st in _attempted
            ),
        )
        return _error_body_response(
            _exhausted_pool_body(label, _attempted, _last_detail), 502
        )
    return _error(f"All '{label}' model candidates failed or are unavailable.", status=503)


def _cycling_candidates(
    candidates: list[tuple[str, dict, str]],
) -> list[tuple[str, dict, str]]:
    """Rotate candidates to a random starting position for load spreading.

    Also demotes any candidate currently cooling in the saturation registry to
    the back so non-free/non-loadbalanced virtuals (per-provider, reasoning,
    capability families) still rotate off a recently rate-limited model/account
    on the next request, while keeping it reachable as a last resort.
    """
    if not candidates:
        return candidates
    start = random.randrange(len(candidates))
    rotated = candidates[start:] + candidates[:start]
    fresh = [c for c in rotated if not _is_candidate_saturated(c[0], c[2], provider_account_id(c[1]))]
    cooling = [c for c in rotated if _is_candidate_saturated(c[0], c[2], provider_account_id(c[1]))]
    return fresh + cooling if cooling else rotated


# — prompt-cache affinity —
#
# Providers that cache prompt prefixes (Anthropic, OpenAI, DeepSeek) only pay off
# if successive requests in a conversation land on the same upstream *and* the
# same credential. Spreading load defeats that. So affinity is applied narrowly:
# to credential choice within a provider, and to the paid tier of the cost
# waterfall — never to free-tier capacity ordering, where spreading is the point.
#
# Selection is rendezvous (highest-random-weight) hashing rather than a modulo
# ring, so adding or losing an account reshuffles only that account's share
# instead of remapping everything. HRW is Thaler & Ravishankar (1996); see
# THIRD_PARTY_NOTICES.md.

# Cap on a client-supplied cache key, matching the largest value providers accept.
_AFFINITY_KEY_MAX_CHARS = 4096
# A system prompt shorter than this is not worth pinning a conversation for.
_AFFINITY_MIN_SYSTEM_CHARS = 200


def _affinity_key(payload: dict) -> str | None:
    """Return a stable key identifying this CONVERSATION, or None.

    Prefers a client-supplied ``prompt_cache_key`` (top-level or under
    ``metadata``), which is the only source that knows what the client considers
    one conversation. Otherwise the key is derived from the conversation's
    **root**: every system turn, plus the first non-system turn. Those are the
    only messages an agentic client does not rewrite, so the key is identical on
    turn 1 and on turn 40.

    This used to key on the whole cacheable prefix — everything before the
    trailing user turn — which is a correct description of what the upstream has
    cached and completely wrong as an identity. An agent loop appends an
    assistant turn and a tool result on every iteration, so the prefix, and
    therefore the key, was different on every single request. Nothing that
    remembers a choice under that key could ever read it back: the sticky pin
    was written 40 times and looked up 0 times, and the conversation restarted
    from the top of the ranking every turn, paying the 429s of every model above
    its own before reaching the one that had just worked. Keying on the root
    fixes both users of this key, since an upstream prompt cache is keyed by
    prefix and the root is a prefix of every turn.

    Returns None for a bare first user turn with no substantial system prompt:
    there is nothing durable to identify the conversation by, so pinning would
    cost load spreading and buy no cache hit. A later turn of that same
    conversation does get a key, because by then the first user turn is a
    settled part of the transcript.

    A client that rewrites its system prompt every turn (injecting a timestamp,
    say) defeats any derived key. ``prompt_cache_key`` is the escape hatch, and
    is what an agent framework should send.
    """
    if not isinstance(payload, dict):
        return None

    explicit = payload.get("prompt_cache_key")
    if not isinstance(explicit, str) or not explicit.strip():
        meta = payload.get("metadata")
        explicit = meta.get("prompt_cache_key") if isinstance(meta, dict) else None
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()[:_AFFINITY_KEY_MAX_CHARS]

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    def _content(msg) -> str:
        if not isinstance(msg, dict):
            return ""
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return ""

    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    system_text = "".join(
        _content(m) for m in messages
        if isinstance(m, dict) and m.get("role") == "system"
    )
    is_continuation = any(r in ("assistant", "tool") for r in roles)

    # Worth pinning only once the conversation has something durable to be
    # identified by: a substantial system prompt, or a transcript that has
    # already progressed past its opening turn.
    if not is_continuation and len(system_text) < _AFFINITY_MIN_SYSTEM_CHARS:
        return None

    # The root: system turns, then the first non-system turn. Everything after
    # it is rewritten as the conversation grows and must not enter the key.
    first_turn = next(
        (m for m in messages if isinstance(m, dict) and m.get("role") != "system"),
        None,
    )
    if first_turn is None and not system_text:
        return None

    digest = hashlib.sha256()
    digest.update(system_text.encode("utf-8", "ignore"))
    digest.update(b"\x01")
    if first_turn is not None:
        digest.update(f"{first_turn.get('role')}\x00".encode())
        digest.update(_content(first_turn).encode("utf-8", "ignore"))
    return f"conv:{digest.hexdigest()}"


def _rendezvous_rank(key: str, identity: str) -> int:
    """Score one candidate identity for *key*; the highest score wins.

    The NUL separator keeps ``("ab", "c")`` from colliding with ``("a", "bc")``.
    """
    digest = hashlib.sha256(f"{key}\x00{identity}".encode("utf-8", "ignore")).hexdigest()
    return int(digest[:32], 16)


# affinity key -> (target, last_touched). Model-level stickiness for free
# pools, opt-in via server.free_tier_cache_affinity.
#
# Rendezvous hashing was the previous mechanism and is the wrong shape for a
# RANKED pool. It is stateless: it remembers no choice, it derives one, and the
# winner is uncorrelated with rank. On an unranked free pool that is invisible,
# because no candidate was better than another to begin with. On
# flagship__free, whose whole premise is strict best-first, it would hand most
# conversations a hash-chosen member from their very first turn.
#
# Sticky-until-failure is what the flag is understood to mean and works on both
# kinds of pool: the first turn gets whatever the ordering says is best, the
# pin records what actually WORKED, and later turns keep it until it stops
# working. Best-first and stickiness stop being in tension, because the pin is
# set by the ordering rather than competing with it.
_AFFINITY_PIN_MAX: int = 2048
_AFFINITY_PIN_TTL_S: float = 6 * 60 * 60
_affinity_pins: dict[str, tuple[tuple[str, str], float]] = {}
_affinity_pin_lock = threading.Lock()


def _prune_affinity_pins_locked() -> None:
    """Drop expired pins, then the oldest, until the map is back inside its cap.

    Caller holds ``_affinity_pin_lock``. The cap matters more than the TTL: an
    unbounded map keyed by conversation is a slow leak on a long-lived process.
    """
    cutoff = time.monotonic() - _AFFINITY_PIN_TTL_S
    for key in [k for k, (_t, seen) in _affinity_pins.items() if seen < cutoff]:
        _affinity_pins.pop(key, None)
    if len(_affinity_pins) <= _AFFINITY_PIN_MAX:
        return
    for key, _ in sorted(_affinity_pins.items(), key=lambda kv: kv[1][1])[
        : len(_affinity_pins) - _AFFINITY_PIN_MAX
    ]:
        _affinity_pins.pop(key, None)


def _record_affinity_success(affinity_key: str | None, provider_name: str, upstream_model: str) -> None:
    """Pin this conversation to the target that just served it successfully.

    Pinned on SUCCESS rather than on selection, so the pin always names a model
    that demonstrably worked for this conversation rather than one that was
    merely tried first.
    """
    if not affinity_key:
        return
    with _affinity_pin_lock:
        _affinity_pins[affinity_key] = ((provider_name, upstream_model), time.monotonic())
        _prune_affinity_pins_locked()


def _affinity_pinned_target(affinity_key: str | None) -> tuple[str, str] | None:
    """The target this conversation is pinned to, if the pin is still live."""
    if not affinity_key:
        return None
    with _affinity_pin_lock:
        entry = _affinity_pins.get(affinity_key)
        if not entry:
            return None
        target, seen = entry
        if seen < time.monotonic() - _AFFINITY_PIN_TTL_S:
            _affinity_pins.pop(affinity_key, None)
            return None
        return target


def _reset_affinity_pins() -> None:
    """Clear every pin. Used by tests and by an explicit usage reset."""
    with _affinity_pin_lock:
        _affinity_pins.clear()


def _order_by_sticky_affinity(
    candidates: list[tuple[str, dict, str]],
    affinity_key: str | None,
) -> list[tuple[str, dict, str]]:
    """Move this conversation's pinned target to the front, if it is still here.

    Never drops and never reorders anything else, so failover is unaffected and
    an unpinned conversation keeps exactly the order the earlier passes built.
    A pin for a target that is no longer a candidate is simply ignored, and the
    next success re-pins.

    A pinned target that is currently COOLING is deliberately not promoted. The
    orderings demote a candidate cooling after a 402/429 to the back, and
    hoisting it straight back to the front would spend the conversation's next
    turn on the one model already known to be rate limited — turning stickiness
    into a guaranteed wasted attempt every turn until the window cleared.

    No explicit unpin is needed on failure. The pin is written on SUCCESS, so a
    turn whose pinned model fails and which is then served by another candidate
    re-pins to that candidate as part of the same request: the pin always names
    the last model that actually worked, and corrects itself in one turn.
    """
    target = _affinity_pinned_target(affinity_key)
    if not target or len(candidates) < 2:
        return candidates
    if _is_candidate_saturated(target[0], target[1]):
        return candidates
    for idx, (pn, _cfg, um) in enumerate(candidates):
        if (pn, um) == target:
            if idx == 0:
                return candidates
            return [candidates[idx]] + candidates[:idx] + candidates[idx + 1:]
    return candidates


def _order_by_cache_affinity(
    candidates: list[tuple[str, dict, str]],
    affinity_key: str | None,
) -> list[tuple[str, dict, str]]:
    """Stable-sort *candidates* so the HRW winner for *affinity_key* comes first.

    A no-op without a key or with fewer than two candidates. Never drops a
    candidate — like every other ordering pass here, it only reorders, so a
    pinned upstream that is down still fails over normally.
    """
    if not affinity_key or len(candidates) < 2:
        return candidates
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            -_rendezvous_rank(affinity_key, f"{item[1][0]}/{item[1][2]}"),
            item[0],
        ),
    )
    return [c for _, c in ranked]


def _cache_affinity_applies(
    candidates: list[tuple[str, dict, str]],
    config: dict,
    is_loadbalanced: bool,
) -> bool:
    """True when affinity could actually reorder *candidates*.

    Affinity only has somewhere to bite in two places: choosing between a
    provider's several credentials, and the paid tier of the cost waterfall.
    Everywhere else — a single-credential provider, a free-tier route — the pass
    runs and changes nothing, and reporting it as a reason would overstate what
    happened. The route header is only worth having if it is accurate.
    """
    for _pn, pc, _um in candidates:
        if len(provider_accounts(pc)) > 1 and provider_account_strategy(pc) == "round_robin":
            return True
    if is_loadbalanced and _allow_implicit_paid(config):
        paid = sum(1 for pn, pc, um in candidates if _cost_tier(pn, um, pc, config) == _TIER_PAID)
        if paid > 1:
            return True
    return False


def _expand_accounts(
    candidates: list[tuple[str, dict, str]],
    payload: dict | None = None,
) -> list[tuple[str, dict, str]]:
    """Fan each candidate out into one per configured account, expanded LAST.

    Ordering upstream runs at model granularity; this final pass replaces each
    ``(provider, cfg, model)`` with one candidate per credential, bound to that
    account's key via :func:`account_bound_cfg`. A model's accounts stay
    **adjacent, in the model's ranked slot**, so the cycling walk tries every
    credential of a model (accounts-first) before moving to the next model —
    same model, fresh quota is always the cheapest way to keep serving.

    Within a model, accounts that are not currently cooling come first (rotated
    to a random start for ``round_robin``, kept in priority order otherwise);
    accounts cooling after a recent 402/429 are appended last but still
    reachable. A single-account provider (the common case) expands to exactly
    one candidate with the original cfg untouched, so keys/headers/behavior are
    byte-identical to before.

    When *payload* carries a cacheable prefix (see :func:`_affinity_key`), the
    ``round_robin`` rotation is replaced by a rendezvous-hash ordering keyed on
    that prefix, so successive requests in one conversation keep landing on the
    same credential and the upstream's prompt cache actually hits. Priority
    strategies are left alone — an operator who ranked their accounts meant it.
    """
    if not candidates:
        return candidates
    affinity_key = _affinity_key(payload) if payload else None
    expanded: list[tuple[str, dict, str]] = []
    for pn, pc, um in candidates:
        accounts = provider_accounts(pc)
        if len(accounts) <= 1:
            expanded.append((pn, pc, um))  # lone credential — leave cfg as-is
            continue
        fresh = [a for a in accounts if not _is_candidate_saturated(pn, um, a.id)]
        cooling = [a for a in accounts if _is_candidate_saturated(pn, um, a.id)]
        if provider_account_strategy(pc) == "round_robin" and len(fresh) > 1:
            if affinity_key:
                # Deterministic per conversation, so the prompt cache survives
                # the next request; still spread across conversations.
                fresh = sorted(
                    fresh,
                    key=lambda a: -_rendezvous_rank(affinity_key, f"{pn}#{a.id}/{um}"),
                )
            else:
                start = random.randrange(len(fresh))
                fresh = fresh[start:] + fresh[:start]
        for acct in fresh + cooling:
            expanded.append((pn, account_bound_cfg(pc, acct), um))
    return expanded


def _capacity_ordered_candidates(
    candidates: list[tuple[str, dict, str]],
    free_limits: dict[str, dict],
) -> list[tuple[str, dict, str]]:
    """
    Order candidates by remaining free-tier capacity using weighted sampling.

    Algorithm:
    - Each candidate is scored via _capacity_score() using its RPM/RPD usage.
    - Candidates with no configured limits score 1.0 (treated as unlimited).
    - A candidate currently cooling in the saturation registry (recent 402/429)
      is forced to score 0.0, so a just-rate-limited account/model drops to the
      back on the *next* request too — the "transparent rotation" the free
      virtual promises — rather than being re-picked first every time.
    - Candidates with score > 0 are drawn via weighted reservoir sampling so
      higher-capacity models are preferred while load is still distributed.
    - Candidates with score == 0 (at limit or cooling) are appended as
      last-resort fallbacks; they still get tried so a saturated model doesn't
      cause an avoidable 503.
    - Every non-zero score is scaled by ``_health_score``, so a candidate whose
      recent attempts have been failing or crawling sinks in the rotation. It is
      a multiplier rather than a filter: quota says what a provider will still
      *accept*, health says whether it currently *works*, and a degraded provider
      that answers still beats a 503.
    - Falls back to random rotation when no candidate has any configured limits
      and none is currently cooling.

    Usage/saturation are keyed per account (see _usage_key), so with several
    accounts on a provider each meters and cools independently.

    Note: tracking is per-worker-process; gunicorn multi-worker deployments
    may undercount usage relative to the provider's actual view.
    """
    if not candidates:
        return candidates

    def _acct(pc):
        return provider_account_id(pc)

    any_limits = any(
        _usage_key(pn, um, _acct(pc)) in free_limits or f"{pn}/{um}".lower() in free_limits
        for pn, pc, um in candidates
    )
    any_saturated = any(
        _is_candidate_saturated(pn, um, _acct(pc)) for pn, pc, um in candidates
    )
    if not any_limits and not any_saturated:
        start = random.randrange(len(candidates))
        return candidates[start:] + candidates[:start]

    scored: list[tuple[tuple[str, dict, str], float]] = []
    for pn, pc, um in candidates:
        account_id = _acct(pc)
        key = _usage_key(pn, um, account_id)
        limits = free_limits.get(key, {}) or free_limits.get(f"{pn}/{um}".lower(), {})
        if _is_candidate_saturated(pn, um, account_id):
            score = 0.0  # recently rate-limited — cool it off, keep it reachable
        else:
            used_min, used_day = _get_usage_snapshot(key)
            used_tok_min, used_tok_day = _get_token_snapshot(key)
            score = _capacity_score(used_min, used_day, limits, used_tok_min, used_tok_day)
            score *= _health_score(pn, um, account_id)
        logger.debug("[capacity] %s  score=%.3f", key, score)
        scored.append(((pn, pc, um), score))

    viable = [(c, s) for c, s in scored if s > 0.0]
    exhausted = [c for c, s in scored if s == 0.0]

    result: list[tuple[str, dict, str]] = []
    remaining = list(viable)
    while remaining:
        total = sum(s for _, s in remaining)
        if total == 0.0:
            result.extend(c for c, _ in remaining)
            break
        r = random.uniform(0.0, total)
        cumulative = 0.0
        picked = len(remaining) - 1
        for i, (_c, s) in enumerate(remaining):
            cumulative += s
            if r <= cumulative:
                picked = i
                break
        result.append(remaining[picked][0])
        remaining.pop(picked)

    result.extend(exhausted)
    return result


def _provider_exposes_to_virtual_models(provider_cfg: dict) -> bool:
    """Return False only when the provider explicitly opts out via expose_to_virtual_models: false."""
    return provider_cfg.get("expose_to_virtual_models", True) is not False


def _allow_implicit_paid(config: dict) -> bool:
    """True when virtual routing may fall back to paid models implicitly.

    Default False: cost-avoiding virtuals (loadbalanced) stop at the free/local
    tiers and surface a clear 429/503 when they are exhausted, so a paid model is
    only ever reached by direct ``provider/model`` name. Set
    ``server.allow_implicit_paid: true`` to restore the free→local→paid waterfall.
    """
    return bool(config.get("server", {}).get("allow_implicit_paid", False))


def _apply_favorite_free_ordering(
    candidates: list[tuple[str, dict, str]],
    config: dict,
) -> list[tuple[str, dict, str]]:
    """Promote favorite_free_models to the front in ranked order.

    Only candidates already present in the pool are promoted — favorites not in
    the pool (e.g. cost-observed, not believed_free) are silently skipped.
    Non-matching candidates retain their existing order after the favorites.

    Matching is case-insensitive and ignores :variant suffixes (e.g. :free,
    :nitro) so that "x/y" matches both "x/y" and "x/y:free".
    """
    favorites = config.get("favorite_free_models", [])
    if not favorites:
        return candidates
    remaining = list(candidates)
    front: list[tuple[str, dict, str]] = []
    for fav in favorites:
        fav_lower = fav.lower()
        for i, (pname, _pcfg, umodel) in enumerate(remaining):
            umodel_lower = umodel.lower()
            umodel_base = umodel_lower.split(":")[0]  # strip :variant suffix
            qualified = f"{pname}/{umodel}".lower()
            qualified_base = f"{pname}/{umodel_base}"
            if fav_lower in (umodel_lower, umodel_base, qualified, qualified_base):
                front.append(remaining.pop(i))
                break
    return front + remaining


def _param_count(model_id: str) -> float:
    """Best-effort parameter count (in billions) parsed from a model id.

    Returns 0.0 when the id carries no "<n>b" hint, so untagged small models
    sort below any model with a known size. Used as the secondary key in
    _quality_key.
    """
    import re as _re
    m = _re.search(r'(\d+(?:\.\d+)?)\s*b\b', model_id.lower())
    return float(m.group(1)) if m else 0.0


def _quality_key(provider_name: str, upstream_id: str,
                 reasoning_map: dict[str, str],
                 flagship_models: set[str] | None = None) -> tuple[int, float]:
    """Sophistication sort key for a candidate — higher is more capable.

    ``(reasoning_rank, param_count)`` where ``reasoning_rank`` is the index of
    the model's tier in _REASONING_LEVELS (flagship > deep > standard >
    exploratory), and ``param_count`` is the inferred size in billions. Sorting
    candidates by this key descending puts the most sophisticated model first.

    Flagship is an overlay: membership is looked up separately rather than read
    from ``model_reasoning``, so a flagship model keeps whatever tier tag it
    carries there and simply outranks it here.

    An unrecognised explicit tag falls back to the name-inferred tier, never to
    rank 0. A config written by a newer build (or a tier this build has since
    renamed) would otherwise sort as the *weakest* candidate rather than the
    strongest, which is the worst possible direction to fail in.
    """
    qualified = f"{provider_name}/{upstream_id}".lower()
    if flagship_models and (qualified in flagship_models
                            or upstream_id.lower() in flagship_models):
        return (_REASONING_LEVELS.index("flagship"), _param_count(upstream_id))
    lvl = (_lookup_model_fact(reasoning_map, provider_name, upstream_id)
           or _infer_reasoning_level(upstream_id))
    if lvl not in _REASONING_LEVELS:
        lvl = _infer_reasoning_level(upstream_id)
    rank = _REASONING_LEVELS.index(lvl) if lvl in _REASONING_LEVELS else 0
    return (rank, _param_count(upstream_id))


def _quality_ordered_candidates(
    candidates: list[tuple[str, dict, str]],
    free_limits: dict[str, dict],
    reasoning_map: dict[str, str],
    flagship_models: set[str] | None = None,
) -> list[tuple[str, dict, str]]:
    """Order free candidates best-first: most sophisticated model with headroom.

    Among candidates that still have free-tier capacity (``_capacity_score`` > 0)
    the most capable model (see _quality_key) is tried first, with remaining
    capacity as a tiebreak among equally-capable models. Saturated candidates
    (score == 0) are appended last — still reachable as a failover so a maxed-out
    model never causes an avoidable 503 — also ordered best-first.

    Deterministic (no random sampling): loadbalanced wants the strongest free
    model each time, and failover handles a rate-limited top pick by moving to
    the next-best on its own.
    """
    if not candidates:
        return candidates

    scored: list[tuple[tuple[str, dict, str], float]] = []
    for pn, pc, um in candidates:
        account_id = provider_account_id(pc)
        key = _usage_key(pn, um, account_id)
        limits = free_limits.get(key, {}) or free_limits.get(f"{pn}/{um}".lower(), {})
        if _is_candidate_saturated(pn, um, account_id):
            score = 0.0  # cooling after a recent 402/429 — demote, keep reachable
        else:
            used_min, used_day = _get_usage_snapshot(key)
            used_tok_min, used_tok_day = _get_token_snapshot(key)
            score = _capacity_score(used_min, used_day, limits, used_tok_min, used_tok_day)
            score *= _health_score(pn, um, account_id)
        scored.append(((pn, pc, um), score))

    def _key(item: tuple[tuple[str, dict, str], float]):
        (pn, _pc, um), score = item
        rank, params = _quality_key(pn, um, reasoning_map, flagship_models)
        return (rank, params, score)

    viable = sorted((it for it in scored if it[1] > 0.0), key=_key, reverse=True)
    exhausted = sorted((it for it in scored if it[1] == 0.0), key=_key, reverse=True)
    return [c for c, _ in viable] + [c for c, _ in exhausted]


# — flagship (benchmark-ranked) ordering —
#
# Flagship is the one tier whose membership is decided by a measured score, so
# it is also the one tier that can be ORDERED by one. Everywhere else the router
# spreads load (free) or rotates (the rest), because it has no basis to call one
# member better than another. Here it does, so the pool is walked strongest-first
# and failover descends it in rank order.


def _flagship_candidate_score(
    provider_name: str, upstream_id: str, scores: dict[str, float]
) -> float | None:
    """This candidate's combined benchmark percentile, or None if nothing scores it.

    Tries the provider's own entry first, then the normalised model key, because
    a score describes the weights rather than the provider serving them: a pinned
    provider that no leaderboard names still ranks correctly when the same model
    was scored under another provider's listing.
    """
    from .flagship import normalize_model_id

    qualified = f"{provider_name}/{upstream_id}".lower()
    if qualified in scores:
        return scores[qualified]
    return scores.get(normalize_model_id(upstream_id))


# Sorts below every real percentile (which live in [0, 1]), so an unscored
# candidate follows every scored one instead of being guessed at.
_FLAGSHIP_UNSCORED: float = -1.0


def _flagship_ordered_candidates(
    candidates: list[tuple[str, dict, str]],
    scores: dict[str, float],
    free_limits: dict[str, dict],
) -> list[tuple[str, dict, str]]:
    """Order a flagship pool strictly best-first by combined benchmark percentile.

    The tier exists to reach the strongest model available, so the highest-ranked
    candidate is tried first and failover walks the rest in descending order. The
    score is the *primary* key and nothing continuous is folded into it: capacity
    and health only break ties, since letting them scale the score would quietly
    turn a strict ranking back into a weighted preference.

    Two departures from a pure sort, both deliberate:

    * A candidate cooling after a recent 402/429, or one with no headroom left,
      is demoted to the back of the list — still reachable, so a saturated top
      pick never causes an avoidable 503, but not re-attempted first on every
      request until its window clears.
    * An unscored candidate (an unscraped pin, or a model whose weights no source
      covers) sorts after every scored one. Nothing can rank it on evidence, and
      promoting it would let one pin preempt a measured top-of-the-field model on
      every request.

    Ties are common — cross-provider duplicates share one model key and therefore
    one score — so remaining capacity breaks them first and the model name breaks
    what is left, which keeps the order deterministic rather than dependent on
    route-cache iteration order. The model leads that last tiebreak so a tied
    block interleaves by model rather than clustering by provider; instances of
    one model still land together, ordered among themselves by provider.

    Returns *candidates* unchanged when nothing in the pool carries a score, so a
    cache written before scores were persisted keeps today's behaviour instead of
    collapsing into an arbitrary order.
    """
    if not candidates:
        return candidates
    if not any(_flagship_candidate_score(pn, um, scores) is not None
               for pn, _pc, um in candidates):
        return candidates

    scored: list[tuple[tuple[str, dict, str], float, float]] = []
    for pn, pc, um in candidates:
        account_id = provider_account_id(pc)
        key = _usage_key(pn, um, account_id)
        limits = free_limits.get(key, {}) or free_limits.get(f"{pn}/{um}".lower(), {})
        if _is_candidate_saturated(pn, um, account_id):
            viability = 0.0  # cooling after a recent 402/429 — demote, keep reachable
        else:
            used_min, used_day = _get_usage_snapshot(key)
            used_tok_min, used_tok_day = _get_token_snapshot(key)
            viability = _capacity_score(
                used_min, used_day, limits, used_tok_min, used_tok_day)
            viability *= _health_score(pn, um, account_id)
        rank = _flagship_candidate_score(pn, um, scores)
        scored.append(((pn, pc, um), rank if rank is not None else _FLAGSHIP_UNSCORED,
                       viability))

    def _key(item: tuple[tuple[str, dict, str], float, float]):
        (pn, _pc, um), rank, viability = item
        # Model before provider in the final tiebreak. Two DISTINCT models that
        # happen to tie should interleave by model rather than cluster by whose
        # provider name sorts first — the tier is meant to be walked model by
        # model. Leading with the provider would group a tied block by provider,
        # which is exactly the shape this ordering exists to get away from. The
        # same weights on several providers share one model id, so they still
        # sort adjacently and order among themselves by provider.
        return (-rank, -viability, um.lower(), pn.lower())

    viable = sorted((it for it in scored if it[2] > 0.0), key=_key)
    exhausted = sorted((it for it in scored if it[2] == 0.0), key=_key)
    return [c for c, _r, _v in viable] + [c for c, _r, _v in exhausted]


# — "free" candidate selector —

# ---------------------------------------------------------------------------
# Routing metadata: three layers, resolved once per request
# ---------------------------------------------------------------------------
#
#   config.json           overrides — yours and the admin UI's, always win
#   routing_metadata.json learned   — rewritten by the refresh cadence
#   providers.json        defaults  — shipped with the repo, PR-able
#
# Five keys resolve this way: believed_free, cost_observed_free_tier,
# model_reasoning, model_capabilities, free_limits. Everything downstream keeps
# reading them off a plain config dict, so the five accessors below change by
# one line each and keep every one of their defensive checks.
#
# List-shaped keys UNION across layers; dict-shaped keys merge per entry with
# the higher layer winning. Union is right for the lists because they are a
# pair: believed_free adds a model to the free pool and cost_observed_free_tier
# removes it again, so additive layers still give complete control. Replacing
# would mean one hand-added free model silently discarded everything the
# refresh had learned.

_ROUTING_LIST_KEYS = ("believed_free", "cost_observed_free_tier")
_ROUTING_DICT_KEYS = ("model_reasoning", "model_capabilities", "free_limits")
# Of those, the ones whose value is a SET OF FACTS rather than a single value,
# so layers and lookup forms combine instead of shadowing one another.
_ROUTING_UNION_KEYS = frozenset({"model_capabilities"})

# Every routing-metadata key that used to live in config.json. Named once so the
# migration, the admin editors and the layer builders cannot disagree about what
# "the five keys" means.
_ROUTING_CONFIG_KEYS = _ROUTING_LIST_KEYS + _ROUTING_DICT_KEYS

# Entries in the two list keys that a PROVIDER declared, rather than a person.
#
# A provider template's believed_free is scoped to that provider: Google's own
# API really does serve google/gemini-3.8-flash free, and that says nothing
# about a gateway which re-serves the identical upstream id and bills for it.
# The two spellings are indistinguishable — Google's QUALIFIED id is character
# for character the gateway's BARE upstream id — so the only thing that can tell
# them apart is where the claim came from.
#
# Deliberately NOT part of _ROUTING_CONFIG_KEYS: it is provenance about those
# keys, not a sixth routing key, and every consumer that enumerates the five
# (the admin editors, the config-shape report) must keep seeing exactly five.
_ROUTING_PROVIDER_SCOPED = "_provider_scoped_ids"

# The sidecar is re-read when it changes on disk. providers.py caches its own
# data for the process lifetime, which is right for a shipped default and wrong
# here: the refresh rewrites this file while the server runs, and a cache that
# outlived the write would pin the router to metadata it had already replaced.
# Keyed by (path, mtime), not mtime alone. One global slot keyed on a float
# meant two sidecars whose mtimes happened to coincide served each other's
# state — and under gunicorn, where a worker may resolve a different
# LLMPROXY_CONFIG, it was the process-wide memo rather than the file that
# decided what routing saw.
_routing_sidecar_cache: tuple[str, float, dict] | None = None
_routing_sidecar_lock = threading.Lock()

# Memoized output of _merged_routing_config.
#
# The merge rebuilds four layers and copies the whole per-provider capability
# snapshot, so it costs milliseconds — and the per-route helpers call it once
# each, twice per model. On a deployment serving a few thousand models that is
# several thousand merges to assemble ONE candidate list, tens of seconds of
# pure CPU, holding the GIL against every other worker thread. It is also
# entirely redundant: every input is identical across those calls.
#
# Invalidation is by generation counter rather than by timestamp, so a change
# is never merely *probably* picked up. The counter is bumped wherever an input
# changes: the capability/context snapshot (a route-cache rebuild) and the
# sidecar (a refresh or an admin write). The sidecar's own path+mtime is folded
# into the key as well, which catches an edit made outside this process.
_routing_merge_cache: dict[tuple, dict] = {}
_routing_merge_lock = threading.Lock()
_routing_layer_generation: int = 0


def _bump_routing_generation() -> None:
    """Invalidate every memoized routing merge. Cheap; call it liberally."""
    global _routing_layer_generation
    with _routing_merge_lock:
        _routing_layer_generation += 1
        _routing_merge_cache.clear()


def _routing_sidecar_key(config_path: str | None = None) -> tuple:
    """(path, mtime) for the sidecar, so an out-of-process edit invalidates."""
    try:
        path = get_routing_metadata_path(config_path)
        return (str(path), path.stat().st_mtime if path.exists() else 0.0)
    except Exception:  # noqa: BLE001 — routing must never fail on a stat
        return ("", 0.0)


def _config_routing_fingerprint(config: dict) -> tuple:
    """A stable key for the routing keys carried by *config* itself.

    config.json is an inbox that ``_migrate_config_routing_keys`` drains at
    startup, so after the first boot these keys are normally absent and this
    returns a constant — which is what keeps the common path free. Only a
    deployment that has not migrated, or one that had keys hand-added since,
    pays for serialising them.
    """
    parts = []
    for key in (*_ROUTING_LIST_KEYS, *_ROUTING_DICT_KEYS):
        val = config.get(key)
        if not val:
            continue
        try:
            parts.append((key, json.dumps(val, sort_keys=True, default=str)))
        except Exception:  # noqa: BLE001 — an unserialisable shape must not fail routing
            parts.append((key, repr(val)))
    return tuple(parts)


def _load_routing_sidecar(config_path: str | None = None) -> dict:
    """The learned layer, re-read whenever the file's mtime changes."""
    global _routing_sidecar_cache
    try:
        path = get_routing_metadata_path(config_path)
        mtime = path.stat().st_mtime if path.exists() else 0.0
    except Exception:  # noqa: BLE001 — routing must never fail on a stat
        return {}
    key = str(path)
    with _routing_sidecar_lock:
        cached = _routing_sidecar_cache
        if cached is not None and cached[0] == key and cached[1] == mtime:
            return cached[2]
    state = load_routing_metadata(config_path) if mtime else {}
    if not isinstance(state, dict):
        state = {}
    with _routing_sidecar_lock:
        _routing_sidecar_cache = (key, mtime, state)
    return state


def _reset_routing_sidecar_cache() -> None:
    """Drop the memoized sidecar. For tests and for post-refresh invalidation.

    Also invalidates the merged-config memo, which is derived from it: dropping
    one without the other would leave the merge serving the old sidecar's facts.
    """
    global _routing_sidecar_cache
    with _routing_sidecar_lock:
        _routing_sidecar_cache = None
    _bump_routing_generation()


# The two nested sources disagree about qualification, and always have.
# providers.json stores ids ALREADY qualified ("google/gemini-2.0-flash"), while
# the sidecar's by_provider stores them bare ("gemini-2.0-flash"). Prepending
# unconditionally — which is what both layers did — produced
# "google/google/gemini-2.0-flash" for every one of providers.json's 603
# entries, so the entire defaults layer matched nothing and the shipped data,
# the very thing the providers PR exists to maintain, never reached a routing
# decision.
#
# This is deliberately NOT solved by sniffing whether an id already starts with
# its provider name. Groq's own model ids carry a "groq/" vendor namespace, so
# providers.json holds "groq/groq/compound" while the sidecar holds
# "groq/compound" — the same string is qualified in one file and bare in the
# other, and no heuristic can tell them apart. Each layer states its own
# convention instead.


def _defaults_layer() -> dict:
    """The five keys as providers.json ships them, flattened to qualified ids.

    providers.json nests per provider; everything downstream keys on
    ``provider/model`` or a bare upstream id, so the nesting is flattened here
    once rather than at every lookup.
    """
    out: dict = {k: [] for k in _ROUTING_LIST_KEYS}
    out.update({k: {} for k in _ROUTING_DICT_KEYS})
    out[_ROUTING_PROVIDER_SCOPED] = set()
    try:
        for _provider, info in get_provider_free_info().items():
            # Already qualified in this file — used verbatim. An entry that is
            # not still resolves, because _lookup_model_fact tries the bare form
            # too; over-prefixing is the failure that matches nothing.
            #
            # Recorded as provider-scoped: this provider vouched for these ids,
            # and no other provider may inherit the claim by happening to serve
            # an upstream id that spells the same.
            for entry in info.get("believed_free") or []:
                if isinstance(entry, str):
                    out["believed_free"].append(entry.lower())
                    out[_ROUTING_PROVIDER_SCOPED].add(entry.lower())
            for key in ("model_reasoning", "model_capabilities", "free_limits"):
                for model, val in (info.get(key) or {}).items():
                    if isinstance(model, str):
                        out[key][model.lower()] = val
    except Exception as e:  # noqa: BLE001 — a bad sidecar must not break routing
        print(f"[server:_defaults_layer] {e}")
        traceback.print_exc()
    return out


def _learned_layer(config_path: str | None = None) -> dict:
    """The five keys as the refresh learned them.

    ``by_provider`` flattens like the defaults. ``by_model`` is keyed by
    normalized model, and its keys are carried through AS normalized keys rather
    than expanded to every provider id: one entry then answers for every
    provider serving those weights, which is the whole reason capabilities are
    keyed that way. The lookups below try the normalized form last.
    """
    out: dict = {k: [] for k in _ROUTING_LIST_KEYS}
    out.update({k: {} for k in _ROUTING_DICT_KEYS})
    out[_ROUTING_PROVIDER_SCOPED] = set()
    state = _load_routing_sidecar(config_path)
    try:
        for provider, info in (state.get("by_provider") or {}).items():
            if not isinstance(info, dict):
                continue
            # Bare in this file — always qualified with the provider that
            # observed them, which is what makes the same weights free on one
            # provider and metered on another.
            for key in _ROUTING_LIST_KEYS:
                for entry in info.get(key) or []:
                    if isinstance(entry, str):
                        qualified = f"{provider}/{entry}".lower()
                        out[key].append(qualified)
                        out[_ROUTING_PROVIDER_SCOPED].add(qualified)
            for model, val in (info.get("free_limits") or {}).items():
                out["free_limits"][f"{provider}/{model}".lower()] = val
        for model_key, facts in (state.get("by_model") or {}).items():
            if not isinstance(model_key, str) or not isinstance(facts, dict):
                continue
            caps = facts.get("capabilities")
            if isinstance(caps, list):
                out["model_capabilities"][model_key.lower()] = caps
            tier = facts.get("reasoning")
            if isinstance(tier, str):
                out["model_reasoning"][model_key.lower()] = tier
    except Exception as e:  # noqa: BLE001
        print(f"[server:_learned_layer] {e}")
        traceback.print_exc()
    return out


def _curated_layer(config: dict | None = None, config_path: str | None = None) -> dict:
    """The facts a person set by hand: the sidecar's curated section, over config.

    This is where config.json's five routing keys went. Their ORIGINAL shape is
    kept — flat lists and flat dicts, keyed however the user keyed them — rather
    than re-derived into per-provider buckets, so migrated data resolves through
    exactly the same lookup it always did and moving it cannot change a single
    routing decision.

    config.json is still READ, because a deployment that has not migrated yet
    has its intent recorded nowhere else and silently dropping it would be worse
    than the staleness this change set out to fix. It is an inbox rather than a
    layer: ``_migrate_config_routing_keys`` drains it into the curated section
    at startup, after which it holds none of these keys and contributes nothing.
    Anything added to it by hand afterwards is honoured until the next restart
    absorbs it. The curated section wins where both speak, so a correction made
    in the admin UI is never undone by a stale config.json.
    """
    layers: list[dict] = []
    if isinstance(config, dict):
        layers.append(config)
    state = _load_routing_sidecar(config_path)
    curated = state.get("curated")
    if isinstance(curated, dict):
        layers.append(curated)

    out: dict = {}
    for layer in layers:
        for key in _ROUTING_LIST_KEYS:
            val = layer.get(key)
            if isinstance(val, list):
                merged = dict.fromkeys(out.get(key) or [])
                merged.update(dict.fromkeys(v.lower() for v in val if isinstance(v, str)))
                out[key] = list(merged)
        for key in _ROUTING_DICT_KEYS:
            val = layer.get(key)
            if isinstance(val, dict):
                acc = dict(out.get(key) or {})
                acc.update({k.lower(): v for k, v in val.items() if isinstance(k, str)})
                out[key] = acc
    return out


def _listing_layer() -> dict:
    """What each provider currently says its own models can do.

    Above the learned layer because a gateway is authoritative about its own
    deployment, and below config because a hand correction outranks anything
    discovered. Capabilities only — a listing says nothing about price or quota.
    """
    caps = _get_model_capability_snapshot()
    return {"model_capabilities": {k: sorted(v) for k, v in caps.items() if v}}


# The layers, weakest first, named for display. The admin UI needs to answer
# "why is this model tagged that way", which a merged dict cannot: it keeps
# values and discards where each came from.
ROUTING_LAYER_NAMES = ("providers.json", "learned", "listing", "curated")


def routing_layers(config: dict | None = None,
                   config_path: str | None = None) -> list[tuple[str, dict]]:
    """Each routing-metadata layer, weakest first, paired with its name."""
    return list(zip(ROUTING_LAYER_NAMES, (
        _defaults_layer(),
        _learned_layer(config_path),
        _listing_layer(),
        _curated_layer(config, config_path),
    ), strict=True))


def routing_fact_sources(
    model_id: str, provider_name: str = "", config: dict | None = None,
    config_path: str | None = None,
    layers: list[tuple[str, dict]] | None = None,
) -> dict[str, list[str]]:
    """Which layers have something to say about *model_id*, per fact.

    Returns ``{fact_key: [layer_name, ...]}`` weakest first. For capabilities
    every contributing layer is listed, because they union; for the
    single-valued facts the LAST name is the one in effect.

    *layers* lets a caller building many rows assemble them once. Rebuilding
    four layers per row is most of the cost of listing a few thousand models.
    """
    out: dict[str, list[str]] = {}
    upstream = model_id.split("/", 1)[1] if "/" in model_id and provider_name else model_id
    forms = set(_model_fact_keys(provider_name or model_id.split("/", 1)[0], upstream))
    forms.add(model_id.lower())
    for name, layer in (layers if layers is not None else routing_layers(config, config_path)):
        for key in _ROUTING_LIST_KEYS:
            entries = layer.get(key)
            if isinstance(entries, list) and forms & {e.lower() for e in entries
                                                      if isinstance(e, str)}:
                out.setdefault(key, []).append(name)
        for key in _ROUTING_DICT_KEYS:
            mapping = layer.get(key)
            if isinstance(mapping, dict) and forms & set(mapping):
                out.setdefault(key, []).append(name)
    return out


def learned_fact_grades(model_key: str, config_path: str | None = None) -> dict[str, str]:
    """The provenance grade recorded beside each learned fact for *model_key*.

    ``{"capabilities": "family", "reasoning": "inferred"}`` and so on. This is
    what distinguishes a reading from a guess within the learned layer, which
    the layer name alone cannot express.
    """
    state = _load_routing_sidecar(config_path)
    facts = (state.get("by_model") or {}).get(model_key)
    if not isinstance(facts, dict):
        return {}
    return {
        fact: facts[f"{fact}_source"]
        for fact in ("capabilities", "reasoning")
        if isinstance(facts.get(f"{fact}_source"), str)
    }


def _merged_routing_config(config: dict, *, include_curated: bool = True) -> dict:
    """*config* with the five routing-metadata keys resolved across all layers.

    ``include_curated=False`` returns what every layer BELOW the hand-set one
    says. The admin editors need that to tell an edit from an unchanged value:
    a whole-section save would otherwise copy the entire learned layer into the
    curated one, freezing today's guesses as permanent hand corrections that no
    later refresh could improve.

    Returns a shallow copy carrying merged values for those five keys only, so
    every other consumer of the config dict is untouched. With no sidecar and no
    defaults this is exactly the config that was passed in, which is what makes
    an un-migrated deployment behave bit-for-bit as it did before.
    """
    cache_key = (
        include_curated,
        _routing_sidecar_key(),
        _routing_layer_generation,
        _config_routing_fingerprint(config),
    )
    with _routing_merge_lock:
        hit = _routing_merge_cache.get(cache_key)
    if hit is not None:
        return hit

    try:
        # config.json is deliberately NOT a layer. It is the file a person hand
        # edits, and machine processes were writing it too — the local-model
        # sync tagged every model a local provider served and saved it back —
        # so it could never be a stable record of intent. Hand-set facts live in
        # the sidecar's `curated` section instead, which occupies the same top
        # position and is what the admin UI writes.
        layers = [_defaults_layer(), _learned_layer(), _listing_layer()]
        if include_curated:
            layers.append(_curated_layer(config))
        layers = tuple(layers)
        merged = dict(config)
        # Provenance rides alongside the five keys rather than being one of
        # them. config.json is not a layer here, so anything a person wrote by
        # hand is absent from this set and keeps matching both ways.
        scoped: set[str] = set()
        for layer in layers:
            scoped |= layer.get(_ROUTING_PROVIDER_SCOPED) or set()
        merged[_ROUTING_PROVIDER_SCOPED] = scoped
        for key in _ROUTING_LIST_KEYS:
            seen: dict[str, None] = {}
            for layer in layers:
                for entry in layer.get(key) or []:
                    if isinstance(entry, str):
                        seen[entry.lower()] = None
            merged[key] = list(seen)
        for key in _ROUTING_DICT_KEYS:
            acc: dict = {}
            union = key in _ROUTING_UNION_KEYS
            for layer in layers:
                raw = layer.get(key)
                if not isinstance(raw, dict):
                    continue
                for k, v in raw.items():
                    if not isinstance(k, str):
                        continue
                    k = k.lower()
                    # Capabilities are a SET belonging to the weights, so layers
                    # add to each other. Replacing would let a gateway that
                    # publishes a thin `supported_parameters` retract what the
                    # catalog or another provider asserted about the same model.
                    # A reasoning tier and a rate limit are single-valued, so
                    # for those the higher layer rightly wins outright.
                    if union and isinstance(v, list) and isinstance(acc.get(k), list):
                        seen = dict.fromkeys(acc[k])
                        seen.update(dict.fromkeys(v))
                        acc[k] = list(seen)
                    else:
                        acc[k] = v
            merged[key] = acc
        with _routing_merge_lock:
            # Bounded purely by how many distinct inputs exist at once, which is
            # one per generation in practice; a stale generation's entries are
            # dropped wholesale by _bump_routing_generation.
            _routing_merge_cache[cache_key] = merged
        return merged
    except Exception as e:  # noqa: BLE001 — never fail a request over a merge
        print(f"[server:_merged_routing_config] {e}")
        traceback.print_exc()
        return config


def _normalized_believed_free(config: dict) -> set[str]:
    """
    Return a lowercased set of valid `believed_free` entries from *config*.

    Defensive against user-edited config.json: a missing field, ``None``,
    a non-list value, or non-string entries never raise — invalid shapes
    are logged once per call and silently dropped so a typo in config.json
    cannot turn /v1/models or /v1/chat/completions into a 500.
    """
    raw = _merged_routing_config(config).get("believed_free")
    if raw is None:
        return set()
    if not isinstance(raw, list):
        logger.warning(
            "config['believed_free'] must be a list of strings; got %s — ignoring.",
            type(raw).__name__,
        )
        return set()
    valid: set[str] = set()
    bad_summary: list[tuple[int, str]] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, str):
            valid.add(entry.lower())
        else:
            bad_summary.append((index, type(entry).__name__))
    if bad_summary:
        logger.warning(
            "config['believed_free'] contains %d non-string entr%s (ignored) at index/type: %s",
            len(bad_summary),
            "y" if len(bad_summary) == 1 else "ies",
            ", ".join(f"{i}:{t}" for i, t in bad_summary),
        )
    return valid


def _provider_scoped_ids(config: dict) -> set[str]:
    """Free-tier ids that a provider declared, rather than a person.

    Used to reject a BARE match on them: see ``_is_model_free_with``.
    """
    raw = _merged_routing_config(config).get(_ROUTING_PROVIDER_SCOPED)
    return raw if isinstance(raw, set) else set()


def _normalized_cost_observed(config: dict) -> set[str]:
    """Lowercased set of config['cost_observed_free_tier'] qualified ids.

    These are models that served a request reporting a real cost while marked
    free; they are treated as paid everywhere from that moment on. Defensive
    against malformed config in the same spirit as _normalized_believed_free.
    """
    raw = _merged_routing_config(config).get(COST_OBSERVED_KEY)
    if not isinstance(raw, list):
        return set()
    return {e.lower() for e in raw if isinstance(e, str)}


# An entry of this shape in `cost_observed_free_tier` marks the whole provider
# as paid rather than one model on it.
_COST_OBSERVED_WILDCARD = "/*"


def _cost_observed_providers(config: dict) -> set[str]:
    """Providers marked entirely paid, via a ``<provider>/*`` entry.

    Aggregators and resellers are the case this exists for. They re-serve other
    vendors' upstream ids, so a provider with no free tier of its own still
    collects free-tier beliefs written about those ids elsewhere, and listing
    its catalog model by model in ``cost_observed_free_tier`` is both tedious
    and permanently out of date. One entry covers the provider, including
    models it has not shipped yet.

    Only the explicit ``provider/*`` spelling is recognised. A bare provider
    name is not, because it is indistinguishable from an unqualified model id,
    which is a meaningful entry in these lists.
    """
    out: set[str] = set()
    for entry in _normalized_cost_observed(config):
        if entry.endswith(_COST_OBSERVED_WILDCARD):
            name = entry[: -len(_COST_OBSERVED_WILDCARD)]
            if name:
                out.add(name)
    return out


def _is_cost_observed(provider_name: str, upstream_id: str, config: dict) -> bool:
    """True when this model has been observed reporting a cost at runtime."""
    if provider_name.lower() in _cost_observed_providers(config):
        return True
    return f"{provider_name}/{upstream_id}".lower() in _normalized_cost_observed(config)


def _is_model_free_with(
    provider_name: str,
    upstream_id: str,
    believed_free: set[str],
    cost_observed: set[str],
    provider_scoped: frozenset[str] | set[str] = frozenset(),
    cost_observed_providers: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """``_is_model_free`` against sets the caller already has.

    The loops that walk every route need this. Deriving the two sets costs a
    full routing-config merge, and doing it per route turned assembling one
    candidate list into thousands of merges. The merge is memoized now, so this
    is no longer the difference between seconds and minutes — but a hot loop
    should not depend on a cache being warm to avoid being quadratic, and
    hoisting the invariant out is simply the right shape.
    """
    qualified = f"{provider_name}/{upstream_id}".lower()
    # A whole provider marked paid outranks every belief about its models,
    # including an id that literally spells "free". An aggregator that bills
    # for everything routinely re-serves ":free"-suffixed ids from upstreams
    # that really are free, and the suffix says nothing about what THIS
    # provider charges.
    if provider_name.lower() in cost_observed_providers:
        return False
    if qualified in cost_observed:
        return False
    uid = upstream_id.lower()
    if "free" in uid or qualified in believed_free:
        return True
    # A BARE match is only honoured when the entry was written by a person, who
    # meant "this model, wherever I have it". An entry a provider template
    # declared is scoped to that provider, and matching it bare is how one
    # vendor's free tier leaked onto a gateway that re-serves the identical
    # upstream id and bills for it. A qualified match above is unambiguous and
    # is always honoured.
    return uid in believed_free and uid not in provider_scoped


def _is_model_free(provider_name: str, upstream_id: str, config: dict) -> bool:
    """True when a model is treated as free-tier: its upstream id contains 'free'
    or it appears (bare or provider-qualified) in config['believed_free'].

    A model in config['cost_observed_free_tier'] is never free — a real cost was
    seen for it at runtime, so it is excluded here even if its id contains 'free'
    or it lingers in believed_free. This makes /free avoid it immediately.

    Shared by the /free candidate selector and the runtime cost flagger so both
    agree on what "free" means. A caller in a loop should hoist the two sets out
    and use ``_is_model_free_with`` instead.
    """
    return _is_model_free_with(
        provider_name, upstream_id,
        _normalized_believed_free(config), _normalized_cost_observed(config),
        _provider_scoped_ids(config), _cost_observed_providers(config),
    )


def _get_free_model_candidates() -> list[tuple[str, dict, str]]:
    """(provider_name, provider_cfg, upstream_model) for every model whose upstream ID contains 'free' or appears in config['believed_free'].

    Models served from a localhost / loopback URL are NEVER included — they
    route via the dedicated llmproxy__local family instead. This is a
    defence-in-depth guard against stale configs that still have local models
    in believed_free; the startup local-sync cleans those up too, but this
    runtime filter ensures /free never leaks a local model even before sync runs.
    """
    config = load_config()
    # Hoisted: both sets are invariant across the walk, and deriving either one
    # costs a full routing-config merge.
    believed_free = _normalized_believed_free(config)
    cost_observed = _normalized_cost_observed(config)
    provider_scoped = _provider_scoped_ids(config)
    paid_providers = _cost_observed_providers(config)
    candidates = []
    for provider_name, upstream_id in _get_distinct_routes():
        provider_cfg = get_provider(config, provider_name)
        if not provider_cfg:
            continue
        if not _provider_exposes_to_virtual_models(provider_cfg):
            continue
        # Skip local providers — they belong to the /local family, not /free.
        if _is_local_url(provider_base_url(provider_cfg)):
            continue
        if _is_model_free_with(provider_name, upstream_id, believed_free,
                               cost_observed, provider_scoped, paid_providers):
            candidates.append((provider_name, provider_cfg, upstream_id))
    return candidates


# — free-limits config parsing —

def _get_normalized_free_limits(config: dict) -> dict[str, dict]:
    """
    Return config['free_limits'] with all string keys lowercased.
    Ignores missing, non-dict, or malformed top-level values without raising.
    """
    raw = _merged_routing_config(config).get("free_limits")
    if not isinstance(raw, dict):
        if raw is not None:
            logger.warning(
                "config['free_limits'] must be a dict; got %s — ignoring.",
                type(raw).__name__,
            )
        return {}
    result: dict[str, dict] = {}
    for key, val in raw.items():
        if isinstance(key, str) and isinstance(val, dict) and not key.startswith("_"):
            result[key.lower()] = val
    return result


def _capacity_score(
    used_minute: int,
    used_day: int,
    limits: dict,
    used_tokens_minute: int = 0,
    used_tokens_day: int = 0,
) -> float:
    """
    Return a capacity score in [0.0, 1.0]: higher = more remaining headroom.
    Returns 1.0 (neutral) when no rpm/rpd/tpm/tpd limit is configured.
    Returns 0.0 when any configured limit is at or exceeded.

    Token limits (tpm/tpd) are enforced the same way as request limits using
    the tokens consumed by prior requests in the sliding/day windows. Configs
    without token limits are unaffected (the token terms simply don't apply).
    """
    rpm = limits.get("requests_per_minute")
    rpd = limits.get("requests_per_day")
    tpm = limits.get("tokens_per_minute")
    tpd = limits.get("tokens_per_day")
    if not rpm and not rpd and not tpm and not tpd:
        return 1.0
    scores: list[float] = []
    if rpm and rpm > 0:
        scores.append(max(0.0, (rpm - used_minute) / rpm))
    if rpd and rpd > 0:
        scores.append(max(0.0, (rpd - used_day) / rpd))
    if tpm and tpm > 0:
        scores.append(max(0.0, (tpm - used_tokens_minute) / tpm))
    if tpd and tpd > 0:
        scores.append(max(0.0, (tpd - used_tokens_day) / tpd))
    return min(scores) if scores else 1.0


# — "local" candidate selector —

def _is_local_url(base_url: str) -> bool:
    """Return True when *base_url* resolves to a local host or local-network domain.

    Matches:
      - loopback: localhost, 127.x.x.x, ::1, 0.0.0.0
      - mDNS / Bonjour: *.local
      - Docker host routing: host.docker.internal, gateway.docker.internal
    """
    try:
        hostname = urllib.parse.urlparse(base_url).hostname or ""
    except Exception:
        return False
    hostname = hostname.strip("[]").lower()
    return (
        hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0",
                     "host.docker.internal", "gateway.docker.internal")
        or hostname.startswith("127.")
        or hostname.endswith(".local")
    )


def _get_local_model_candidates() -> list[tuple[str, dict, str]]:
    """(provider_name, provider_cfg, upstream_model) for every model whose provider base_url is localhost."""
    config = load_config()
    candidates = []
    for provider_name, upstream_id in _get_distinct_routes():
        provider_cfg = get_provider(config, provider_name)
        if not provider_cfg:
            continue
        if not _provider_exposes_to_virtual_models(provider_cfg):
            continue
        if _is_local_url(provider_base_url(provider_cfg)):
            candidates.append((provider_name, provider_cfg, upstream_id))
    return candidates


# — reasoning-level candidate selectors —

def _get_model_reasoning(config: dict) -> dict[str, str]:
    """Return the model_reasoning map from config, with keys and values lowercased."""
    raw = _merged_routing_config(config).get("model_reasoning")
    if not isinstance(raw, dict):
        if raw is not None:
            logger.warning(
                "config['model_reasoning'] must be a dict; got %s — ignoring.",
                type(raw).__name__,
            )
        return {}
    result: dict[str, str] = {}
    assignable = [lvl for lvl in _REASONING_LEVELS
                  if lvl not in _OVERLAY_REASONING_LEVELS]
    for key, val in raw.items():
        if isinstance(key, str) and isinstance(val, str) and val.lower() in assignable:
            result[key.lower()] = val.lower()
        elif isinstance(val, str) and val.lower() in _OVERLAY_REASONING_LEVELS:
            logger.warning(
                "config['model_reasoning']: %r is a computed tier and cannot be set by "
                "hand (entry %r) — skipping. Membership lives in config['flagship_models'].",
                val, key,
            )
        else:
            logger.warning(
                "config['model_reasoning']: invalid entry %r: %r (level must be one of %s) — skipping.",
                key, val, "/".join(assignable),
            )
    return result


def _get_flagship_models(config: dict | None = None,
                         config_path: str | None = None) -> set[str]:
    """Lowercased set of qualified ids in the flagship tier.

    Flagship is an overlay rather than a value in ``model_reasoning``: a member
    keeps whatever tier tag it carries there, so promoting a model does not
    remove it from ``llmproxy/deep``.

    Membership is never hardcoded and never committed. It depends on which
    providers this deployment has configured and what each currently serves, so
    it is computed locally and cached in ``flagship_models.json`` beside
    config.json, alongside the other machine-managed state files. What lives in
    the user's config is only the policy: ``flagship_tier.pin`` and
    ``.exclude``.

    Pins are applied here rather than only at refresh time so that pinning a
    model takes effect immediately instead of on the next cadence tick, and
    excludes are applied last so they always win.

    Entries are qualified ``provider/model`` ids, because free-tier status and
    availability are per-provider: the same weights may be free on one provider
    and paid on another, and each provider's instance is its own routing target.
    """
    cfg = config if config is not None else load_config()
    members: set[str] = set()

    state = load_flagship_state(config_path)
    raw = state.get("members")
    if isinstance(raw, list):
        members |= {m.lower() for m in raw if isinstance(m, str)}
    elif raw is not None:
        logger.warning(
            "flagship_models.json: 'members' must be a list; got %s — ignoring.",
            type(raw).__name__,
        )

    tier_cfg = flagship_tier_cfg(cfg)
    # Entries may be bare strings or {"name": ..., "percentile": ...} objects,
    # freely mixed; parse_pins normalises both to names. Before it did, a dict
    # entry was silently dropped here, so a placed pin would not have joined the
    # tier at all.
    from .flagship import parse_pins
    members |= set(parse_pins(tier_cfg.get("pin")))
    exclude = tier_cfg.get("exclude")
    if isinstance(exclude, list):
        members -= {m.lower() for m in exclude if isinstance(m, str)}
    return members


def _warn_flagship_unranked(model_full: str) -> None:
    """Say once that a flagship pool is being served without its ranking.

    Serving unranked is a correct degradation, not a failure, so it must not
    warn per request. But it is also invisible: the only other signal is the
    route reason on the response, and an operator has no reason to look at that
    until something already seems wrong. One line naming the cause and the cure
    turns a silent week of the old ordering into something noticed on the first
    request.
    """
    global _flagship_unranked_warned
    if _flagship_unranked_warned:
        return
    _flagship_unranked_warned = True
    logger.warning(
        "[flagship] %s is being served UNRANKED: no benchmark scores in "
        "flagship_models.json yet, so the pool falls back to its previous "
        "ordering. The refresh recomputes them on its own cadence; set "
        "flagship_tier.refresh_frequency_days to 0 to force it now.",
        model_full,
    )


def _get_flagship_scores(config_path: str | None = None) -> dict[str, float]:
    """Combined benchmark percentile per flagship routing target, from the cache.

    Keyed by BOTH the lowercased qualified ``provider/model`` id and the
    normalised model key, mirroring the dual lookup ``_get_flagship_models``'
    callers already do. The two key spaces cannot collide: ``normalize_model_id``
    strips everything outside ``[a-z0-9]``, so a model key never contains a "/".

    The model-key entries are what let a pinned provider, or a routing target
    that appeared after the last refresh, inherit the score of the same weights
    scored elsewhere. A score is a property of the *model*, not of the provider
    serving it, so the same weights rank identically wherever they are served.

    Returns ``{}`` for a cache file written before scores were persisted, which
    is what makes a stale cache degrade to the previous ordering rather than
    sorting every candidate as unscored.
    """
    state = load_flagship_state(config_path)
    out: dict[str, float] = {}

    def _coerce(raw) -> float | None:
        # Two shapes are accepted per entry: the {"combined": float, ...} the
        # refresh writes, and a bare number, so a hand-edited or future cache
        # file costs an ordering rather than a request.
        if isinstance(raw, dict):
            raw = raw.get("combined")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        return float(raw)

    for key in ("model_scores", "scores"):
        block = state.get(key)
        if block is None:
            continue
        if not isinstance(block, dict):
            logger.warning(
                "flagship_models.json: %r must be a dict; got %s — ignoring.",
                key, type(block).__name__,
            )
            continue
        for ident, raw in block.items():
            if not isinstance(ident, str):
                continue
            value = _coerce(raw)
            if value is not None:
                out[ident.lower()] = value
    return _apply_pin_percentiles(out)


# Latched per (target, measured, pinned) so an override that displaces a real
# score is said once rather than on every request.
_pin_override_warned: set[tuple[str, float, float]] = set()


def _apply_pin_percentiles(scores: dict[str, float],
                           config: dict | None = None) -> dict[str, float]:
    """Overlay the placements named by ``flagship_tier.pin`` onto *scores*.

    Applied at READ time rather than only at refresh, so editing a pin's
    percentile takes effect on the next request instead of on the next cadence
    tick — matching how pins and excludes already behave in
    ``_get_flagship_models``.

    A pin wins over a measured score. That is what makes the field able to
    demote a model you distrust as well as promote one nothing has scored, and
    it matches pins already bypassing the bar and the spec veto. Because the
    override is otherwise invisible, displacing real evidence is logged once.

    Pins resolve against the LIVE route cache by the same precedence
    ``select_flagship`` uses — an exact qualified id first, a bare upstream id
    second — rather than by writing a normalised key and hoping. That is not a
    refinement: ``normalize_model_id`` strips the provider, so
    "atria-asi/Atria-Dawn-Preview" and "Atria-Dawn-Preview" normalise to the
    SAME key. Placing a qualified pin through that key would silently move
    every provider serving those weights, which is precisely what naming the
    provider was meant to prevent.
    """
    try:
        from .flagship import parse_pins
        cfg = config if config is not None else load_config()
        pins = parse_pins(flagship_tier_cfg(cfg).get("pin"))
    except Exception as e:  # noqa: BLE001 — a bad pin must not break routing
        print(f"[server:_apply_pin_percentiles] {e}")
        traceback.print_exc()
        return scores

    if not any(v is not None for v in pins.values()):
        return scores  # no placements: nothing to do, and no route walk

    known: set[str] = set()
    by_upstream: dict[str, set[str]] = {}
    try:
        for provider_name, upstream_id in _get_distinct_routes():
            qualified = f"{provider_name}/{upstream_id}".lower()
            known.add(qualified)
            by_upstream.setdefault(upstream_id.lower(), set()).add(qualified)
    except Exception as e:  # noqa: BLE001
        print(f"[server:_apply_pin_percentiles] route walk failed: {e}")
        traceback.print_exc()

    for name, percentile in pins.items():
        if percentile is None:
            continue  # a pin with no placement keeps today's behaviour
        if name in known:
            targets = {name}
        elif name in by_upstream:
            targets = by_upstream[name]
        else:
            # Unresolvable against anything this deployment serves — most often
            # a cold route cache on the first request after a restart. Honoured
            # literally, which is right for a qualified pin and a no-op for a
            # bare one until the cache warms.
            targets = {name}
        for target in targets:
            measured = scores.get(target)
            if measured is not None and measured != percentile:
                token = (target, measured, percentile)
                if token not in _pin_override_warned:
                    _pin_override_warned.add(token)
                    logger.info(
                        "[flagship] pin places %s at %.3f, overriding its "
                        "measured percentile of %.3f",
                        target, percentile, measured,
                    )
            scores[target] = percentile
    return scores


def _get_reasoning_model_candidates(level: str) -> list[tuple[str, dict, str]]:
    """(provider_name, provider_cfg, upstream_model) for every model in *level*.

    For the ordinary tiers this is an equality match on ``model_reasoning``. For
    an overlay tier (flagship) it is membership in the computed set instead, so
    the ordinary tiers keep every model they had.

    Either way the route cache is walked per provider, so a model served by
    several providers yields one candidate each — which is what gives failover
    something to fail over to.
    """
    config = load_config()
    overlay = level in _OVERLAY_REASONING_LEVELS
    flagship = _get_flagship_models(config) if overlay else set()
    reasoning = {} if overlay else _get_model_reasoning(config)
    candidates = []
    for provider_name, upstream_id in _get_distinct_routes():
        qualified = f"{provider_name}/{upstream_id}".lower()
        if overlay:
            matched = qualified in flagship or upstream_id.lower() in flagship
        else:
            matched = _lookup_model_fact(
                reasoning, provider_name, upstream_id) == level
        if matched:
            provider_cfg = get_provider(config, provider_name)
            if not provider_cfg:
                continue
            if not _provider_exposes_to_virtual_models(provider_cfg):
                continue
            candidates.append((provider_name, provider_cfg, upstream_id))
    return candidates


def _get_reasoning_free_candidates(level: str) -> list[tuple[str, dict, str]]:
    """Candidates that are both tagged *level* AND qualify as free."""
    reasoning_set = {(pn, um) for pn, _, um in _get_reasoning_model_candidates(level)}
    return [(pn, pc, um) for pn, pc, um in _get_free_model_candidates() if (pn, um) in reasoning_set]


def _get_reasoning_local_candidates(level: str) -> list[tuple[str, dict, str]]:
    """Candidates that are both tagged *level* AND served from localhost."""
    reasoning_set = {(pn, um) for pn, _, um in _get_reasoning_model_candidates(level)}
    return [(pn, pc, um) for pn, pc, um in _get_local_model_candidates() if (pn, um) in reasoning_set]


# — capability candidate selectors —

def _get_capability_model_candidates(cap: str) -> list[tuple[str, dict, str]]:
    """(provider, cfg, upstream) for every model tagged with capability *cap*."""
    config = load_config()
    cap_map = _model_capabilities(config)
    candidates = []
    for provider_name, upstream_id in _get_distinct_routes():
        if _model_has_capability(provider_name, upstream_id, cap, cap_map):
            provider_cfg = get_provider(config, provider_name)
            if not provider_cfg:
                continue
            if not _provider_exposes_to_virtual_models(provider_cfg):
                continue
            candidates.append((provider_name, provider_cfg, upstream_id))
    return candidates


def _get_capability_free_candidates(cap: str) -> list[tuple[str, dict, str]]:
    """Candidates that both have capability *cap* AND qualify as free-tier."""
    cap_set = {(pn, um) for pn, _, um in _get_capability_model_candidates(cap)}
    return [(pn, pc, um) for pn, pc, um in _get_free_model_candidates() if (pn, um) in cap_set]


# — loadbalanced (cost-tiered) candidate selector + ordering —

# Cost tiers, lowest = preferred. Free cloud is tried before local (also $0) so
# local compute is reserved for when no free cloud capacity is left; paid is the
# last resort.
_TIER_FREE, _TIER_LOCAL, _TIER_PAID = 0, 1, 2


def _is_loadbalanced_model(model_full: str) -> bool:
    """True when *model_full* is the cost-tiered loadbalanced virtual model."""
    return model_full in _LOADBALANCED_MODELS


def _get_loadbalanced_candidates() -> list[tuple[str, dict, str]]:
    """(provider, cfg, upstream) for every virtual-eligible model in the route cache.

    This is the FULL pool — free, local, and paid. Cost tiering happens at
    ordering time in _loadbalanced_ordered_candidates, not here, so a request can
    fail over down the waterfall when an upper tier is exhausted or unsuitable.
    """
    config = load_config()
    candidates = []
    for provider_name, upstream_id in _get_distinct_routes():
        provider_cfg = get_provider(config, provider_name)
        if not provider_cfg:
            continue
        if not _provider_exposes_to_virtual_models(provider_cfg):
            continue
        candidates.append((provider_name, provider_cfg, upstream_id))
    return candidates


def _provider_free_allowance(provider_cfg: dict) -> dict | None:
    """Return a provider's ``free_allowance`` as a {rpm,rpd,tpm,tpd} dict, or None.

    Best-effort: a provider MAY advertise a provider-wide free quota/session that
    applies on top of its explicitly-free models. Missing field, non-dict, or all
    malformed values → None (no provider-wide allowance to claim). Bools are
    rejected (``True`` is an int subclass) so a stray flag never becomes a limit.
    """
    raw = provider_cfg.get("free_allowance")
    if not isinstance(raw, dict):
        return None
    out: dict[str, int | None] = {}
    has_any = False
    for k in ("requests_per_minute", "requests_per_day", "tokens_per_minute", "tokens_per_day"):
        v = raw.get(k)
        if isinstance(v, bool):
            v = None
        if isinstance(v, int) and v >= 0:
            out[k] = v
            has_any = True
        else:
            out[k] = None
    return out if has_any else None


def _provider_free_headroom(provider_name: str, provider_cfg: dict) -> bool:
    """True when *provider_name* still has provider-wide free-tier headroom now.

    Aggregates this provider's recent request/token usage across all of its
    cached models and compares it to the configured ``free_allowance`` via
    _capacity_score. Returns False when no allowance is configured (nothing to
    claim as free) or when it is exhausted in the current window. Best-effort:
    counters are per-worker, so this is "as far as we can tell in the moment".
    """
    allowance = _provider_free_allowance(provider_cfg)
    if allowance is None:
        return False
    used_min = used_day = used_tok_min = used_tok_day = 0
    for pn, upstream_id in _get_distinct_routes():
        if pn != provider_name:
            continue
        key = f"{pn}/{upstream_id}".lower()
        m, d = _get_usage_snapshot(key)
        tm, td = _get_token_snapshot(key)
        used_min += m
        used_day += d
        used_tok_min += tm
        used_tok_day += td
    return _capacity_score(used_min, used_day, allowance, used_tok_min, used_tok_day) > 0.0


def _cost_tier(provider_name: str, upstream_id: str, provider_cfg: dict, config: dict) -> int:
    """Classify a model into a cost tier: 0=free, 1=local, 2=paid.

    Local models are $0 but kept in their own tier so free *cloud* models are
    preferred first (local compute is reserved for when free cloud is exhausted).
    A non-local, non-``believed_free`` model counts as free (0) only while its
    provider still has ``free_allowance`` headroom right now; once exhausted it
    falls back to paid (2).
    """
    if _is_local_url(provider_base_url(provider_cfg)):
        return _TIER_LOCAL
    # A model observed reporting a cost is paid, full stop — never let the
    # provider's free-allowance headroom pull it back into the free tier.
    if _is_cost_observed(provider_name, upstream_id, config):
        return _TIER_PAID
    if _is_model_free(provider_name, upstream_id, config):
        return _TIER_FREE
    if _provider_free_headroom(provider_name, provider_cfg):
        return _TIER_FREE
    return _TIER_PAID


def _loadbalanced_ordered_candidates(
    candidates: list[tuple[str, dict, str]],
    payload: dict,
    config: dict,
) -> list[tuple[str, dict, str]]:
    """Order candidates as a cost waterfall: free → local → (paid, opt-in).

    Cost tier is the dominant (outer) key — a paid model is NEVER ordered before
    a free or local one, so cost-avoidance always wins. Within the $0 tiers
    candidates are ordered **best-first**: among free models that still have
    headroom the most sophisticated (see _quality_key) is tried first, with
    capacity as a tiebreak; local is likewise strongest-first. This keeps spend at
    ~$0 while elevating answer quality, rather than picking a weak free model just
    because the prompt is short. A final capability sort still pulls models that
    satisfy a *forced* tool/vision/JSON requirement to the front of each tier.

    **Paid is opt-in.** Paid models are dropped from the implicit waterfall unless
    ``server.allow_implicit_paid`` is true; they stay reachable only by direct
    ``provider/model`` name. With the gate off (the default) an exhausted free +
    local pool surfaces a clear 429/503 rather than silently spending money.
    """
    allow_paid = _allow_implicit_paid(config)
    tiers: dict[int, list[tuple[str, dict, str]]] = {
        _TIER_FREE: [], _TIER_LOCAL: [], _TIER_PAID: [],
    }
    for pn, pc, um in candidates:
        tiers[_cost_tier(pn, um, pc, config)].append((pn, pc, um))

    free_limits = _get_normalized_free_limits(config)
    pricing = load_pricing_map()
    needed = _needed_capabilities(payload)
    cap_map = _model_capabilities(config)
    reasoning_map = _get_model_reasoning(config)
    # Flagship is an overlay, so it is not in reasoning_map; fetch it once here
    # rather than per candidate.
    flagship_models = _get_flagship_models(config)

    def _price(c: tuple[str, dict, str]) -> float:
        pn, _pc, um = c
        prices = pricing.get(f"{pn}/{um}".lower()) or pricing.get(um.lower())
        if not prices:
            return float("inf")  # unknown price sorts last but is still tried
        return sum(prices)

    ordered: list[tuple[str, dict, str]] = []
    for tier in (_TIER_FREE, _TIER_LOCAL, _TIER_PAID):
        bucket = tiers[tier]
        if not bucket:
            continue
        if tier == _TIER_PAID and not allow_paid:
            continue  # paid never an implicit fallback unless explicitly enabled
        if tier == _TIER_FREE:
            bucket = _quality_ordered_candidates(
                bucket, free_limits, reasoning_map, flagship_models)
            bucket = _apply_favorite_free_ordering(bucket, config)
        elif tier == _TIER_LOCAL:
            # $0 like free — prefer the strongest local model (e.g. the larger
            # Ollama model) rather than rotating randomly.
            bucket = sorted(
                bucket,
                key=lambda c: _quality_key(c[0], c[2], reasoning_map, flagship_models),
                reverse=True,
            )
        else:
            # Paid: cost first, then sophistication as a tiebreak among equals.
            bucket = sorted(
                bucket,
                key=lambda c: (_price(c),
                               tuple(-x for x in _quality_key(
                                   c[0], c[2], reasoning_map, flagship_models))),
            )
            # Paid providers are the ones that actually bill prompt caching, so
            # this is where affinity pays for itself: keep a conversation on the
            # upstream that already holds its prefix. Applied only within the
            # paid bucket — the free tier wants spreading, not stickiness.
            bucket = _order_by_cache_affinity(bucket, _affinity_key(payload))
        if needed:
            bucket, _ = _apply_capability_gate(bucket, needed, cap_map, "loadbalanced")
        ordered.extend(bucket)
    return ordered


def _strip_virtual_prefix(model_full: str) -> str:
    """Strip the leading "llmproxy__" or legacy "llmproxy/" virtual-model prefix."""
    if model_full.startswith("llmproxy__"):
        return model_full[len("llmproxy__"):]
    if model_full.startswith("llmproxy/"):
        return model_full[len("llmproxy/"):]
    return model_full


# — per-provider virtual models —

def _split_per_provider_virtual(model_full: str) -> tuple[str, str] | None:
    """Recognise a per-provider virtual model "llmproxy__<provider>[/<dimension>]".

    Returns ``(provider_name, dimension)`` where *dimension* is "" for the bare
    aggregator form (llmproxy__<provider>) or one of ``_PER_PROVIDER_DIMENSIONS``,
    otherwise ``None``.

    Precedence rule: existing GLOBAL virtual names always win — if *model_full*
    is in ``_VIRTUAL_MODELS`` this returns ``None`` so the global selector handles
    it (100% backward compatible).  Only then is the leading token resolved as a
    provider, which must be configured, non-reserved, non-local, and not opted out
    of virtual exposure.
    """
    if not (model_full.startswith("llmproxy__") or model_full.startswith("llmproxy/")):
        return None
    # Existing global forms take precedence over any same-named provider.
    if model_full in _VIRTUAL_MODELS:
        return None
    name = _strip_virtual_prefix(model_full)
    if "/" in name:
        provider_name, dimension = name.split("/", 1)
        if dimension not in _PER_PROVIDER_DIMENSIONS:
            return None
    else:
        provider_name, dimension = name, ""
    if not provider_name or provider_name in RESERVED_PROVIDER_NAMES:
        return None
    config = load_config()
    provider_cfg = get_provider(config, provider_name)
    if not provider_cfg:
        return None
    if _is_local_url(provider_base_url(provider_cfg)):
        return None
    if not _provider_exposes_to_virtual_models(provider_cfg):
        return None
    return provider_name, dimension


def _is_per_provider_virtual(model_full: str) -> bool:
    """True when *model_full* is a recognised per-provider virtual model."""
    return _split_per_provider_virtual(model_full) is not None


def _is_virtual_model(model_full: str) -> bool:
    """True for any virtual model: a static global name OR a per-provider form."""
    return model_full in _VIRTUAL_MODELS or _is_per_provider_virtual(model_full)


def _is_fusion_model(model_full: str) -> bool:
    """True when *model_full* is a fusion virtual model (bare or /free)."""
    return model_full in _FUSION_VIRTUAL_MODELS


def _is_fusion_free_model(model_full: str) -> bool:
    """True when *model_full* is the free-pool fusion variant."""
    return model_full in ("llmproxy__fusion/free", "llmproxy/fusion/free")


def _is_free_virtual_model(model_full: str) -> bool:
    """True for capacity-aware free virtuals (global free set or <provider>/free)."""
    if model_full in _FREE_VIRTUAL_MODELS:
        return True
    split = _split_per_provider_virtual(model_full)
    return split is not None and split[1] == "free"


def _is_local_virtual_model(model_full: str) -> bool:
    """True for localhost-pool virtuals: the global local aggregator and the
    reasoning-level /local sub-virtuals.

    There is no per-provider <provider>/local form — "local" is not a per-provider
    dimension, and per-provider virtuals exclude localhost-backed providers — so
    membership in _LOCAL_VIRTUAL_MODELS is the complete test.
    """
    return model_full in _LOCAL_VIRTUAL_MODELS


def _is_flagship_virtual_model(model_full: str) -> bool:
    """True for any virtual backed by a benchmark-ranked overlay tier.

    Covers the global forms (``llmproxy__flagship`` and its ``/free`` and
    ``/local`` sub-virtuals, in both the new and legacy spellings) and the
    per-provider ``llmproxy__<provider>/flagship`` slice, which is a flagship
    pool narrowed to one provider and wants the same ordering.
    """
    if model_full in _FLAGSHIP_VIRTUAL_MODELS:
        return True
    split = _split_per_provider_virtual(model_full)
    return split is not None and split[1] in _OVERLAY_REASONING_LEVELS


def _get_provider_virtual_candidates(provider_name: str, dimension: str) -> list[tuple[str, dict, str]]:
    """Candidates for llmproxy__<provider>[/<dimension>], scoped to one provider.

    Reuses the matching global selector then filters to *provider_name*, so the
    local / expose / free guards inside each global selector are inherited.  The
    bare ("") form cycles through every cached model of the provider.
    """
    if dimension == "":
        config = load_config()
        provider_cfg = get_provider(config, provider_name)
        if not provider_cfg:
            return []
        return [
            (provider_name, provider_cfg, upstream_id)
            for pn, upstream_id in _get_distinct_routes()
            if pn == provider_name
        ]
    if dimension == "free":
        base = _get_free_model_candidates()
    elif dimension in _REASONING_LEVELS:
        base = _get_reasoning_model_candidates(dimension)
    elif dimension in _CAPABILITY_VIRTUALS:
        base = _get_capability_model_candidates(dimension)
    else:
        return []
    return [(pn, pc, um) for pn, pc, um in base if pn == provider_name]


def _get_virtual_candidates(model_full: str) -> list[tuple[str, dict, str]]:
    """Dispatch to the correct candidate selector for any virtual model name."""
    split = _split_per_provider_virtual(model_full)
    if split is not None:
        return _get_provider_virtual_candidates(*split)
    name = _strip_virtual_prefix(model_full)
    if name == "loadbalanced":
        return _get_loadbalanced_candidates()
    if name == "free":
        return _get_free_model_candidates()
    if name == "local":
        return _get_local_model_candidates()
    if name in _REASONING_LEVELS:
        return _get_reasoning_model_candidates(name)
    if name in _CAPABILITY_VIRTUALS:
        return _get_capability_model_candidates(name)
    for level in _REASONING_LEVELS:
        if name == f"{level}/free":
            return _get_reasoning_free_candidates(level)
        if name == f"{level}/local":
            return _get_reasoning_local_candidates(level)
    for cap in _CAPABILITY_VIRTUALS:
        if name == f"{cap}/free":
            return _get_capability_free_candidates(cap)
    return []


# ---------------------------------------------------------------------------
# Shared routing logic for all proxied endpoints
# ---------------------------------------------------------------------------

def _canonicalize_model_id(model_full: str, config: dict) -> str:
    """Map any client-supplied virtual or real-model id to the canonical ``provider__model`` form.

    **Virtual models** are advertised as ``llmproxy/model`` where any ``/`` inside the
    model part is encoded as ``__`` (e.g. ``llmproxy/deep__free``).  Inbound ids are
    accepted in all of these equivalent forms:

    * ``llmproxy/deep__free``   — new advertised form
    * ``llmproxy/deep/free``    — legacy slash form (pre-PR #88)
    * ``llmproxy__deep/free``   — canonical internal form
    * ``llmproxy__deep__free``  — ``__`` used everywhere (robust fuzzy match)

    All four resolve to the canonical ``llmproxy__deep/free`` that the virtual-model
    frozensets and routing use internally.

    **Real models** are advertised in canonical ``provider__model`` form.  The dual-keyed
    route cache means the raw id hits directly without any string manipulation.

    Resolution priority:
    1. Route cache hit → return unchanged (hot path, lossless for real models).
    2. ``/`` present and leading token is a provider or ``llmproxy`` → reverse the
       slash-form encoding (split on first ``/``, rewrite ``__`` → ``/`` in remainder).
    3. No ``/`` but starts with ``llmproxy__`` → fuzzy virtual match: compare each
       known virtual's suffix with ``__``/``/`` collapsed, return the match if found.
    4. Otherwise → return unchanged (foreign id or already canonical).
    """
    with _model_route_cache_lock:
        in_cache = model_full in _model_route_cache
    if in_cache:
        return model_full

    if "/" in model_full:
        left, _, rest = model_full.partition("/")
        if left == "llmproxy" or get_provider(config, left):
            return left + "__" + rest.replace("__", "/")
        return model_full

    # No "/" — check if it's an llmproxy virtual with "__" used where "/" is expected.
    if model_full.startswith("llmproxy__") and model_full not in _VIRTUAL_MODELS:
        suffix = model_full[len("llmproxy__"):]
        # Normalise: collapse "__" → "/" so we can compare against canonical suffixes.
        normalised = suffix.replace("__", "/")
        for vid in _VIRTUAL_MODELS:
            if not vid.startswith("llmproxy__"):
                continue
            vsuffix = vid[len("llmproxy__"):]
            if vsuffix.replace("__", "/") == normalised:
                return vid
    return model_full


def _resolve_provider(model_full: str) -> tuple[str | None, dict | None, str | None, Response | None]:
    """
    Parse *model_full* into (provider_name, provider_cfg, upstream_model).

    Returns a 4-tuple where the last element is an error Response if
    resolution fails, otherwise None.  Callers should check the last element
    before using the first three.
    """
    config = load_config()

    # Cache-first: display ID formats ("provider__model", and the legacy
    # "model__provider" / "model (provider)") are not parseable by
    # parse_model_string, so the cache (populated by /v1/models) is authoritative.
    with _model_route_cache_lock:
        cached_route = _model_route_cache.get(model_full)
    if not cached_route and "__" in model_full and "/" in model_full.partition("__")[2]:
        # Possible cold-cache flattened multi-slash display id
        # (e.g. "provider__sub_model/leaf"): the heuristic partition below cannot
        # losslessly recover the original upstream ("sub_model/leaf" vs the real
        # "sub/model/leaf"), because _flatten_display_model turned interior "/"
        # into "_". Rebuild the route cache once from the providers' /models
        # endpoints and retry the lookup, but only when the left token names a
        # configured provider so unknown/garbage ids never trigger upstream fetches.
        left_guess = model_full.partition("__")[0]
        if get_provider(config, left_guess):
            rebuild_providers = {
                k: v for k, v in config.get("providers", {}).items()
                if k not in RESERVED_PROVIDER_NAMES
            }
            rebuild_timeout = config.get("server", {}).get("request_timeout", 120)
            _rebuild_route_cache(rebuild_providers, rebuild_timeout)
            with _model_route_cache_lock:
                cached_route = _model_route_cache.get(model_full)
    if cached_route:
        provider_name, upstream_model = cached_route
    elif "__" in model_full:
        # Try the current "provider__model" form first (provider on the left).
        # If the left side isn't a configured provider, fall back to the legacy
        # "model__provider" form from PR #27 (provider on the right). If neither
        # side matches a known provider, keep the right-side-as-provider guess so
        # the downstream "Unknown provider" error message is unchanged.
        left, _, right = model_full.partition("__")
        if get_provider(config, left):
            provider_name, upstream_model = left, right
        else:
            left2, _, right2 = model_full.rpartition("__")
            provider_name, upstream_model = right2, left2
    elif model_full.endswith(")") and " (" in model_full:
        # Cold-cache fallback for legacy "model (provider)" format (backward compat).
        model_part, _, provider_name = model_full[:-1].rpartition(" (")
        upstream_model = model_part
    else:
        try:
            provider_name, upstream_model = parse_model_string(model_full)
        except ValueError as e:
            return None, None, None, _error(str(e), status=400)

    provider_cfg = get_provider(config, provider_name)
    if not provider_cfg:
        return None, None, None, _error(
            f"No provider named '{provider_name}' is configured. "
            f"Run 'llmproxy --setup' to add it.",
            status=404,
        )

    if not model_is_allowed(provider_cfg, upstream_model):
        return None, None, None, _error(
            f"Model '{upstream_model}' is not permitted by the filter "
            f"configured for provider '{provider_name}'.",
            status=403,
            code="model_not_allowed",
        )

    return provider_name, provider_cfg, upstream_model, None


# Where a routing tag actually comes from. The hints used to name
# config['model_reasoning'] and config['model_capabilities'], which sent anyone
# hitting an empty pool to a file that is no longer a routing layer — and, once
# the refresh had wiped a tier, to the one place that could not explain why.
_WHERE_TAGS_LIVE = (
    "Tags are learned into routing_metadata.json on the "
    "routing_metadata.refresh_frequency_days cadence, seeded from "
    "llmproxy/providers.json, and can be set by hand in the admin UI "
    "(Models tab) which records them as curated so no refresh undoes them."
)


def _virtual_model_hint(model_full: str) -> str:
    """Return a one-sentence config hint for an unavailable virtual model."""
    split = _split_per_provider_virtual(model_full)
    if split is not None:
        provider_name, dim = split
        if dim == "":
            return f"Provider '{provider_name}' has no models in the route cache; check its base_url and api_key."
        if dim == "free":
            return (
                f"Provider '{provider_name}' has no free-tier model "
                f"(upstream ID contains 'free', or mark one free in the admin UI). "
                f"{_WHERE_TAGS_LIVE}"
            )
        if dim in _REASONING_LEVELS:
            return (f"No model of provider '{provider_name}' is tagged '{dim}'. "
                    f"{_WHERE_TAGS_LIVE}")
        return (f"No model of provider '{provider_name}' is known to support "
                f"'{dim}'. {_WHERE_TAGS_LIVE}")
    name = _strip_virtual_prefix(model_full)
    if name == "loadbalanced":
        return "Check that at least one provider exposes any model to virtual routing."
    if name == "free":
        return (
            "Check that at least one provider exposes a free-tier model "
            "(upstream ID contains 'free', or mark one free in the admin UI). "
            + _WHERE_TAGS_LIVE
        )
    if name == "local":
        return "Check that at least one provider has a localhost base_url."
    for level in _REASONING_LEVELS:
        # Overlay tiers are computed, so "go tag a model" is the wrong advice.
        if level in _OVERLAY_REASONING_LEVELS:
            where = (f"config['{level}_models'], refreshed on "
                     f"config['{level}_tier'].refresh_frequency_days")
            if name == level:
                return (
                    f"No model currently qualifies for '{level}'. Membership is "
                    f"computed into {where}; pin one with "
                    f"config['{level}_tier'].pin to force it in."
                )
            if name == f"{level}/free":
                return (
                    f"No '{level}' model is currently free on any configured "
                    f"provider. Membership is computed into {where}; free status "
                    f"is per-provider, so the same model may be paid here and "
                    f"free elsewhere."
                )
            if name == f"{level}/local":
                return (
                    f"No '{level}' model is served by a localhost provider. "
                    f"Membership is computed into {where}."
                )
            continue
        if name == level:
            return (f"No model is tagged '{level}'. {_WHERE_TAGS_LIVE}")
        if name == f"{level}/free":
            return (
                f"Need a model tagged '{level}' in config['model_reasoning'] "
                f"that is also free-tier."
            )
        if name == f"{level}/local":
            return (
                f"Need a model tagged '{level}' in config['model_reasoning'] "
                f"that is also served by a localhost provider."
            )
    for cap in _CAPABILITY_VIRTUALS:
        if name == cap:
            return f"Tag at least one model with '{cap}' in config['model_capabilities']."
        if name == f"{cap}/free":
            return (
                f"Need a model tagged '{cap}' in config['model_capabilities'] "
                f"that is also free-tier."
            )
    return ""


# ---------------------------------------------------------------------------
# Fusion (multi-model deliberation) — see llmproxy/fusion.py for the pipeline
# ---------------------------------------------------------------------------

def _get_all_model_candidates() -> list[tuple[str, dict, str]]:
    """(provider_name, provider_cfg, upstream_model) for every non-local model
    from a virtual-exposing provider. The full pool a bare ``fusion`` panel
    draws from (subject to the allow_paid filter applied by the caller)."""
    config = load_config()
    out: list[tuple[str, dict, str]] = []
    for provider_name, upstream_id in _get_distinct_routes():
        provider_cfg = get_provider(config, provider_name)
        if not provider_cfg:
            continue
        if not _provider_exposes_to_virtual_models(provider_cfg):
            continue
        if _is_local_url(provider_base_url(provider_cfg)):
            continue
        out.append((provider_name, provider_cfg, upstream_id))
    return out


def _resolve_panel_list(panel_ids: list, config: dict) -> list[tuple[str, dict, str]]:
    """Resolve an explicit fusion.panel list of model ids to candidate tuples.

    Unresolvable entries are logged and skipped rather than failing the request.
    """
    out: list[tuple[str, dict, str]] = []
    seen: set[str] = set()
    for mid in panel_ids:
        if not isinstance(mid, str):
            continue
        pn, pc, uid, err = _resolve_provider(mid)
        if err is not None or pc is None:
            logger.warning("[fusion] panel entry %r could not be resolved; skipping.", mid)
            continue
        key = f"{pn}/{uid}"
        if key not in seen:
            seen.add(key)
            out.append((pn, pc, uid))
    return out


def _strip_tool_keys(payload: dict) -> dict:
    """Drop forced-output keys so panel/judge calls return plain text.

    The synthesizer call re-attaches the original tools/tool_choice/
    response_format so the user's forced-capability contract is still honored on
    the final answer; the panel and judge deliberate in text.
    """
    return {k: v for k, v in payload.items() if k not in ("tools", "tool_choice", "response_format")}


def _fusion_pool(model_full: str, config: dict, fcfg: dict, payload: dict, free: bool) -> list[tuple[str, dict, str]]:
    """Build the ordered candidate pool a fusion panel is selected from.

    For the /free variant the pool is the capacity-ordered free pool; for bare
    fusion it is an explicit fusion.panel (if set) or the full non-local pool,
    filtered to free models when allow_paid is false. When the request forces a
    capability (tools/json) and forced_capability is "restrict", the pool is
    narrowed to models carrying every needed capability; under "bypass" the pool
    is merely reordered capable-first.
    """
    if free:
        pool = _get_free_model_candidates()
        pool = _capacity_ordered_candidates(pool, _get_normalized_free_limits(config))
    else:
        explicit = fcfg.get("panel")
        if explicit:
            pool = _resolve_panel_list(explicit, config)
        else:
            pool = _get_all_model_candidates()
            if not fcfg.get("allow_paid", True):
                pool = [c for c in pool if _is_model_free(c[0], c[2], config)]
            pool = _cycling_candidates(pool)

    needed = _needed_capabilities(payload)
    if needed:
        cap_map = _model_capabilities(config)
        if fcfg.get("forced_capability") == "restrict":
            # "restrict" is the stricter opt-in: it additionally excludes models
            # that are merely UNPROVEN, not just ones disproven. The non-empty
            # floor matters more here than anywhere, because sparse metadata
            # could otherwise leave a forced-tools request with no panel at all.
            strict = [
                c for c in pool
                if all(_model_has_capability(c[0], c[2], cap, cap_map) for cap in needed)
            ]
            if strict:
                pool = strict
            else:
                logger.warning(
                    "[fusion] forced_capability=restrict matched no model for %s; "
                    "falling back to the capability gate so the panel is not empty",
                    "+".join(sorted(needed)),
                )
                pool, _ = _apply_capability_gate(pool, needed, cap_map, "fusion")
        else:  # "bypass": drop the disproven, then order capable-first
            pool, _ = _apply_capability_gate(pool, needed, cap_map, "fusion")
    return pool


def _rank_aux_models(
    pool: list[tuple[str, dict, str]],
    explicit: str | None,
    config: dict,
    prefer_caps: frozenset[str] = frozenset(),
    exclude_first: tuple[str, dict, str] | None = None,
) -> list[tuple[str, dict, str]]:
    """Rank judge/synthesizer candidate models in preference order.

    An explicit configured model leads when it resolves. Otherwise *pool* models
    tagged with any of *prefer_caps* (e.g. reasoning) come first, then the rest,
    with *exclude_first* (the model already chosen for the other stage) pushed
    last so the judge and synthesizer differ where possible. Callers try each in
    order — with per-account failover — until one answers, so a rate-limited
    judge/synth rotates to the next model instead of collapsing the pipeline.
    """
    ordered: list[tuple[str, dict, str]] = []
    seen: set[str] = set()

    def _add(c: tuple[str, dict, str]) -> None:
        key = f"{c[0]}/{c[2]}".lower()
        if key not in seen:
            seen.add(key)
            ordered.append(c)

    if explicit:
        pn, pc, uid, err = _resolve_provider(explicit)
        if err is None and pc is not None:
            _add((pn, pc, uid))
        else:
            logger.warning("[fusion] configured model %r unresolved; auto-picking.", explicit)

    if pool:
        cap_map = _model_capabilities(config)
        exclude_key = f"{exclude_first[0]}/{exclude_first[2]}".lower() if exclude_first else None

        def _has(c: tuple[str, dict, str]) -> bool:
            return (any(_model_has_capability(c[0], c[2], cap, cap_map) for cap in prefer_caps)
                    if prefer_caps else False)

        def _key(c: tuple[str, dict, str]) -> str:
            return f"{c[0]}/{c[2]}".lower()

        preferred = [c for c in pool if _has(c) and _key(c) != exclude_key]
        others = [c for c in pool if not _has(c) and _key(c) != exclude_key]
        excluded = [c for c in pool if _key(c) == exclude_key]
        for c in preferred + others + excluded:
            _add(c)
    return ordered


def _pick_aux_model(
    pool: list[tuple[str, dict, str]],
    explicit: str | None,
    config: dict,
    prefer_caps: frozenset[str] = frozenset(),
    exclude_first: tuple[str, dict, str] | None = None,
) -> tuple[str, dict, str] | None:
    """Return the single best judge/synthesizer model (see _rank_aux_models)."""
    ranked = _rank_aux_models(pool, explicit, config, prefer_caps, exclude_first)
    return ranked[0] if ranked else None


def _proxy_fusion(
    endpoint: str,
    model_full: str,
    payload: dict,
    config: dict,
    inbound_adapter,
    is_streaming: bool,
) -> Response:
    """Run the fusion pipeline: panel fan-out, judge, synthesis.

    See the module docstring of llmproxy/fusion.py for the four-step pipeline and
    the graceful-degradation policy. ``on_success`` accounting is recorded per
    upstream touched (each panel member, the judge, and the synthesizer), so a
    fusion request is costed like the several real requests it issues.
    """
    fcfg = _fusion.get_fusion_config(config)
    if fcfg.get("enabled") is False:
        return _error("Fusion is disabled (set config['fusion']['enabled'] = true).", status=404)
    if endpoint != "chat/completions":
        return _error("Fusion models are only available on chat/completions.", status=400)

    free = _is_fusion_free_model(model_full)
    server_cfg = config.get("server", {})
    timeout = server_cfg.get("request_timeout", 120)
    candidate_timeout = min(timeout, _VIRTUAL_CANDIDATE_TIMEOUT)

    pool = _fusion_pool(model_full, config, fcfg, payload, free)
    if len(pool) < _fusion.MIN_PANEL:
        return _error(
            f"No '{model_full}' panel available (need at least {_fusion.MIN_PANEL} "
            f"eligible models). " + _virtual_model_hint(model_full),
            status=503,
        )

    panel_cands = _fusion.select_panel(pool, fcfg["panel_size"], fcfg["diversity"] == "provider")
    original_messages = payload.get("messages", [])
    stripped = _strip_tool_keys(
        {k: v for k, v in payload.items() if k not in ("model", "stream", "stream_options")}
    )

    logger.info("  [fusion] %s panel of %d (free=%s)", model_full, len(panel_cands), free)

    # 1 + 2. Fan the prompt out to the panel in parallel (non-streaming).
    # The fan-out runs on ThreadPoolExecutor worker threads, which do not have a
    # Flask request context. Rather than copy the request context into each worker
    # — which is unsafe to reuse across the many ex.map rounds the backfill/retry
    # loops issue (the shared RequestContext's contextvars token stack corrupts,
    # raising "Token was created in a different Context") — capture the forwarded
    # client headers once here on the request thread and hand them to each call,
    # so the workers never touch ``request``.
    forwarded_headers = _forwarded_client_headers()

    def _call_panel(cand: tuple[str, dict, str]):
        pn, pc, uid = cand
        try:
            resp, acct = _call_with_account_failover(
                endpoint, pn, pc, {**stripped, "model": uid}, candidate_timeout,
                forwarded_headers=forwarded_headers,
            )
            return cand, resp, acct
        except Exception as e:  # noqa: BLE001
            print(f"[server:_proxy_fusion:panel] {pn}/{uid}: {e}")
            traceback.print_exc()
            return cand, None, None

    # One panel fan-out, with reserve backfill so a few transient upstream
    # failures (rate limits, blips) on the chosen members don't collapse the whole
    # panel — mirroring the resilient cycling of the plain /free route. ``reserve``
    # is the ordered pool minus the chosen members; each failed slot is retried
    # with the next reserve candidate until the pool is exhausted.
    def _run_panel(cands, candidate_pool):
        chosen_keys = {f"{c[0]}/{c[2]}" for c in cands}
        reserve = [c for c in candidate_pool if f"{c[0]}/{c[2]}" not in chosen_keys]
        entries: list[dict] = []
        used: list[str] = []
        failed: list[dict] = []
        success: list[tuple[tuple[str, dict, str], bytes]] = []
        pending = list(cands)
        while pending:
            with ThreadPoolExecutor(max_workers=min(len(pending), 8)) as ex:
                results = list(ex.map(_call_panel, pending))
            failures = 0
            for cand, resp, acct in results:
                pn, _pc, uid = cand
                key = f"{pn}/{uid}"
                if resp is not None and resp.status_code < 400:
                    body = resp.get_data()
                    text = _fusion.extract_message_text(body)
                    if text.strip():
                        _record_usage(pn, uid, usage=extract_usage(body), config=config, account_id=acct)
                        entries.append({"label": key, "content": text})
                        used.append(key)
                        success.append((cand, body))
                        continue
                    failed.append({"model": key, "reason": "empty response"})
                else:
                    status = resp.status_code if resp is not None else "exception"
                    failed.append({"model": key, "reason": f"status {status}"})
                failures += 1
            # Pull one replacement per failed slot from the reserve (if any remain).
            pending = [reserve.pop(0) for _ in range(min(failures, len(reserve)))]
        return entries, used, failed, success

    # When the panel is auto-selected (no explicit fusion.panel), a whole fan-out
    # that fails is retried a few times against a freshly re-derived pool: the free
    # and bare pools re-randomize their ordering each call, so select_panel lands
    # on a different mix of models (and re-attempts transiently-failed ones) before
    # the request gives up. An explicitly configured panel is honored as-is.
    explicit_panel = (not free) and bool(fcfg.get("panel"))
    attempts = 1 if explicit_panel else _fusion.PANEL_SELECTION_ATTEMPTS
    panel_entries: list[dict] = []
    panel_used: list[str] = []
    failed_models: list[dict] = []
    panel_success: list[tuple[tuple[str, dict, str], bytes]] = []
    for attempt in range(attempts):
        if attempt > 0:
            pool = _fusion_pool(model_full, config, fcfg, payload, free)
            panel_cands = _fusion.select_panel(
                pool, fcfg["panel_size"], fcfg["diversity"] == "provider"
            )
            logger.info(
                "  [fusion] %s panel retry %d/%d (fresh selection of %d)",
                model_full, attempt + 1, attempts, len(panel_cands),
            )
        panel_entries, panel_used, failed_models, panel_success = _run_panel(panel_cands, pool)
        if panel_success:
            break

    if not panel_success:
        reasons = "; ".join(f"{f['model']} ({f['reason']})" for f in failed_models)
        detail = f" Panel failures: {reasons}." if reasons else ""
        return _error(
            f"All fusion panel models failed for '{model_full}'.{detail}",
            status=503,
        )

    # 3. Judge compares the panel responses and emits structured analysis. Try
    # judge candidates in ranked order, each with per-account failover, so a
    # rate-limited judge rotates to the next model instead of dropping analysis.
    judge_ranked = _rank_aux_models(pool, fcfg.get("judge_model"), config,
                                    prefer_caps=frozenset({"reasoning"}))
    judge_tuple = judge_ranked[0] if judge_ranked else None  # for synth exclusion
    analysis: dict | None = None
    judge_id: str | None = None
    jmsgs = _fusion.build_judge_messages(original_messages, panel_entries)
    for cand in judge_ranked:
        jpn, jpc, juid = cand
        try:
            jresp, jacct = _call_with_account_failover(
                endpoint, jpn, jpc, {"model": juid, "messages": jmsgs}, candidate_timeout,
                forwarded_headers=forwarded_headers,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[server:_proxy_fusion:judge] {jpn}/{juid}: {e}")
            traceback.print_exc()
            continue
        if jresp is not None and jresp.status_code < 400 and not _response_unusable(jresp.get_data()):
            _record_usage(jpn, juid, usage=extract_usage(jresp.get_data()), config=config, account_id=jacct)
            analysis = _fusion.parse_analysis(_fusion.extract_message_text(jresp.get_data()))
            judge_id = f"{jpn}/{juid}"
            judge_tuple = cand
            break
        status = jresp.status_code if jresp is not None else "error"
        logger.warning("  [fusion] judge %s/%s -> %s, trying next", jpn, juid, status)

    # 4. Synthesizer writes the final answer grounded in the analysis. Rank synth
    # candidates (excluding the judge where possible); a panel member is the
    # last-resort synth model so there is always at least one.
    synth_ranked = _rank_aux_models(
        pool, fcfg.get("synthesizer_model"), config,
        prefer_caps=frozenset({"reasoning"}), exclude_first=judge_tuple,
    ) or [panel_success[0][0]]
    smsgs = _fusion.build_synthesizer_messages(original_messages, panel_entries, analysis)

    # Provenance names the synth model actually used; default to the first ranked
    # (updated below when a later candidate wins after rotation).
    spn, spc, suid = synth_ranked[0]
    synth_id = f"{spn}/{suid}"

    def _report(fell_back: bool, with_analysis: bool) -> dict:
        return _fusion.build_report(
            panel_used=panel_used, judge_model=judge_id, synthesizer_model=synth_id,
            failed_models=failed_models, analysis=analysis if with_analysis else None,
            fell_back=fell_back, free=free,
        )

    # Streaming: stream only the synthesis stage; provenance rides the header.
    # Can't rotate mid-stream, so bind the freshest account of the first synth
    # model and degrade to a panel answer if it fails to start.
    if is_streaming:
        stream_cfg, _sacct = _bind_freshest_account(spn, spc, suid)
        header_report = json.dumps(_report(False, with_analysis=False), ensure_ascii=True)
        stream_timeout = server_cfg.get("stream_timeout", 300)
        resp = _proxy_streaming(
            endpoint, spn, stream_cfg,
            {**stripped, "model": suid, "messages": smsgs, "stream": True},
            stream_timeout, config=config, inbound=inbound_adapter,
        )
        if getattr(resp, "status_code", 200) < 400:
            # Fusion has no single "selected" candidate, but the synthesizer is
            # the model whose words reach the client, so that is what the
            # provenance header names. The panel and judge stay in the fusion
            # report, which is where the full picture belongs.
            _note_selected_model(spn, suid)
            with contextlib.suppress(Exception):
                resp.headers["X-LLMProxy-Fusion"] = header_report
            return resp
        # Synth failed to start: degrade to the first panel answer (non-streamed).
        logger.warning("  [fusion] synth %s failed to stream; falling back to panel answer", synth_id)
        body = panel_success[0][1]
        fallback_pn, _fallback_cfg, fallback_um = panel_success[0][0]
        _note_selected_model(fallback_pn, fallback_um)
        out = _fusion.inject_report(body, _report(True, with_analysis=True))
        if not inbound_adapter.is_identity:
            out = inbound_adapter.render_response(out)
        resp = Response(out, status=200, content_type="application/json")
        with contextlib.suppress(Exception):
            resp.headers["X-LLMProxy-Fusion"] = header_report
        return resp

    # Non-streaming synthesis: rotate across synth candidates (each with
    # per-account failover) until one returns a usable answer.
    sresp = None
    sacct = None
    for cand in synth_ranked:
        cpn, cpc, cuid = cand
        try:
            r, a = _call_with_account_failover(
                endpoint, cpn, cpc, {**stripped, "model": cuid, "messages": smsgs}, timeout,
                forwarded_headers=forwarded_headers,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[server:_proxy_fusion:synth] {cpn}/{cuid}: {e}")
            traceback.print_exc()
            continue
        if r is not None and r.status_code < 400 and not _response_unusable(r.get_data()):
            sresp, sacct = r, a
            spn, spc, suid = cand
            synth_id = f"{spn}/{suid}"
            break
        status = r.status_code if r is not None else "error"
        logger.warning("  [fusion] synth %s/%s -> %s, trying next", cpn, cuid, status)

    if sresp is None:
        # Graceful fallback: return the first successful panel response, flagged.
        logger.warning("  [fusion] all synth candidates failed; falling back to panel answer")
        fallback_pn, _fallback_cfg, fallback_um = panel_success[0][0]
        _note_selected_model(fallback_pn, fallback_um)
        out = _fusion.inject_report(panel_success[0][1], _report(True, with_analysis=True))
    else:
        _record_usage(spn, suid, usage=extract_usage(sresp.get_data()), config=config, account_id=sacct)
        _note_selected_model(spn, suid)
        out = _fusion.inject_report(sresp.get_data(), _report(False, with_analysis=True))

    header_report = json.dumps(_report(False, with_analysis=False), ensure_ascii=True)
    if not inbound_adapter.is_identity:
        out = inbound_adapter.render_response(out)
    resp = Response(out, status=200, content_type="application/json")
    with contextlib.suppress(Exception):
        resp.headers["X-LLMProxy-Fusion"] = header_report
    return resp


def _proxy_endpoint(
    endpoint: str,
    inbound: str = "openai",
    *,
    model_override: str | None = None,
    stream_override: bool | None = None,
) -> Response:
    """
    Generic handler that routes a POST request to the correct upstream provider.

    The ``inbound`` dialect (``openai`` for ``/v1/chat/completions``,
    ``anthropic`` for ``/v1/messages``) is normalized to the canonical OpenAI
    schema up front, so all routing, virtual-model, capability, caching, and
    usage logic below operate on one representation. The canonical response (and
    stream) is rendered back into the client's dialect at the boundary.

    Reads the 'model' field from the JSON body, resolves the provider, and
    delegates to the streaming or non-streaming proxy helper.  For non-streaming
    requests, successful responses are stored in a short-lived cache so that
    harnesses which replay the same request in quick succession avoid redundant
    upstream round-trips.

    The special model names "free" and "local" cycle through all matching
    cached models until one returns a successful response.
    """
    raw_body = request.get_json(force=True, silent=True)
    if raw_body is None:
        return _error("Request body must be valid JSON.", status=400)

    inbound_adapter = get_inbound(inbound)
    try:
        payload = inbound_adapter.to_canonical_request(raw_body)
    except UnknownPreviousResponse as e:
        # A Responses client referenced a conversation this process does not
        # hold (different worker, or a restart since). Answering anyway would
        # silently drop the whole prior transcript and return a confident, wrong
        # reply, so say so instead and let the client resend its history.
        return _error(
            f"Unknown previous_response_id '{e}'. llmproxy stores conversation "
            "state in memory only, so it does not survive a restart and is not "
            "shared between workers; resend the conversation in 'input'.",
            status=400, code="invalid_request_error",
        )

    # Dialects that carry the model id / stream flag outside the JSON body
    # (e.g. Gemini puts them in the URL path) override them here.
    if model_override is not None:
        payload["model"] = model_override
    if stream_override is not None:
        payload["stream"] = stream_override

    model_full: str = payload.get("model", "")
    if not model_full:
        return _error("Request body must include a 'model' field.", status=400)

    config = load_config()
    # Accept a slash-form id (provider/model) and normalize it back to the
    # canonical provider__model form before any virtual/route resolution runs.
    model_full = _canonicalize_model_id(model_full, config)
    payload["model"] = model_full
    server_cfg = config.get("server", {})
    is_streaming: bool = payload.get("stream", False)

    # Ask the upstream to emit a final usage chunk so streamed responses can be
    # accounted (token/cost). Standard OpenAI option; opt out per-server via
    # server.stream_include_usage=false if an upstream rejects it.
    if is_streaming and endpoint == "chat/completions" and server_cfg.get("stream_include_usage", True):
        if "stream_options" not in payload:
            payload = {**payload, "stream_options": {"include_usage": True}}

    # Check the short-lived response cache for non-streaming requests.
    # Virtual cycling models bypass the cache so their load-spreading and
    # failover logic runs on every request rather than pinning to one upstream.
    cache_key: str | None = None
    if not is_streaming and not _is_virtual_model(model_full):
        cache_ttl: int = server_cfg.get("response_cache_ttl", _DEFAULT_RESPONSE_CACHE_TTL)
        if cache_ttl > 0:
            cache_key = _response_cache_key(endpoint, payload, request.headers.get("Authorization", ""))
            cached = _response_cache_get(cache_key, cache_ttl)
            if cached:
                content, status, ct, cached_model = cached
                logger.info("  [cache] HIT  key=%s…", cache_key[:12])
                if cached_model:
                    provider_part, _, model_part = cached_model.partition("/")
                    _note_selected_model(provider_part, model_part)
                return Response(content, status=status, content_type=ct)

    if _is_virtual_model(model_full):
        # Fusion is virtual but fans out + judges + synthesizes rather than
        # cycling to one upstream, so it dispatches on its own path first.
        if _is_fusion_model(model_full):
            return _proxy_fusion(
                endpoint, model_full, payload, config, inbound_adapter, is_streaming
            )
        candidates = _get_virtual_candidates(model_full)
        if not candidates:
            return _error(
                f"No '{model_full}' models are currently available. "
                + _virtual_model_hint(model_full),
                status=503,
            )
        # Whether this is a single-tier free/local virtual whose pool we triage
        # by request fit. Distinct from loadbalanced (which crosses tiers); these
        # only ever serve their own tier — the fit pass reorders, never adds or
        # crosses tiers, so a */free virtual stays in the free list and a */local
        # virtual stays in the local list.
        is_free_virtual = _is_free_virtual_model(model_full)
        is_local_virtual = _is_local_virtual_model(model_full)
        is_flagship_virtual = _is_flagship_virtual_model(model_full)
        # True once the pool has actually been ranked by benchmark score, which
        # is what licenses suppressing the soft ordering passes below. It stays
        # False when the membership cache carries no scores yet, so a fresh or
        # pre-upgrade deployment keeps its previous behaviour untouched.
        flagship_ranked = False
        # Every pass that reorders the pool records itself here, so the pick can
        # be explained afterwards instead of reconstructed from log archaeology.
        decisions: list[str] = []
        if _is_loadbalanced_model(model_full):
            # Cost waterfall (free → local → paid), optimized per-prompt within
            # each tier. Owns its full ordering, so the capability/reasoning
            # passes below are stable no-ops over it.
            ordered = _loadbalanced_ordered_candidates(candidates, payload, config)
            decisions.append(ROUTE_SOURCE_LOADBALANCED)
        elif is_flagship_virtual:
            # Tested BEFORE is_free_virtual: llmproxy__flagship/free belongs to
            # both sets, and the benchmark ranking is the stronger signal — it
            # is the only one measured per model rather than inferred from quota
            # or a name. Checked after loadbalanced, which crosses tiers and owns
            # its own waterfall.
            flagship_scores = _get_flagship_scores()
            free_limits = _get_normalized_free_limits(config)
            ranked = sum(
                1 for pn, _pc, um in candidates
                if _flagship_candidate_score(pn, um, flagship_scores) is not None
            )
            flagship_ranked = ranked > 0
            if flagship_ranked:
                ordered = _flagship_ordered_candidates(
                    candidates, flagship_scores, free_limits)
                decisions.append(
                    f"{ROUTE_SOURCE_FLAGSHIP_RANK}={ranked}/{len(candidates)}")
            else:
                # No scores cached — fall back to what this pool did before, and
                # report that honestly rather than claiming a ranking we lack.
                _warn_flagship_unranked(model_full)
                if is_free_virtual:
                    ordered = _capacity_ordered_candidates(candidates, free_limits)
                    decisions.append(ROUTE_SOURCE_CAPACITY)
                else:
                    ordered = _cycling_candidates(candidates)
                    decisions.append(ROUTE_SOURCE_CYCLING)
        elif is_free_virtual:
            free_limits = _get_normalized_free_limits(config)
            ordered = _capacity_ordered_candidates(candidates, free_limits)
            decisions.append(ROUTE_SOURCE_CAPACITY)
        else:
            ordered = _cycling_candidates(candidates)
            decisions.append(ROUTE_SOURCE_CYCLING)

        # Record token/cost (and request count) for every cycled candidate,
        # scoped to the account that actually served it so per-account free-tier
        # quota is metered independently.
        def on_success(pn: str, um: str, body=None, account_id=None) -> None:
            _record_usage(
                pn, um,
                usage=extract_usage(body) if body is not None else None,
                config=config,
                account_id=account_id,
            )
            # Pin this conversation to whatever actually served it, so the next
            # turn keeps the same model (and therefore the same prompt cache and
            # the same tool-calling conventions) until it stops working.
            if is_free_virtual and _free_tier_cache_affinity_enabled(config):
                _record_affinity_success(_affinity_key(payload), pn, um)
        # Proactively prefer candidates that support the capabilities this
        # request needs (tools/vision/reasoning/json).  Stable, never drops
        # candidates, and a no-op when nothing is needed or no metadata exists.
        needed = _needed_capabilities(payload)
        # Request-fit triage for ALL */free and */local virtuals — strictly
        # within the tier. Bias the order by how well each candidate's reasoning
        # tier AND size fit the request (light/regular/deep), layered on top of
        # the capacity/random base order and below the hard capability ordering
        # (which still wins for forced tools/vision/JSON). Within a constrained
        # sub-virtual like deep/free the tier term is constant, so the size term
        # picks the right-sized model from what's available. Never crosses tiers.
        #
        # Skipped on a ranked flagship pool. Inside flagship the tier term is
        # constant too, which leaves _param_count — billions guessed from the
        # model id — as the effective key, and that is precisely the crude proxy
        # the benchmark score replaces. Letting it run would invert the ranking
        # whenever a large model scores below a smaller one.
        if (is_free_virtual or is_local_virtual) and not flagship_ranked:
            ordered = _order_by_request_fit(ordered, payload, _get_model_reasoning(config))
            tier, tier_source = _target_reasoning_tier_explained(payload)
            decisions.append(f"{ROUTE_SOURCE_REQUEST_FIT}={tier}({tier_source})")
            # Log the scorer's inputs, not just its verdict — a tier that looks
            # wrong is only debuggable if the evidence behind it is visible.
            logger.info(
                "  [%s] request-fit first-pick tier=%s via=%s", model_full, tier, tier_source
            )
            if tier_source.startswith(TIER_SOURCE_TOOL_SIGNALS):
                sig = extract_tool_signals(payload)
                logger.info(
                    "  [%s] tool signals: severity=%.2f depth=%d compacted=%s "
                    "recent(edit=%d write=%d read=%d plan=%d) tests_passed=%s score=%.3f",
                    model_full, sig.severity, sig.turn_depth, sig.compacted,
                    sig.recent_edit_count, sig.recent_write_count,
                    sig.recent_read_count, sig.recent_plan_count,
                    sig.tests_passed, score_signals(sig)[0],
                )
        if needed:
            ordered, _cap_dropped = _apply_capability_gate(
                ordered, needed, _model_capabilities(config), model_full,
            )
            decisions.append(f"{ROUTE_SOURCE_CAPABILITY}={'+'.join(sorted(needed))}")
            if _cap_dropped:
                decisions.append(f"{ROUTE_SOURCE_CAPABILITY}_dropped={_cap_dropped}")
        # favorite_free_models is a soft preference over an otherwise unranked
        # free pool. Flagship is ranked on measured capability, which is the
        # stronger claim, so a favorite does not reorder it.
        if is_free_virtual and not flagship_ranked:
            before = ordered[0] if ordered else None
            ordered = _apply_favorite_free_ordering(ordered, config)
            if ordered and ordered[0] is not before:
                decisions.append(ROUTE_SOURCE_FAVORITE)
        # Model-level stickiness for free pools, opt-in. Runs after favorites so
        # an explicitly ranked favorite still wins; among the rest, a
        # conversation keeps landing on the model that last worked for it.
        #
        # This applies to a ranked flagship pool too, which rendezvous hashing
        # could not: the pin is SET by whatever the ordering chose and only
        # moves the pinned target forward, so the first turn of a conversation
        # still gets the best-ranked candidate and best-first is never
        # contradicted. See _order_by_sticky_affinity.
        if is_free_virtual and _free_tier_cache_affinity_enabled(config):
            akey = _affinity_key(payload)
            if akey and len(ordered) > 1:
                pinned = ordered[0] if ordered else None
                ordered = _order_by_sticky_affinity(ordered, akey)
                if ordered and ordered[0] is not pinned:
                    decisions.append(f"{ROUTE_SOURCE_AFFINITY}=free")
        # Context fit runs last of the model-level passes. It is an identity
        # when nothing is known to overflow, so in the common case it costs the
        # earlier passes nothing; in the uncommon case it correctly outranks
        # them, because a pinned favorite or a tools-capable model that cannot
        # hold the conversation is not a usable pick at all. Capability metadata
        # only predicts a soft failure; a known-undersized window predicts a hard
        # one.
        if _config_bool("context_aware_routing", False, config):
            before_ctx = ordered
            ordered = _order_by_context_fit(
                ordered, payload, _get_model_context(config),
                _get_model_context_snapshot(), config,
            )
            if ordered is not before_ctx:
                decisions.append(
                    f"{ROUTE_SOURCE_CONTEXT_FIT}=~{_required_context_tokens(payload, config)}tok"
                )
        # Route around anything that has already refused a request this large.
        # Runs after every other model-level pass and before accounts are
        # expanded: a body limit is a hard fact about the endpoint, so it should
        # outrank a preference, but it says nothing about which credential to
        # use. A no-op until some candidate has actually returned a 413.
        ordered, _demoted = _demote_oversize_candidates(ordered, payload)
        if _demoted:
            decisions.append(f"{ROUTE_SOURCE_OVERSIZE}={_demoted}")
            logger.info(
                "  [%s] %d candidate(s) demoted: they have refused a request this large",
                model_full, _demoted,
            )
        # Expand accounts LAST: each model's credentials become adjacent
        # candidates in its ranked slot, so cycling rotates accounts-first then
        # models. A no-op for single-credential providers.
        affinity_applied = bool(_affinity_key(payload)) and _cache_affinity_applies(
            ordered, config, _is_loadbalanced_model(model_full)
        )
        ordered = _expand_accounts(ordered, payload)
        if affinity_applied:
            decisions.append(ROUTE_SOURCE_AFFINITY)
        route_reason = ",".join(decisions)
        logger.info("  [%s] cycling through %d candidate(s) [%s]", model_full, len(ordered), route_reason)
        if is_streaming:
            timeout = server_cfg.get("stream_timeout", 300)
            return _proxy_cycling_streaming(
                endpoint, model_full, ordered, payload, timeout,
                on_success=on_success, config=config, inbound=inbound_adapter,
                route_reason=route_reason, virtual_model=model_full,
            )
        timeout = server_cfg.get("request_timeout", 120)
        resp = _proxy_cycling_non_streaming(
            endpoint, model_full, ordered, payload, timeout,
            on_success=on_success, route_reason=route_reason,
            virtual_model=model_full, config=config,
        )
    else:
        provider_name, provider_cfg, upstream_model, err = _resolve_provider(model_full)
        if err is not None:
            return err

        logger.info("  provider=%s  model=%s", provider_name, upstream_model)
        # A pinned request names its own model, but reporting it anyway means a
        # client never has to branch on whether it asked for a virtual or a real
        # id to find out what answered.
        _note_selected_model(provider_name, upstream_model)
        upstream_payload = {**payload, "model": upstream_model}

        if is_streaming:
            timeout = server_cfg.get("stream_timeout", 300)
            return _proxy_streaming(endpoint, provider_name, provider_cfg, upstream_payload,
                                    timeout, config=config, inbound=inbound_adapter)
        timeout = server_cfg.get("request_timeout", 120)
        resp = _proxy_request(endpoint, provider_name, provider_cfg, upstream_payload, timeout)
        # Account pinned (non-virtual) non-streaming requests too. Usage is read
        # from the canonical (OpenAI-shaped) body before any inbound rendering.
        if 200 <= resp.status_code < 300:
            _record_usage(provider_name, upstream_model,
                          usage=extract_usage(resp.get_data()), config=config)

    # Render the canonical response into the client's dialect (no-op for openai).
    if not inbound_adapter.is_identity and 200 <= resp.status_code < 300:
        # Rebuilding the Response drops every header stamped upstream of here,
        # which is how /v1/messages and /v1/responses came to lose the route
        # provenance. Carry the X-LLMProxy-* set across explicitly; Content-Length
        # and Content-Type belong to the rendered body, so they are not copied.
        resp = _carry_route_headers(resp, Response(
            inbound_adapter.render_response(resp.get_data()),
            status=resp.status_code,
            content_type="application/json",
        ))

    # Store successful non-streaming responses in the short-lived cache (rendered
    # bytes, so a cache hit returns the correct dialect).
    if cache_key is not None and 200 <= resp.status_code < 300:
        _response_cache_put(
            cache_key, resp.get_data(), resp.status_code, resp.content_type, cache_ttl,
            selected_model=g.get("llmproxy_selected_model") if has_request_context() else None,
        )
    return resp


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions() -> Response:
    """Proxy OpenAI chat completions (supports streaming via SSE)."""
    return _proxy_endpoint("chat/completions")


@app.route("/v1/completions", methods=["POST"])
def completions() -> Response:
    """Proxy legacy text completions, with a chat/completions fallback.

    The request is first forwarded verbatim to the provider's own
    ``/completions`` endpoint. If the upstream doesn't implement the legacy
    endpoint (HTTP 404), the same prompt is transparently retried against
    ``/chat/completions`` — wrapped as a single user message — and the chat
    response is rendered back into the legacy ``text_completion`` shape. Clients
    thus keep the legacy surface even against providers that only speak chat.

    Two cases translate to chat up front rather than probing ``/completions``:

    * **Virtual models** (``llmproxy/free``, ``loadbalanced``, fusion, …) are an
      llmproxy abstraction with no real legacy endpoint to forward to.
    * **Streaming** requests — a streamed passthrough can't surface a pre-stream
      404 without buffering the whole response, so the fallback couldn't be
      applied once bytes are already flowing to the client.
    """
    body = request.get_json(force=True, silent=True)
    if isinstance(body, dict):
        model = _canonicalize_model_id(body.get("model", ""), load_config())
        if body.get("stream") or _is_virtual_model(model):
            return _proxy_endpoint("chat/completions", inbound="openai-completions")

    resp = _proxy_endpoint("completions")
    if resp.status_code == 404:
        logger.info("  [legacy-completions] upstream has no /completions — falling back to chat/completions")
        return _proxy_endpoint("chat/completions", inbound="openai-completions")
    return resp


@app.route("/v1/messages", methods=["POST"])
def anthropic_messages() -> Response:
    """Anthropic Messages API surface (supports streaming via Anthropic SSE).

    The request is translated to canonical OpenAI form, routed exactly like
    /v1/chat/completions (virtual models, capacity routing, native upstreams all
    apply), and the response is rendered back into the Anthropic Messages shape.
    """
    return _proxy_endpoint("chat/completions", inbound="anthropic")


@app.route("/v1/responses", methods=["POST"])
def openai_responses() -> Response:
    """OpenAI Responses API surface (``POST /v1/responses``).

    Without this route a Responses request fell through to the generic
    ``/v1/<subpath>`` passthrough, which resolves a provider directly and has no
    cycling engine behind it — so a virtual model like ``llmproxy/free`` could
    not be used from a Responses-speaking client at all. Here the request is
    translated to canonical OpenAI chat form and routed exactly like
    ``/v1/chat/completions``, so virtual models, capacity and capability
    ordering, context fit and failover all apply, and the result is rendered back
    into the Responses shape.
    """
    return _proxy_endpoint("chat/completions", inbound="responses")


@app.route("/v1/responses/<response_id>", methods=["GET"])
def openai_response_get(response_id: str) -> Response:
    """Fetch a stored Response by id.

    llmproxy keeps conversation state only in the bounded in-process store that
    backs ``previous_response_id`` (see ``dialects/responses._ResponseStore``),
    and that store holds the transcript rather than the rendered Response object.
    So this reports whether the conversation is still known and says plainly that
    the body is not retained, instead of fabricating one.
    """
    from .dialects.responses import STORE
    if STORE.get(response_id) is None:
        return _error(f"No stored response with id '{response_id}'.",
                      status=404, code="not_found")
    return jsonify({
        "id": response_id,
        "object": "response",
        "status": "completed",
        "_note": (
            "llmproxy stores the conversation transcript for previous_response_id "
            "but does not retain rendered response bodies."
        ),
    })


@app.route("/v1/responses/<response_id>", methods=["DELETE"])
def openai_response_delete(response_id: str) -> Response:
    """Forget a stored conversation."""
    from .dialects.responses import STORE
    deleted = STORE.delete(response_id)
    return jsonify({"id": response_id, "object": "response.deleted", "deleted": deleted})


@app.route("/v1/messages/count_tokens", methods=["POST"])
def anthropic_count_tokens() -> Response:
    """Approximate token count for the Anthropic Messages API.

    llmproxy has no model-exact tokenizer, so this returns a heuristic estimate
    (~4 characters/token over the rendered text) — enough for SDK clients that
    call count_tokens before sending a request.
    """
    body = request.get_json(force=True, silent=True)
    if body is None:
        return _error("Request body must be valid JSON.", status=400)
    return jsonify({"input_tokens": _estimate_tokens("anthropic", body)})


def _estimate_tokens(dialect: str, body: dict) -> int:
    """Heuristic token estimate (~4 chars/token) over a request's text."""
    payload = get_inbound(dialect).to_canonical_request(body)
    chars = 0
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            chars += sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
    return max(1, chars // 4)


@app.route("/v1beta/models/<path:model_action>", methods=["POST"])
def gemini_generate(model_action: str) -> Response:
    """Google Gemini generateContent API surface.

    Routes ``/v1beta/models/{model}:generateContent`` (and
    ``:streamGenerateContent`` / ``:countTokens``) so the Google GenAI SDK can
    point at llmproxy. The model id is taken from the URL path and the streaming
    flag from the method verb; both are injected into the canonical request.
    """
    model, _, verb = model_action.rpartition(":")
    if not model or not verb:
        return _error("Expected /v1beta/models/<model>:<method>.", status=404)
    if verb == "countTokens":
        body = request.get_json(force=True, silent=True) or {}
        return jsonify({"totalTokens": _estimate_tokens("gemini", body)})
    return _proxy_endpoint(
        "chat/completions", inbound="gemini",
        model_override=model, stream_override=verb.startswith("stream"),
    )


@app.route("/v1/embeddings", methods=["POST"])
def embeddings() -> Response:
    """Proxy embeddings requests (streaming not applicable)."""
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return _error("Request body must be valid JSON.", status=400)

    model_full: str = payload.get("model", "")
    if not model_full:
        return _error("Request body must include a 'model' field.", status=400)

    config = load_config()
    model_full = _canonicalize_model_id(model_full, config)
    provider_name, provider_cfg, upstream_model, err = _resolve_provider(model_full)
    if err is not None:
        return err

    timeout = config.get("server", {}).get("request_timeout", 120)
    upstream_payload = {**payload, "model": upstream_model}
    resp = _proxy_request("embeddings", provider_name, provider_cfg, upstream_payload, timeout)
    if 200 <= resp.status_code < 300:
        _record_usage(provider_name, upstream_model,
                      usage=extract_usage(resp.get_data()), config=config)
    return resp


# ---------------------------------------------------------------------------
# /v1/usage — token + cost accounting report
# ---------------------------------------------------------------------------

def _split_usage_key(key: str) -> tuple[str, str | None, str]:
    """Parse a usage/saturation key into (provider_name, account_id, model).

    Handles both the anonymous ``provider/model`` form (account_id ``None``) and
    the per-account ``provider#account/model`` form written when a provider has
    multiple credentials.
    """
    provider_part, _, upstream_model = key.partition("/")
    provider_name, sep, account_id = provider_part.partition("#")
    return provider_name, (account_id if sep else None), upstream_model


def _build_usage_report() -> dict:
    """Snapshot the in-memory usage registry into a JSON-serializable report.

    Each metered credential is one entry; a provider with several accounts yields
    one row per account (with an ``account`` field), so per-account free-tier
    consumption is visible. Single-credential providers omit ``account`` and read
    exactly as before. Account labels are surfaced, never the key material.
    """
    config = load_config()
    with _usage_registry_lock:
        items = list(_usage_registry.items())

    models: list[dict] = []
    totals = {
        "requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "cost": 0.0,
    }
    for key, tracker in items:
        snap = tracker.cost_snapshot()
        tok_min, tok_day = tracker.token_snapshot()
        success_rate, avg_latency_ms, health_samples = tracker.health_snapshot()
        provider_name, account_id, upstream_model = _split_usage_key(key)
        believed_free = bool(upstream_model) and _is_model_free(provider_name, upstream_model, config)
        entry = {
            "model": f"{provider_name}/{upstream_model}" if upstream_model else key,
            "requests": snap["requests"],
            "prompt_tokens": snap["prompt_tokens"],
            "completion_tokens": snap["completion_tokens"],
            "total_tokens": snap["total_tokens"],
            "tokens_last_60s": tok_min,
            "tokens_today": tok_day,
            "cost": snap["cost"],
            "cost_currency": "USD",
            "cost_sources": snap["cost_sources"],
            "believed_free": believed_free,
            "unexpected_cost": believed_free and snap["cost"] > 0,
            # Health over the recent attempt window (see usage.HEALTH_WINDOW).
            # "health_samples" is what makes the rate readable: a success_rate of
            # 1.0 over zero samples means untried, not proven good.
            "success_rate": round(success_rate, 4),
            "avg_latency_ms": round(avg_latency_ms, 1),
            "health_samples": health_samples,
            "health_score": round(_health_score(provider_name, upstream_model, account_id), 4),
        }
        if account_id is not None:
            entry["account"] = account_id  # label/id only — never the key
        models.append(entry)
        for field in ("requests", "prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            totals[field] += snap[field]
    totals["cost"] = round(totals["cost"], 8)
    models.sort(key=lambda m: (m["model"], m.get("account") or ""))

    with _paid_free_lock:
        flagged = [
            {"model": k, **v} for k, v in sorted(_paid_free_flags.items())
        ]

    return {
        "object": "usage.report",
        "since": _usage_since,
        "models": models,
        "totals": totals,
        "flagged_paid_free_models": flagged,
    }


@app.route("/v1/usage", methods=["GET"])
@app.route("/usage/stats", methods=["GET"])
def usage_stats() -> Response:
    """Report per-model and aggregate token + cost usage for this worker.

    In-memory and per-process: under a multi-worker WSGI server each worker
    reports only the requests it served. Resets on restart or POST /v1/usage/reset.
    """
    return jsonify(_build_usage_report())


@app.route("/v1/failures", methods=["GET"])
@app.route("/failures", methods=["GET"])
def failure_report() -> Response:
    """Report which routing targets have failed recently, and why.

    Answers the question a 502 from an exhausted pool always raises: which
    models were tried and what did each say. Two views of the same ring buffer —
    ``by_model``, aggregated per ``provider/model`` so a repeatedly failing
    target reads as one row with a count, and ``recent``, the flat newest-first
    list.

    Query parameters: ``limit`` bounds the ``recent`` list (default 50), and
    ``since`` takes a unix timestamp or a relative age like ``15m`` / ``2h``.

    In-memory and per-process, like /v1/usage: under a multi-worker WSGI server
    each worker reports only the requests it served. No secrets are recorded —
    request headers never enter the record at all, and each upstream detail is
    scrubbed of credential-shaped text and truncated.
    """
    since_ts = _parse_since(request.args.get("since"))
    rows = _failure_records(since_ts)
    try:
        limit = max(0, int(request.args.get("limit", 50)))
    except (TypeError, ValueError):
        limit = 50

    by_model: dict[str, dict] = {}
    for row in rows:  # newest first, so the first sighting of a target is its latest
        entry = by_model.get(row["target"])
        if entry is None:
            entry = by_model[row["target"]] = {
                "target": row["target"],
                "provider": row["provider"],
                "model": row["model"],
                "failures": 0,
                "last_status": row.get("status"),
                "last_seen": row.get("at"),
                "last_detail": row.get("detail") or "",
                "slowest_ms": None,
                "kinds": {},
                "statuses": {},
            }
        entry["failures"] += 1
        # A pool that burns a full candidate timeout every attempt and one that
        # returns fast 404s are different problems wearing the same status code.
        took = row.get("duration_ms")
        if took is not None and (entry["slowest_ms"] is None or took > entry["slowest_ms"]):
            entry["slowest_ms"] = took
        kind = row.get("kind") or "upstream"
        entry["kinds"][kind] = entry["kinds"].get(kind, 0) + 1
        status = row.get("status")
        if status is not None:
            key = str(status)
            entry["statuses"][key] = entry["statuses"].get(key, 0) + 1

    ranked = sorted(by_model.values(), key=lambda e: (-e["failures"], e["target"]))
    return jsonify({
        "object": "llmproxy.failures",
        "window_seconds": _FAILURE_LOG_TTL_S,
        "capacity": _FAILURE_LOG_MAX,
        "total": len(rows),
        "distinct_targets": len(ranked),
        "by_model": ranked,
        "recent": rows[:limit],
    })


# ---------------------------------------------------------------------------
# Read-only introspection: /v1/providers and /v1/config
# ---------------------------------------------------------------------------
#
# Both existed only as 404-shaped 400s before: no route matched, so they fell
# into the OpenAI passthrough below, which demands "?provider=<name>" to know
# which upstream to forward to. A client asking llmproxy about ITSELF got
# "Supply '?provider=<name>'", which reads like a malformed request rather than
# like a missing endpoint.
#
# They are deliberately in the same family as /v1/usage and /v1/failures:
# unauthenticated, read-only, and carrying NO secrets. That last one is a hard
# constraint rather than an aspiration, which is why neither endpoint emits a
# key even in masked form. A mask still leaks its last characters, and these
# endpoints answer to anyone who can reach the port. Whether a credential is
# configured is the only fact about it worth publishing, and it is a bool.
#
# The token-gated /admin/api/config remains the place to read or edit the
# actual configuration, masks included.

# Key names whose VALUES never appear in these responses, matched case-
# insensitively as substrings. Belt-and-braces: everything below is assembled
# field by field rather than copied wholesale, so nothing secret should reach
# the filter in the first place. It exists because "should" is doing load-
# bearing work in that sentence, and a future key added to the server block is
# exactly how that assumption breaks.
_SECRET_KEY_HINTS = ("key", "token", "secret", "password", "passwd", "credential")


def _looks_secret(name: str) -> bool:
    """True when a config field name suggests it holds a credential.

    ``api_key_set`` and friends are deliberately NOT caught: the suffix marks a
    boolean derived from a secret, which is the safe form and the whole point of
    publishing it.
    """
    lowered = name.lower()
    if lowered.endswith(("_set", "_is_env", "_count")):
        return False
    return any(hint in lowered for hint in _SECRET_KEY_HINTS)


def _public_config_block(block: dict) -> dict:
    """A copy of *block* with credential-shaped fields dropped entirely.

    Dropped rather than masked, for the reason in the note above. Named apart
    from ``_scrub_secrets``, which redacts secrets out of free TEXT for the
    failure log: same intent, different input, and one shadowing the other is a
    bug that only shows up when a request actually fails.
    """
    return {k: v for k, v in block.items()
            if isinstance(k, str) and not _looks_secret(k)}


def _provider_summary(name: str, cfg: dict, route_counts: dict[str, int]) -> dict:
    """One provider's public shape: what it is, not how to authenticate to it."""
    base_url = provider_base_url(cfg)
    try:
        accounts = provider_accounts(cfg)
    except Exception:  # noqa: BLE001 — introspection must not fail on bad config
        accounts = []
    return {
        "name": name,
        "base_url": base_url,
        "api_key_set": bool(provider_api_key(cfg)),
        "accounts": len(accounts),
        "account_strategy": cfg.get("account_strategy"),
        "models": route_counts.get(name, 0),
        # Both are per-provider switches an operator is likely to be checking
        # when they call this at all: "why is this provider not in my free pool"
        # is usually one of the two.
        "expose_to_virtual_models": _provider_exposes_to_virtual_models(cfg),
        "local": _is_local_url(base_url),
        "model_filter": cfg.get("model_filter"),
    }


@app.route("/v1/providers", methods=["GET"])
@app.route("/providers", methods=["GET"])
def list_providers() -> Response:
    """The configured providers, with no credentials.

    ``?provider=<name>`` is still honoured as a passthrough to that upstream's
    own ``/v1/providers``, so adding this route cannot break anyone who was
    relying on the previous behaviour. Without it, the question is about
    llmproxy and is answered here.
    """
    if request.args.get("provider"):
        return passthrough("providers")

    config = load_config()
    route_counts: dict[str, int] = {}
    for provider_name, _upstream in _get_distinct_routes():
        route_counts[provider_name] = route_counts.get(provider_name, 0) + 1

    providers = [
        _provider_summary(name, cfg, route_counts)
        for name, cfg in sorted((config.get("providers") or {}).items())
        if isinstance(cfg, dict)
    ]
    return jsonify({
        "object": "llmproxy.providers",
        "total": len(providers),
        # A provider configured but serving nothing is the single most common
        # "why is my model missing" cause, and counting it here saves diffing
        # this response against /v1/models by hand.
        "serving_models": sum(1 for p in providers if p["models"]),
        "providers": providers,
    })


@app.route("/v1/config", methods=["GET"])
@app.route("/config", methods=["GET"])
def effective_config() -> Response:
    """The EFFECTIVE configuration llmproxy is running on, with no credentials.

    Effective, not config.json's contents: the routing keys are merged from four
    layers (provider defaults, learned, listing, curated) before the router sees
    them, so reading config.json alone shows a deployment as having no free
    models and no capability data at all. What is reported here is what actually
    decides routing.

    The routing keys are summarised by SIZE rather than listed. They run to
    thousands of entries on an ordinary deployment, which is a different request
    from "show me my settings" — ``/admin/api/config`` and
    ``/admin/api/routing-metadata`` serve that one, with editing.
    """
    if request.args.get("provider"):
        return passthrough("config")

    config = load_config()
    merged = _merged_routing_config(config)

    def _size(key: str) -> int:
        value = merged.get(key)
        return len(value) if isinstance(value, (list, dict)) else 0

    admin_cfg = config.get("admin") or {}
    return jsonify({
        "object": "llmproxy.config",
        "version": __version__,
        "config_path": str(get_config_path()),
        "server": _public_config_block(config.get("server") or {}),
        "flagship_tier": flagship_tier_cfg(config),
        "providers": sorted((config.get("providers") or {}).keys()),
        # Sizes, not contents. See the docstring.
        "routing_metadata": {key: _size(key) for key in _ROUTING_CONFIG_KEYS},
        "request_log": _request_log_mode(config),
        "free_tier_cache_affinity": _free_tier_cache_affinity_enabled(config),
        "allow_implicit_paid": _allow_implicit_paid(config),
        "admin": {
            "enabled": bool(admin_cfg.get("enabled", True) is not False),
            # Whether a token is configured, never the token.
            "token_set": bool(admin_cfg.get("token")
                              or os.environ.get("LLMPROXY_ADMIN_TOKEN")),
        },
    })


@app.route("/v1/failures/reset", methods=["POST"])
def failure_reset() -> Response:
    """Clear this worker's failure ring. Gated by the admin auth guard."""
    from .admin import enforce_admin_auth  # local import: admin is wired after routes
    auth_err = enforce_admin_auth()
    if auth_err is not None:
        body, status = auth_err
        return make_response(body, status)
    _reset_failures()
    return jsonify({"object": "llmproxy.failures.reset", "ok": True})


@app.route("/v1/usage/reset", methods=["POST"])
def usage_reset() -> Response:
    """Clear this worker's usage counters. Gated by the admin auth guard."""
    from .admin import enforce_admin_auth  # local import: admin is wired after routes
    auth_err = enforce_admin_auth()
    if auth_err is not None:
        body, status = auth_err
        return make_response(body, status)
    _reset_usage()
    return jsonify({"object": "usage.reset", "ok": True, "since": _usage_since})


# ---------------------------------------------------------------------------
# Catch-all pass-through for other /v1/* endpoints
# ---------------------------------------------------------------------------

@app.route("/v1/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def passthrough(subpath: str) -> Response:
    """
    Best-effort pass-through for any /v1/* endpoint not explicitly handled
    above (e.g., /v1/audio/transcriptions, /v1/images/generations).

    For POST/PUT/PATCH, the 'model' field is used to determine the provider.
    For GET/DELETE, a query parameter 'provider=<name>' must be supplied.
    """
    config = load_config()
    server_cfg = config.get("server", {})
    timeout = server_cfg.get("request_timeout", 120)

    if request.method in ("POST", "PUT", "PATCH"):
        payload = request.get_json(force=True, silent=True) or {}
        model_full = payload.get("model", "")
        provider_name_hint = request.args.get("provider", "")

        if model_full:
            model_full = _canonicalize_model_id(model_full, config)
            provider_name, provider_cfg, upstream_model, err = _resolve_provider(model_full)
            if err:
                return err
            upstream_payload = {**payload, "model": upstream_model}
        elif provider_name_hint:
            provider_name = provider_name_hint
            provider_cfg = get_provider(config, provider_name)
            if not provider_cfg:
                return _error(f"Unknown provider '{provider_name}'.", status=404)
            upstream_payload = payload
        else:
            return _error(
                "Cannot determine upstream provider: supply 'model' in the request body "
                "or '?provider=<name>' as a query parameter.",
                status=400,
            )

        is_streaming = payload.get("stream", False)
        if is_streaming:
            return _proxy_streaming(subpath, provider_name, provider_cfg, upstream_payload,
                                    server_cfg.get("stream_timeout", 300))
        else:
            return _proxy_request(subpath, provider_name, provider_cfg, upstream_payload, timeout)

    else:  # GET / DELETE
        provider_name = request.args.get("provider", "")
        if not provider_name:
            return _error(
                f"Supply '?provider=<name>' to route GET /v1/{subpath} to the correct upstream.",
                status=400,
            )
        provider_cfg = get_provider(config, provider_name)
        if not provider_cfg:
            return _error(f"Unknown provider '{provider_name}'.", status=404)

        base_url = provider_base_url(provider_cfg)
        url = f"{base_url}/{subpath}"
        api_key = provider_api_key(provider_cfg)
        # Passthrough, so the client's own forwarded headers apply here too --
        # including the resolved User-Agent. This site built its dict by hand
        # and sent none at all.
        headers = {**_forwarded_client_headers()}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        params = {k: v for k, v in request.args.items() if k != "provider"}
        try:
            resp = requests.request(
                request.method, url, headers=headers, params=params, timeout=timeout
            )
            return Response(
                resp.content,
                status=resp.status_code,
                content_type=resp.headers.get("Content-Type", "application/json"),
            )
        except Exception as e:
            return _upstream_error(provider_name, e)


# ---------------------------------------------------------------------------
# Server launcher
# ---------------------------------------------------------------------------

def run_server(config_path: str | None = None) -> None:
    """
    Start the Flask development server using settings from the config file.

    In production, prefer running with a WSGI server (gunicorn) by calling
    the Flask app object directly.  The Dockerfile uses gunicorn for this
    reason.
    """
    config = load_config(config_path)
    server_cfg = config.get("server", {})

    host: str = server_cfg.get("host", "0.0.0.0")
    port: int = int(server_cfg.get("port", 8080))
    log_level: str = server_cfg.get("log_level", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    providers_cfg: dict = config.get("providers", {})
    logger.info("llmproxy starting — providers: %s", list(providers_cfg) or ["(none — run --setup)"])
    logger.info("Listening on %s:%d", host, port)

    # Warm the virtual-model route cache (so routing works before the first
    # /v1/models call) and, if enabled, run the free-models updater. Runs in a
    # background daemon thread, guarded to fire once.
    _run_startup_tasks_once(config_path)

    app.run(host=host, port=port, threaded=True, debug=False)


# ---------------------------------------------------------------------------
# Web admin UI / config API (/admin, /admin/api/*)
# ---------------------------------------------------------------------------
# Registered at import time so the blueprint is present whether the app is run
# via gunicorn (which imports `app` directly) or the Flask dev server. The
# blueprint's before_request guard enforces the localhost-only / token policy.
from .admin import register_admin  # noqa: E402  (deferred to avoid import cycle)

register_admin(app)
