"""
server.py — OpenAI-compatible proxy server for llmproxy.

Implements the following OpenAI API endpoints:
  GET  /v1/models                  Aggregate models from all providers
  GET  /v1/models/<model_id>       Single model metadata lookup
  POST /v1/chat/completions        Proxy chat completions (streaming + non-streaming)
  POST /v1/completions             Proxy legacy completions
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
import random
import re
import threading
import time
import traceback
import urllib.parse
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

from . import __version__
from . import fusion as _fusion
from .config import (
    RESERVED_PROVIDER_NAMES,
    account_bound_cfg,
    get_config_path,
    get_provider,
    load_config,
    model_is_allowed,
    parse_model_string,
    provider_account_id,
    provider_account_strategy,
    provider_accounts,
    provider_api_key,
    provider_base_url,
    resolve_env_refs,
    save_config,
)
from .dialects import get_inbound, get_outbound
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

_REASONING_LEVELS: tuple[str, ...] = ("exploratory", "standard", "deep")
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

# HTTP statuses that mean "out of quota / rate limited": 402 Payment Required
# (out of credits) and 429 Too Many Requests. Both mark the model unavailable
# until its reset; a plain 5xx is transient and retried without a cooldown.
_QUOTA_STATUSES = frozenset({402, 429})

# Machine-readable quota codes (specific enough to trust on their own) and
# generic phrases (only trusted inside an error-shaped body).
_QUOTA_CODES = ("resource_exhausted", "insufficient_quota", "rate_limit_exceeded")
_QUOTA_PHRASES = ("quota", "rate limit", "too many requests")


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
    """Append *qualified_id* (``provider/model``) to config['cost_observed_free_tier'].

    Best-effort and idempotent: reads the on-disk config.json directly (so the
    hand-edited file is preserved rather than overwritten with merged defaults),
    adds the id if absent (case-insensitive), and writes it back atomically. Any
    failure is logged and swallowed — a usage-accounting side effect must never
    break request handling.
    """
    try:
        with _cost_observed_persist_lock:
            path = get_config_path()
            raw: dict = {}
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raw = {}
            if not isinstance(raw, dict):
                return
            existing = raw.get(COST_OBSERVED_KEY)
            ids = [x for x in existing if isinstance(x, str)] if isinstance(existing, list) else []
            if qualified_id.lower() in {x.lower() for x in ids}:
                return  # already recorded
            ids.append(qualified_id)
            raw[COST_OBSERVED_KEY] = sorted(set(ids), key=str.lower)
            # Also drop it from the live believed_free right now, so the config on
            # disk is self-consistent even before the updater/PR runs. Routing
            # already avoids it via _is_cost_observed, but this keeps the file clean.
            bf = raw.get("believed_free")
            if isinstance(bf, list):
                raw["believed_free"] = [
                    m for m in bf
                    if not (isinstance(m, str) and m.lower() == qualified_id.lower())
                ]
            save_config(raw)
            logger.info("[usage] recorded %s in %s and dropped it from live believed_free",
                        qualified_id, COST_OBSERVED_KEY)
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
# Periodic interval probe checks (endpoint probe, cost probe, PR creation)
# ---------------------------------------------------------------------------
# State files are re-read at most once per _PROBE_INTERVAL_GATE_SEC so
# concurrent requests don't all hit disk simultaneously. The actual probe
# frequency is controlled by the per-probe frequency_minutes / frequency_days
# settings in config.json.
_PROBE_INTERVAL_GATE_SEC = 60   # check state files at most once per minute
_last_probe_interval_check: float = 0.0
_probe_interval_check_lock = threading.Lock()

_endpoint_probe_inflight: bool = False
_endpoint_probe_lock = threading.Lock()
_cost_probe_inflight: bool = False
_cost_probe_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Short-lived response cache (non-streaming only)
# ---------------------------------------------------------------------------
# Keyed on SHA-256(endpoint + sorted JSON payload).  Only 2xx responses are
# stored.  Entries expire after server.response_cache_ttl seconds (default 120).

_response_cache: dict[str, tuple[bytes, int, str, float]] = {}
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
    expired = [k for k, (_, _, _, ts) in _response_cache.items() if now - ts > ttl]
    for k in expired:
        del _response_cache[k]


def _response_cache_get(key: str, ttl: int) -> tuple[bytes, int, str] | None:
    with _response_cache_lock:
        _response_cache_prune(ttl)
        entry = _response_cache.get(key)
    if entry is None:
        return None
    content, status, content_type, _ = entry
    return content, status, content_type


def _response_cache_put(key: str, content: bytes, status: int, content_type: str, ttl: int) -> None:
    with _response_cache_lock:
        _response_cache_prune(ttl)
        _response_cache[key] = (content, status, content_type, time.monotonic())


@app.before_request
def _log_request() -> None:
    g._start_time = time.monotonic()
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
    return response


# ---------------------------------------------------------------------------
# Utility: build upstream headers
# ---------------------------------------------------------------------------

_FORWARDED_REQUEST_HEADERS = {
    "Content-Type",
    "HTTP-Referer",
    "X-Title",
    "User-Agent",
    "X-Request-ID",
}


def _upstream_headers(provider_cfg: dict) -> dict:
    """
    Build the header dict to send to an upstream provider.

    Always injects the provider's API key as the Bearer token.  Selected
    client-supplied headers are forwarded where the upstream is likely to
    consume them (e.g., HTTP-Referer for OpenRouter rate-limit attribution).
    """
    headers = {"Content-Type": "application/json"}
    api_key = provider_api_key(provider_cfg)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(_forwarded_client_headers())
    return headers


def _forwarded_client_headers() -> dict:
    """Selected client headers we relay upstream (OpenRouter attribution etc.).

    Returns only the present subset of _FORWARDED_REQUEST_HEADERS. Outbound
    dialect adapters decide whether to merge these (the OpenAI adapter does;
    native Anthropic/Gemini ignore them).

    Returns ``{}`` when there is no active request context (e.g. a background
    worker thread), so callers off the request thread degrade gracefully rather
    than raising. Such callers should instead capture these on the request thread
    and pass them down (see _proxy_fusion's panel fan-out, which forwards them via
    ``_proxy_request(..., forwarded_headers=...)``).
    """
    if not has_request_context():
        return {}
    out: dict = {}
    for header in _FORWARDED_REQUEST_HEADERS - {"Content-Type"}:
        value = request.headers.get(header)
        if value:
            out[header] = value
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
    if reasoning.get(upstream_id.lower()) or reasoning.get(f"{provider_name}/{upstream_id}".lower()):
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
    # at a different path entirely (e.g. GitHub Models serves chat at
    # /inference/chat/completions but the catalog at /catalog/models; Cloudflare
    # Workers AI has no GET /v1/models and lists at /ai/models/search). Allow a
    # per-provider override so those upstreams can still be discovered.
    url = resolve_env_refs(provider_cfg.get("models_url")) or f"{base_url}/models"
    # Field on each model object that carries the upstream model id. Defaults to
    # the OpenAI "id"; Cloudflare's /ai/models/search puts the usable id (the
    # "@cf/..." name) in "name" and reserves "id" for an internal UUID.
    id_field = provider_cfg.get("models_id_field") or "id"
    # Optional task filter: when set, keep only models whose task.name matches
    # (case-insensitive). Cloudflare's catalog mixes Text Generation, embeddings,
    # image, etc. into one list; this restricts it to chat-capable models.
    keep_task = provider_cfg.get("models_keep_task")
    headers = {"Content-Type": "application/json"}
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
    for m in all_models:
        route = m.pop("_route", None)
        if route:
            # Dual-key on both the canonical "provider__model" id and the advertised
            # "provider/model" form so an inbound id in either form resolves to the
            # exact upstream losslessly (no string-level reverse needed on the hot path).
            new_cache[m["id"]] = route
            new_cache[_display_id(m["id"])] = route

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

    logger.info("[server:_rebuild_route_cache] %d entries", len(new_cache))
    return all_models


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

        existing_kf: list = config.setdefault("believed_free", [])
        existing_mr: dict = config.setdefault("model_reasoning", {})
        existing_fl: dict = config.setdefault("free_limits", {})
        modified = False

        for provider_key, provider_cfg in local_providers.items():
            base_url = provider_base_url(provider_cfg)
            api_key = provider_api_key(provider_cfg)
            local_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
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
                existing_kf.remove(e)
                modified = True
                logger.info(
                    "[local-sync] Removed %s from believed_free "
                    "(local provider — routed via llmproxy__local instead).", e,
                )

            # Same for free_limits — local models don't use the capacity-aware
            # free-tier scheduler.
            stale_fl = [k for k in existing_fl if isinstance(k, str) and k.startswith(prefix)]
            for k in stale_fl:
                del existing_fl[k]
                modified = True
                logger.info("[local-sync] Removed %s from free_limits.", k)

            # Prune stale model_reasoning entries that this provider previously
            # contributed but no longer serves.
            stale_mr = [k for k in existing_mr if k.startswith(prefix) and k not in expected]
            for k in stale_mr:
                del existing_mr[k]
                modified = True
                logger.info("[local-sync] Pruned stale model_reasoning: %s", k)

            # Add new model_reasoning entries so /local/<level> routing works.
            for qualified in expected:
                if qualified not in existing_mr:
                    model_id = qualified[len(prefix):]
                    existing_mr[qualified] = _infer_local_reasoning_level(model_id)
                    modified = True
                    logger.info("[local-sync] Added model_reasoning: %s -> %s", qualified, existing_mr[qualified])

        if modified:
            save_config(config)

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
    """Check frequency intervals for endpoint probe, cost probe, and PR creation.

    Fires each as a background daemon thread if its interval has elapsed.
    Gated by _PROBE_INTERVAL_GATE_SEC so state files are not read on every
    single request — the actual probe frequency is set in config.json.

    Endpoint probe: gated by sync_on_startup OR update_on_startup.
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

    # Endpoint probe — gated by sync_on_startup OR update_on_startup.
    if free_tier.get("sync_on_startup") or free_tier.get("update_on_startup"):
        _maybe_fire_endpoint_probe(config, free_tier, config_path)

    # Cost probe — gated by update_on_startup + cost_probe.enabled.
    if free_tier.get("update_on_startup") and free_tier.get("cost_probe", {}).get("enabled"):
        _maybe_fire_cost_probe(config, free_tier, config_path)

    # PR creation interval — independent of startup flags.
    _maybe_fire_pr_if_due(config, config_path)


def _maybe_fire_endpoint_probe(
    config: dict, free_tier: dict, config_path: str | None
) -> None:
    ep_cfg = free_tier.get("endpoint_probe", {})
    freq_min = ep_cfg.get("frequency_minutes", 30)
    freq_days = freq_min / 1440.0
    try:
        import os as _os
        import sys
        repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from llmproxy.config import load_endpoint_probe_state
        from scripts.update_free_models import _probe_due
    except Exception:  # noqa: BLE001 — scripts/ may not be available
        return
    state = load_endpoint_probe_state(config_path)
    due, _ = _probe_due(state.get("last_probe_at"), freq_days)
    if not due:
        return

    global _endpoint_probe_inflight
    with _endpoint_probe_lock:
        if _endpoint_probe_inflight:
            return
        _endpoint_probe_inflight = True

    def _run() -> None:
        global _endpoint_probe_inflight
        try:
            logger.info("[endpoint-probe] interval due — running endpoint probe")
            _run_free_models_update(load_config(), config_path)
            with _models_list_cache_lock:
                global _models_list_cache
                _models_list_cache = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("[endpoint-probe] failed: %s", exc)
        finally:
            with _endpoint_probe_lock:
                _endpoint_probe_inflight = False

    threading.Thread(target=_run, daemon=True, name="endpoint-probe-interval").start()


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
         the full free-models updater (streaming its progress to the log).
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
        ran = False
        if config.get("free_tier", {}).get("update_on_startup") is True:
            ran = _run_free_models_update(config, config_path)

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

        # 6. Check frequency intervals for endpoint probe, cost probe, and PR
        #    creation. Fires background threads for any that are due.
        _maybe_fire_interval_probes(config_path)

    threading.Thread(target=_run, daemon=True, name="startup-tasks").start()


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
                "deployment (`providers_pr.enabled`). Free-tier status is best-effort — "
                "review the diff before merging."
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
def list_models() -> Response:
    """
    Aggregate model listings from all configured providers.

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
def get_model(model_id: str) -> Response:
    """
    Return metadata for a single proxy model ID.

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
        return _error(
            f"Upstream provider '{provider_name}' timed out after {timeout}s.",
            status=504,
            code="timeout",
        )
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
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            logger.error("[server:_translated_stream] provider=%s timed out", provider_name)
            yield b'data: {"error":{"message":"Upstream stream timed out."}}\n\n'
        except Exception as e:  # noqa: BLE001
            logger.error("[server:_translated_stream] provider=%s: %s", provider_name, e)
            traceback.print_exc()
            msg = str(e).replace('"', "'")
            yield f'data: {{"error":{{"message":"Upstream error: {msg}"}}}}\n\n'.encode()
        finally:
            if config is not None and upstream_model and captured:
                _record_usage(provider_name, upstream_model, usage=captured,
                              config=config, account_id=account_id)

    return Response(generate(), content_type="text/event-stream")


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

    # Identity fast path: raw passthrough, unchanged.
    @stream_with_context
    def generate():
        tail = bytearray()
        try:
            with requests.post(
                url,
                headers=headers,
                json=body,
                stream=True,
                timeout=timeout,
            ) as upstream_resp:
                first = True
                for chunk in upstream_resp.iter_content(chunk_size=None):
                    if chunk:
                        if first:
                            logger.info(
                                "  upstream %d  first chunk: %s",
                                upstream_resp.status_code,
                                chunk[:200],
                            )
                            first = False
                        yield chunk
                        tail += chunk
                        if len(tail) > _STREAM_TAIL_BYTES:
                            del tail[:-_STREAM_TAIL_BYTES]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            logger.error(
                "[server:_proxy_streaming] provider=%s timed out", provider_name
            )
            yield (
                b'data: {"error":{"message":"Upstream stream timed out."}}\n\n'
            )
        except Exception as e:
            logger.error(
                "[server:_proxy_streaming] provider=%s: %s", provider_name, e
            )
            traceback.print_exc()
            msg = str(e).replace('"', "'")
            yield (
                f'data: {{"error":{{"message":"Upstream error: {msg}"}}}}\n\n'.encode()
            )
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
    raw = config.get("model_capabilities")
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


def _model_has_capability(provider_name: str, upstream_id: str, cap: str, cap_map: dict[str, set[str]]) -> bool:
    """Two-form lookup (bare id or provider/id) for a single capability."""
    caps = cap_map.get(upstream_id.lower()) or cap_map.get(f"{provider_name}/{upstream_id}".lower())
    return bool(caps and cap in caps)


def _needed_capabilities(payload: dict) -> set[str]:
    """The set of capabilities this request needs, per the request-detectors."""
    return {cap for cap, (detect, _s, _v) in _CAPABILITIES.items() if detect(payload)}


def _order_by_capability(
    candidates: list[tuple[str, dict, str]],
    needed: set[str],
    cap_map: dict[str, set[str]],
) -> list[tuple[str, dict, str]]:
    """Stable-sort candidates so those satisfying the most needed caps come first.

    Never drops candidates — incomplete capability metadata must not turn a
    request into a hard 503.  A no-op when *needed* is empty.
    """
    if not needed:
        return candidates

    def satisfied(c: tuple[str, dict, str]) -> int:
        pn, _cfg, uid = c
        return sum(1 for cap in needed if _model_has_capability(pn, uid, cap, cap_map))

    return sorted(candidates, key=satisfied, reverse=True)


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


def _target_reasoning_tier(payload: dict) -> str:
    """Choose the reasoning tier that best fits a request's size and type.

    Explicit thinking requests go to ``deep``; otherwise the estimated input
    size buckets the request into ``exploratory`` (small/fast), ``standard``, or
    ``deep`` (large). This drives the first-pick order for the general virtuals;
    failover still walks the rest of the candidates, so the choice is only a
    preference, never a restriction.
    """
    if _wants_thinking(payload):
        return "deep"
    tokens = _estimate_payload_tokens(payload)
    if tokens <= _TIER_SMALL_MAX_TOKENS:
        return "exploratory"
    if tokens <= _TIER_MEDIUM_MAX_TOKENS:
        return "standard"
    return "deep"


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


def _response_unusable(body_bytes: bytes) -> bool:
    """True when a non-streaming HTTP 200 isn't actually a usable completion.

    Some upstreams answer ``200 OK`` while the body carries an error object or
    an empty result (no ``choices``).  Treating these as failures lets the
    cycling loop fail over instead of handing the client a dead response.

    Deliberately conservative to avoid false failover: a body with at least one
    ``choices`` entry is accepted even when its ``content`` is empty (a model
    may legitimately answer with tool calls or an empty string).  A body that
    isn't JSON at all is treated as unusable, since every cycled endpoint speaks
    JSON chat/completions.
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


def _proxy_cycling_non_streaming(
    endpoint: str,
    label: str,
    candidates: list[tuple[str, dict, str]],
    payload: dict,
    timeout: int,
    on_success: Callable[..., None] | None = None,
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

    ``on_success`` is invoked as ``on_success(provider, model, body)`` with the
    successful response bytes so the caller can record token + cost usage.
    """
    candidate_timeout = min(timeout, _VIRTUAL_CANDIDATE_TIMEOUT)
    total = len(candidates)
    last: Response | None = None
    for idx, (provider_name, provider_cfg, upstream_model) in enumerate(candidates):
        account_id = provider_account_id(provider_cfg)
        upstream_payload = {**payload, "model": upstream_model}
        max_attempts = _candidate_max_attempts(idx, total)
        for attempt in range(max_attempts):
            logger.info("  [%s] trying %s/%s", label, provider_name, upstream_model)
            resp = _proxy_request(endpoint, provider_name, provider_cfg, upstream_payload, candidate_timeout)
            if resp.status_code < 400 or not _is_transient_status(resp.status_code):
                break
            if attempt < max_attempts - 1:
                logger.warning(
                    "  [%s] %s/%s returned %d, retrying (%d/%d)",
                    label, provider_name, upstream_model, resp.status_code,
                    attempt + 1, max_attempts - 1,
                )
                time.sleep(_VIRTUAL_RETRY_BACKOFF)
        if resp.status_code < 400:
            body = resp.get_data()
            if _capability_failed(payload, body):
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
                logger.warning(
                    "  [%s] %s/%s returned 200 with an unusable body (error/empty), trying next",
                    label, provider_name, upstream_model,
                )
                last = resp
                continue
            if on_success is not None:
                on_success(provider_name, upstream_model, body, account_id)
            return resp
        if _is_quota_error(resp.status_code, resp.get_data()):
            _record_quota_saturation(
                provider_name, provider_cfg, upstream_model, resp.headers.get("Retry-After")
            )
        logger.warning(
            "  [%s] %s/%s returned %d, trying next", label, provider_name, upstream_model, resp.status_code
        )
        last = resp
    return last or _error(f"No '{label}' models available.", status=503)


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
    candidate_timeout = min(timeout, _VIRTUAL_CANDIDATE_TIMEOUT)
    inbound = inbound or get_inbound("openai")
    total = len(candidates)
    last_error: tuple[bytes, int, str] | None = None

    for idx, (provider_name, provider_cfg, upstream_model) in enumerate(candidates):
        account_id = provider_account_id(provider_cfg)
        upstream_payload = {**payload, "model": upstream_model}
        max_attempts = _candidate_max_attempts(idx, total)
        base_url = provider_base_url(provider_cfg)
        outbound = get_outbound(provider_cfg.get("protocol"))
        url, headers, body = outbound.build_request(
            endpoint, base_url, provider_cfg, upstream_payload,
            stream=True, forwarded_headers=_forwarded_client_headers(),
        )

        # Open the upstream. Transient failures fail over to the next candidate
        # immediately unless this is the last one (then same-candidate retries).
        resp = None
        for attempt in range(max_attempts):
            logger.info("  [%s] trying %s/%s  [streaming]", label, provider_name, upstream_model)
            try:
                resp = requests.post(url, headers=headers, json=body, stream=True,
                                     timeout=(candidate_timeout, timeout))
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_attempts - 1:
                    logger.warning("  [%s] %s/%s connect error: %s, retrying (%d/%d)",
                                   label, provider_name, upstream_model, e, attempt + 1, max_attempts - 1)
                    time.sleep(_VIRTUAL_RETRY_BACKOFF)
                    continue
                logger.warning("  [%s] %s/%s error: %s, trying next", label, provider_name, upstream_model, e)
                resp = None
                break
            except Exception as e:
                logger.warning("  [%s] %s/%s error: %s, trying next", label, provider_name, upstream_model, e)
                resp = None
                break
            if resp.status_code < 400 or not _is_transient_status(resp.status_code):
                break
            if attempt < max_attempts - 1:
                logger.warning("  [%s] %s/%s -> %d, retrying (%d/%d)",
                               label, provider_name, upstream_model, resp.status_code, attempt + 1, max_attempts - 1)
                resp.close()
                time.sleep(_VIRTUAL_RETRY_BACKOFF)

        if resp is None:
            continue
        if resp.status_code >= 400:
            if _is_quota_error(resp.status_code, resp.content):
                _record_quota_saturation(
                    provider_name, provider_cfg, upstream_model, resp.headers.get("Retry-After")
                )
            last_error = (
                resp.content,
                resp.status_code,
                resp.headers.get("Content-Type", "application/json"),
            )
            resp.close()
            logger.warning(
                "  [%s] %s/%s -> %d, trying next", label, provider_name, upstream_model, resp.status_code
            )
            continue

        try:
            # Peek the opening of the stream so a 200 that immediately emits an
            # SSE error event fails over like an HTTP error instead of being
            # handed to the client.  The peeked prefix is replayed verbatim, so
            # the first token is never dropped.
            error_body, prefix, rest = _peek_stream(resp)
            if error_body is not None:
                # A stream that opens with a quota error cools the candidate too.
                if _is_quota_error(None, error_body):
                    _record_quota_saturation(provider_name, provider_cfg, upstream_model, None)
                last_error = (error_body, 502, "text/event-stream")
                resp.close()
                logger.warning(
                    "  [%s] %s/%s -> 200 then stream error, trying next",
                    label, provider_name, upstream_model,
                )
                continue
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning("  [%s] %s/%s error mid-peek: %s, trying next", label, provider_name, upstream_model, e)
            resp.close()
            continue

        # Reactive capability detection (forced-tool/json 200-body checks) is
        # intentionally NOT applied beyond the first-chunk error peek: inspecting
        # delta.tool_calls would require buffering the whole SSE stream before
        # committing.  Proactive capability ordering still steers streaming
        # requests to capable models.
        if on_success is not None:
            on_success(provider_name, upstream_model, None, account_id)

        # Translation path: pipe the native stream through the adapters.
        if not (outbound.is_identity and inbound.is_identity):
            return _translated_stream_response(
                resp, outbound, inbound, provider_name, upstream_model, config,
                prefix=prefix, account_id=account_id,
            )

        captured_resp = resp
        captured_provider = provider_name
        captured_model = upstream_model
        captured_prefix = prefix
        captured_rest = rest

        @stream_with_context
        def generate(r=captured_resp, pn=captured_provider, um=captured_model,
                     pfx=captured_prefix, rst=captured_rest, acct=account_id):
            tail = bytearray()
            try:
                with r:
                    first = True
                    for chunk in itertools.chain((pfx,) if pfx else (), rst):
                        if chunk:
                            if first:
                                logger.info("  upstream %d  first chunk: %s", r.status_code, chunk[:200])
                                first = False
                            yield chunk
                            tail += chunk
                            if len(tail) > _STREAM_TAIL_BYTES:
                                del tail[:-_STREAM_TAIL_BYTES]
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                logger.error("[%s] provider=%s timed out mid-stream", label, pn)
                yield b'data: {"error":{"message":"Upstream stream timed out."}}\n\n'
            except Exception as e:
                logger.error("[%s] provider=%s mid-stream error: %s", label, pn, e)
                msg = str(e).replace('"', "'")
                yield f'data: {{"error":{{"message":"Upstream error: {msg}"}}}}\n\n'.encode()
            finally:
                _record_stream_usage(pn, um, bytes(tail), config, account_id=acct)

        return Response(generate(), content_type="text/event-stream")

    if last_error:
        body, status, ct = last_error
        return Response(body, status=status, content_type=ct)
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


def _expand_accounts(
    candidates: list[tuple[str, dict, str]],
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
    """
    if not candidates:
        return candidates
    expanded: list[tuple[str, dict, str]] = []
    for pn, pc, um in candidates:
        accounts = provider_accounts(pc)
        if len(accounts) <= 1:
            expanded.append((pn, pc, um))  # lone credential — leave cfg as-is
            continue
        fresh = [a for a in accounts if not _is_candidate_saturated(pn, um, a.id)]
        cooling = [a for a in accounts if _is_candidate_saturated(pn, um, a.id)]
        if provider_account_strategy(pc) == "round_robin" and len(fresh) > 1:
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
                 reasoning_map: dict[str, str]) -> tuple[int, float]:
    """Sophistication sort key for a candidate — higher is more capable.

    ``(reasoning_rank, param_count)`` where ``reasoning_rank`` is the configured
    ``model_reasoning`` tier (deep=2 > standard=1 > exploratory=0), falling back
    to a tier inferred from the model name when untagged, and ``param_count`` is
    the inferred size in billions. Sorting candidates by this key descending puts
    the most sophisticated model first.
    """
    lvl = (reasoning_map.get(upstream_id.lower())
           or reasoning_map.get(f"{provider_name}/{upstream_id}".lower())
           or _infer_reasoning_level(upstream_id))
    rank = _REASONING_LEVELS.index(lvl) if lvl in _REASONING_LEVELS else 0
    return (rank, _param_count(upstream_id))


def _quality_ordered_candidates(
    candidates: list[tuple[str, dict, str]],
    free_limits: dict[str, dict],
    reasoning_map: dict[str, str],
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
        scored.append(((pn, pc, um), score))

    def _key(item: tuple[tuple[str, dict, str], float]):
        (pn, _pc, um), score = item
        rank, params = _quality_key(pn, um, reasoning_map)
        return (rank, params, score)

    viable = sorted((it for it in scored if it[1] > 0.0), key=_key, reverse=True)
    exhausted = sorted((it for it in scored if it[1] == 0.0), key=_key, reverse=True)
    return [c for c, _ in viable] + [c for c, _ in exhausted]


# — "free" candidate selector —

def _normalized_believed_free(config: dict) -> set[str]:
    """
    Return a lowercased set of valid `believed_free` entries from *config*.

    Defensive against user-edited config.json: a missing field, ``None``,
    a non-list value, or non-string entries never raise — invalid shapes
    are logged once per call and silently dropped so a typo in config.json
    cannot turn /v1/models or /v1/chat/completions into a 500.
    """
    raw = config.get("believed_free")
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


def _normalized_cost_observed(config: dict) -> set[str]:
    """Lowercased set of config['cost_observed_free_tier'] qualified ids.

    These are models that served a request reporting a real cost while marked
    free; they are treated as paid everywhere from that moment on. Defensive
    against malformed config in the same spirit as _normalized_believed_free.
    """
    raw = config.get(COST_OBSERVED_KEY)
    if not isinstance(raw, list):
        return set()
    return {e.lower() for e in raw if isinstance(e, str)}


def _is_cost_observed(provider_name: str, upstream_id: str, config: dict) -> bool:
    """True when this model has been observed reporting a cost at runtime."""
    return f"{provider_name}/{upstream_id}".lower() in _normalized_cost_observed(config)


def _is_model_free(provider_name: str, upstream_id: str, config: dict) -> bool:
    """True when a model is treated as free-tier: its upstream id contains 'free'
    or it appears (bare or provider-qualified) in config['believed_free'].

    A model in config['cost_observed_free_tier'] is never free — a real cost was
    seen for it at runtime, so it is excluded here even if its id contains 'free'
    or it lingers in believed_free. This makes /free avoid it immediately.

    Shared by the /free candidate selector and the runtime cost flagger so both
    agree on what "free" means.
    """
    if _is_cost_observed(provider_name, upstream_id, config):
        return False
    believed_free = _normalized_believed_free(config)
    uid = upstream_id.lower()
    return (
        "free" in uid
        or uid in believed_free
        or f"{provider_name}/{upstream_id}".lower() in believed_free
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
    candidates = []
    for _proxy_id, (provider_name, upstream_id) in _get_route_cache_snapshot().items():
        provider_cfg = get_provider(config, provider_name)
        if not provider_cfg:
            continue
        if not _provider_exposes_to_virtual_models(provider_cfg):
            continue
        # Skip local providers — they belong to the /local family, not /free.
        if _is_local_url(provider_base_url(provider_cfg)):
            continue
        if _is_model_free(provider_name, upstream_id, config):
            candidates.append((provider_name, provider_cfg, upstream_id))
    return candidates


# — free-limits config parsing —

def _get_normalized_free_limits(config: dict) -> dict[str, dict]:
    """
    Return config['free_limits'] with all string keys lowercased.
    Ignores missing, non-dict, or malformed top-level values without raising.
    """
    raw = config.get("free_limits")
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
    for _proxy_id, (provider_name, upstream_id) in _get_route_cache_snapshot().items():
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
    raw = config.get("model_reasoning")
    if not isinstance(raw, dict):
        if raw is not None:
            logger.warning(
                "config['model_reasoning'] must be a dict; got %s — ignoring.",
                type(raw).__name__,
            )
        return {}
    result: dict[str, str] = {}
    for key, val in raw.items():
        if isinstance(key, str) and isinstance(val, str) and val.lower() in _REASONING_LEVELS:
            result[key.lower()] = val.lower()
        else:
            logger.warning(
                "config['model_reasoning']: invalid entry %r: %r (level must be one of %s) — skipping.",
                key, val, "/".join(_REASONING_LEVELS),
            )
    return result


def _get_reasoning_model_candidates(level: str) -> list[tuple[str, dict, str]]:
    """(provider_name, provider_cfg, upstream_model) for every model tagged with *level*."""
    config = load_config()
    reasoning = _get_model_reasoning(config)
    candidates = []
    for _proxy_id, (provider_name, upstream_id) in _get_route_cache_snapshot().items():
        lvl = (
            reasoning.get(upstream_id.lower())
            or reasoning.get(f"{provider_name}/{upstream_id}".lower())
        )
        if lvl == level:
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
    for _proxy_id, (provider_name, upstream_id) in _get_route_cache_snapshot().items():
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
    for _proxy_id, (provider_name, upstream_id) in _get_route_cache_snapshot().items():
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
    for _proxy_id, (pn, upstream_id) in _get_route_cache_snapshot().items():
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
            bucket = _quality_ordered_candidates(bucket, free_limits, reasoning_map)
            bucket = _apply_favorite_free_ordering(bucket, config)
        elif tier == _TIER_LOCAL:
            # $0 like free — prefer the strongest local model (e.g. the larger
            # Ollama model) rather than rotating randomly.
            bucket = sorted(
                bucket,
                key=lambda c: _quality_key(c[0], c[2], reasoning_map),
                reverse=True,
            )
        else:
            # Paid: cost first, then sophistication as a tiebreak among equals.
            bucket = sorted(
                bucket,
                key=lambda c: (_price(c), tuple(-x for x in _quality_key(c[0], c[2], reasoning_map))),
            )
        if needed:
            bucket = _order_by_capability(bucket, needed, cap_map)
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
            for _proxy_id, (pn, upstream_id) in _get_route_cache_snapshot().items()
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
                f"(upstream ID contains 'free', or add it to config['believed_free'])."
            )
        if dim in _REASONING_LEVELS:
            return f"Tag at least one of provider '{provider_name}'s models with '{dim}' in config['model_reasoning']."
        return f"Tag at least one of provider '{provider_name}'s models with '{dim}' in config['model_capabilities']."
    name = _strip_virtual_prefix(model_full)
    if name == "loadbalanced":
        return "Check that at least one provider exposes any model to virtual routing."
    if name == "free":
        return (
            "Check that at least one provider exposes a free-tier model "
            "(upstream ID contains 'free', or add it to config['believed_free'])."
        )
    if name == "local":
        return "Check that at least one provider has a localhost base_url."
    for level in _REASONING_LEVELS:
        if name == level:
            return f"Tag at least one model with '{level}' in config['model_reasoning']."
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
    for _proxy_id, (provider_name, upstream_id) in _get_route_cache_snapshot().items():
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
            pool = [
                c for c in pool
                if all(_model_has_capability(c[0], c[2], cap, cap_map) for cap in needed)
            ]
        else:  # "bypass": keep all, but order capable-first
            pool = _order_by_capability(pool, needed, cap_map)
    return pool


def _pick_aux_model(
    pool: list[tuple[str, dict, str]],
    explicit: str | None,
    config: dict,
    prefer_caps: frozenset[str] = frozenset(),
    exclude_first: tuple[str, dict, str] | None = None,
) -> tuple[str, dict, str] | None:
    """Choose a judge or synthesizer model.

    An explicit configured model id wins when it resolves; otherwise a model is
    auto-picked from *pool*, preferring one tagged with any of *prefer_caps*
    (e.g. reasoning) and, where possible, not the same model already chosen for
    the other stage (*exclude_first*) so the judge and synthesizer differ.
    """
    if explicit:
        pn, pc, uid, err = _resolve_provider(explicit)
        if err is None and pc is not None:
            return (pn, pc, uid)
        logger.warning("[fusion] configured model %r unresolved; auto-picking.", explicit)

    if not pool:
        return None
    cap_map = _model_capabilities(config)
    exclude_key = f"{exclude_first[0]}/{exclude_first[2]}" if exclude_first else None
    ranked = [
        (c, f"{c[0]}/{c[2]}",
         any(_model_has_capability(c[0], c[2], cap, cap_map) for cap in prefer_caps) if prefer_caps else False)
        for c in pool
    ]
    for c, key, has in ranked:
        if has and key != exclude_key:
            return c
    for c, key, _has in ranked:
        if key != exclude_key:
            return c
    return pool[0]


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
            resp = _proxy_request(
                endpoint, pn, pc, {**stripped, "model": uid}, candidate_timeout,
                forwarded_headers=forwarded_headers,
            )
            return cand, resp
        except Exception as e:  # noqa: BLE001
            print(f"[server:_proxy_fusion:panel] {pn}/{uid}: {e}")
            traceback.print_exc()
            return cand, None

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
            for cand, resp in results:
                pn, _pc, uid = cand
                key = f"{pn}/{uid}"
                if resp is not None and resp.status_code < 400:
                    body = resp.get_data()
                    text = _fusion.extract_message_text(body)
                    if text.strip():
                        _record_usage(pn, uid, usage=extract_usage(body), config=config)
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

    # 3. Judge compares the panel responses and emits structured analysis.
    judge_tuple = _pick_aux_model(pool, fcfg.get("judge_model"), config, prefer_caps=frozenset({"reasoning"}))
    analysis: dict | None = None
    judge_id: str | None = None
    if judge_tuple is not None:
        jpn, jpc, juid = judge_tuple
        judge_id = f"{jpn}/{juid}"
        jmsgs = _fusion.build_judge_messages(original_messages, panel_entries)
        try:
            jresp = _proxy_request(endpoint, jpn, jpc, {"model": juid, "messages": jmsgs}, candidate_timeout)
            if jresp.status_code < 400:
                _record_usage(jpn, juid, usage=extract_usage(jresp.get_data()), config=config)
                analysis = _fusion.parse_analysis(_fusion.extract_message_text(jresp.get_data()))
            else:
                logger.warning("  [fusion] judge %s returned %d", judge_id, jresp.status_code)
        except Exception as e:  # noqa: BLE001
            print(f"[server:_proxy_fusion:judge] {judge_id}: {e}")
            traceback.print_exc()

    # 4. Synthesizer writes the final answer grounded in the analysis.
    synth_tuple = _pick_aux_model(
        pool, fcfg.get("synthesizer_model"), config,
        prefer_caps=frozenset({"reasoning"}), exclude_first=judge_tuple,
    ) or panel_success[0][0]
    spn, spc, suid = synth_tuple
    synth_id = f"{spn}/{suid}"
    smsgs = _fusion.build_synthesizer_messages(original_messages, panel_entries, analysis)
    synth_payload = {**stripped, "model": suid, "messages": smsgs}

    def _report(fell_back: bool, with_analysis: bool) -> dict:
        return _fusion.build_report(
            panel_used=panel_used, judge_model=judge_id, synthesizer_model=synth_id,
            failed_models=failed_models, analysis=analysis if with_analysis else None,
            fell_back=fell_back, free=free,
        )

    header_report = json.dumps(_report(False, with_analysis=False), ensure_ascii=True)

    # Streaming: stream only the synthesis stage; provenance rides the header.
    if is_streaming:
        stream_timeout = server_cfg.get("stream_timeout", 300)
        resp = _proxy_streaming(
            endpoint, spn, spc, {**synth_payload, "stream": True},
            stream_timeout, config=config, inbound=inbound_adapter,
        )
        if getattr(resp, "status_code", 200) < 400:
            with contextlib.suppress(Exception):
                resp.headers["X-LLMProxy-Fusion"] = header_report
            return resp
        # Synth failed to start: degrade to the first panel answer (non-streamed).
        logger.warning("  [fusion] synth %s failed to stream; falling back to panel answer", synth_id)
        body = panel_success[0][1]
        out = _fusion.inject_report(body, _report(True, with_analysis=True))
        if not inbound_adapter.is_identity:
            out = inbound_adapter.render_response(out)
        resp = Response(out, status=200, content_type="application/json")
        with contextlib.suppress(Exception):
            resp.headers["X-LLMProxy-Fusion"] = header_report
        return resp

    # Non-streaming synthesis.
    try:
        sresp = _proxy_request(endpoint, spn, spc, synth_payload, timeout)
    except Exception as e:  # noqa: BLE001
        print(f"[server:_proxy_fusion:synth] {synth_id}: {e}")
        traceback.print_exc()
        sresp = None

    if sresp is None or sresp.status_code >= 400:
        # Graceful fallback: return the first successful panel response, flagged.
        logger.warning("  [fusion] synth %s failed; falling back to panel answer", synth_id)
        out = _fusion.inject_report(panel_success[0][1], _report(True, with_analysis=True))
    else:
        _record_usage(spn, suid, usage=extract_usage(sresp.get_data()), config=config)
        out = _fusion.inject_report(sresp.get_data(), _report(False, with_analysis=True))

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
    payload = inbound_adapter.to_canonical_request(raw_body)

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
                content, status, ct = cached
                logger.info("  [cache] HIT  key=%s…", cache_key[:12])
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
        if _is_loadbalanced_model(model_full):
            # Cost waterfall (free → local → paid), optimized per-prompt within
            # each tier. Owns its full ordering, so the capability/reasoning
            # passes below are stable no-ops over it.
            ordered = _loadbalanced_ordered_candidates(candidates, payload, config)
        elif is_free_virtual:
            free_limits = _get_normalized_free_limits(config)
            ordered = _capacity_ordered_candidates(candidates, free_limits)
        else:
            ordered = _cycling_candidates(candidates)

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
        if is_free_virtual or is_local_virtual:
            ordered = _order_by_request_fit(ordered, payload, _get_model_reasoning(config))
            logger.info("  [%s] request-fit first-pick tier=%s", model_full, _target_reasoning_tier(payload))
        if needed:
            ordered = _order_by_capability(ordered, needed, _model_capabilities(config))
        if is_free_virtual:
            ordered = _apply_favorite_free_ordering(ordered, config)
        # Expand accounts LAST: each model's credentials become adjacent
        # candidates in its ranked slot, so cycling rotates accounts-first then
        # models. A no-op for single-credential providers.
        ordered = _expand_accounts(ordered)
        logger.info("  [%s] cycling through %d candidate(s)", model_full, len(ordered))
        if is_streaming:
            timeout = server_cfg.get("stream_timeout", 300)
            return _proxy_cycling_streaming(
                endpoint, model_full, ordered, payload, timeout,
                on_success=on_success, config=config, inbound=inbound_adapter,
            )
        timeout = server_cfg.get("request_timeout", 120)
        resp = _proxy_cycling_non_streaming(endpoint, model_full, ordered, payload, timeout, on_success=on_success)
    else:
        provider_name, provider_cfg, upstream_model, err = _resolve_provider(model_full)
        if err is not None:
            return err

        logger.info("  provider=%s  model=%s", provider_name, upstream_model)
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
        resp = Response(
            inbound_adapter.render_response(resp.get_data()),
            status=resp.status_code,
            content_type="application/json",
        )

    # Store successful non-streaming responses in the short-lived cache (rendered
    # bytes, so a cache hit returns the correct dialect).
    if cache_key is not None and 200 <= resp.status_code < 300:
        _response_cache_put(cache_key, resp.get_data(), resp.status_code, resp.content_type, cache_ttl)
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
    """Proxy legacy text completions."""
    return _proxy_endpoint("completions")


@app.route("/v1/messages", methods=["POST"])
def anthropic_messages() -> Response:
    """Anthropic Messages API surface (supports streaming via Anthropic SSE).

    The request is translated to canonical OpenAI form, routed exactly like
    /v1/chat/completions (virtual models, capacity routing, native upstreams all
    apply), and the response is rendered back into the Anthropic Messages shape.
    """
    return _proxy_endpoint("chat/completions", inbound="anthropic")


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

def _build_usage_report() -> dict:
    """Snapshot the in-memory usage registry into a JSON-serializable report."""
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
        provider_name, _, upstream_model = key.partition("/")
        believed_free = bool(upstream_model) and _is_model_free(provider_name, upstream_model, config)
        models.append({
            "model": key,
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
        })
        for field in ("requests", "prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            totals[field] += snap[field]
    totals["cost"] = round(totals["cost"], 8)
    models.sort(key=lambda m: m["model"])

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
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
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
