"""
server.py — OpenAI-compatible proxy server for llmproxy.

Implements the following OpenAI API endpoints:
  GET  /v1/models                  Aggregate models from all providers
  GET  /v1/models/<model_id>       Single model metadata lookup
  POST /v1/chat/completions        Proxy chat completions (streaming + non-streaming)
  POST /v1/completions             Proxy legacy completions
  POST /v1/embeddings              Proxy embeddings
  GET  /health                     Health check

Model naming convention
-----------------------
GET /v1/models advertises every model in the display form:
    <provider_name>__<upstream_model_id>

For example, an "ollama" provider serving "qwen2.5vl:3b" is shown as
"ollama__qwen2.5vl:3b".  Spaces in either side are replaced with "_".

Upstream ids that contain multiple slashes are flattened so the display id
carries at most one "/": all but the last slash become "_".  For example an
"openrouter" provider serving "meta-llama/llama-3/instruct" is shown as
"openrouter__meta-llama_llama-3/instruct".  This keeps "__" the unambiguous
provider separator and leaves at most a single "/" in any id.  Routing always
uses the original (un-flattened) upstream id when forwarding upstream.

Three additional input forms also resolve on every proxied endpoint, for
backward compatibility with pinned client configs:
    <provider_name>/<upstream_model_id>     (canonical slash form)
    <upstream_model_id>__<provider_name>    (PR #27 legacy display form)
    <upstream_model_id> (<provider_name>)   (pre-PR #27 legacy display form)

The server strips the provider prefix/suffix before forwarding each request
to the appropriate upstream base URL.
"""

import collections
import datetime
import hashlib
import json
import logging
import random
import threading
import time
import traceback
import urllib.parse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, Response, g, jsonify, make_response, request, stream_with_context

from . import __version__
from .config import (
    RESERVED_PROVIDER_NAMES,
    get_provider,
    load_config,
    model_is_allowed,
    parse_model_string,
    provider_api_key,
    provider_base_url,
    resolve_env_refs,
    save_config,
)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)
logger = logging.getLogger("llmproxy.server")

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
# Virtual models use the "llmproxy__" prefix (same double-underscore as the
# provider display form) so strict clients accept them and they sort together.
# The legacy "llmproxy/" prefix is kept in the membership set so pinned client
# configs continue to resolve; only the new form is advertised in /v1/models.
_NEW_VIRTUAL_MODELS: frozenset[str] = frozenset({
    "llmproxy__free", "llmproxy__local",
    *(f"llmproxy__{lvl}" for lvl in _REASONING_LEVELS),
    *(f"llmproxy__{lvl}/free" for lvl in _REASONING_LEVELS),
    *(f"llmproxy__{lvl}/local" for lvl in _REASONING_LEVELS),
    *(f"llmproxy__{cap}" for cap in _CAPABILITY_VIRTUALS),
    *(f"llmproxy__{cap}/free" for cap in _CAPABILITY_VIRTUALS),
})
_LEGACY_VIRTUAL_MODELS: frozenset[str] = frozenset({
    "llmproxy/free", "llmproxy/local",
    *(f"llmproxy/{lvl}" for lvl in _REASONING_LEVELS),
    *(f"llmproxy/{lvl}/free" for lvl in _REASONING_LEVELS),
    *(f"llmproxy/{lvl}/local" for lvl in _REASONING_LEVELS),
    *(f"llmproxy/{cap}" for cap in _CAPABILITY_VIRTUALS),
    *(f"llmproxy/{cap}/free" for cap in _CAPABILITY_VIRTUALS),
})
_VIRTUAL_MODELS: frozenset[str] = _NEW_VIRTUAL_MODELS | _LEGACY_VIRTUAL_MODELS
# Virtual models that use capacity-aware free-tier load balancing.
_FREE_VIRTUAL_MODELS: frozenset[str] = frozenset({
    "llmproxy__free", "llmproxy/free",
    *(f"llmproxy__{lvl}/free" for lvl in _REASONING_LEVELS),
    *(f"llmproxy/{lvl}/free" for lvl in _REASONING_LEVELS),
    *(f"llmproxy__{cap}/free" for cap in _CAPABILITY_VIRTUALS),
    *(f"llmproxy/{cap}/free" for cap in _CAPABILITY_VIRTUALS),
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

# ---------------------------------------------------------------------------
# Per-model usage tracking (free-tier capacity-aware load balancing)
# ---------------------------------------------------------------------------
# In-memory only; resets on server restart.  Each gunicorn worker process
# maintains its own counters — usage tracking is per-worker, not cross-process.
# For cross-process accuracy, configure a single worker or use a shared store.


def _today_start_ts() -> float:
    """Unix timestamp for the start of today (local midnight)."""
    return time.mktime(datetime.date.today().timetuple())


class _ModelUsage:
    """Thread-safe sliding-window request counter for one upstream model."""

    __slots__ = ("_lock", "_minute_ts", "_day_count", "_day_start")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._minute_ts: collections.deque = collections.deque()
        self._day_count: int = 0
        self._day_start: float = _today_start_ts()

    def record(self) -> None:
        now_mono = time.monotonic()
        now_wall = time.time()
        with self._lock:
            if now_wall >= self._day_start + 86400:
                self._day_start = _today_start_ts()
                self._day_count = 0
            self._day_count += 1
            self._minute_ts.append(now_mono)

    def snapshot(self) -> tuple[int, int]:
        """Return (used_last_60s, used_today), pruning stale minute entries."""
        now_mono = time.monotonic()
        now_wall = time.time()
        with self._lock:
            if now_wall >= self._day_start + 86400:
                return 0, 0
            cutoff = now_mono - 60.0
            while self._minute_ts and self._minute_ts[0] < cutoff:
                self._minute_ts.popleft()
            return len(self._minute_ts), self._day_count


_usage_registry: dict[str, _ModelUsage] = {}
_usage_registry_lock = threading.Lock()


def _record_usage(provider_name: str, upstream_model: str) -> None:
    """Record one successful request for a free-tier model."""
    key = f"{provider_name}/{upstream_model}".lower()
    with _usage_registry_lock:
        if key not in _usage_registry:
            _usage_registry[key] = _ModelUsage()
        tracker = _usage_registry[key]
    tracker.record()


def _get_usage_snapshot(key: str) -> tuple[int, int]:
    """Return (used_last_60s, used_today) for the given provider/model key."""
    with _usage_registry_lock:
        tracker = _usage_registry.get(key)
    return tracker.snapshot() if tracker else (0, 0)


# ---------------------------------------------------------------------------
# Local provider startup sync
# ---------------------------------------------------------------------------
# Tracks whether the one-time local model sync has run since startup.
_local_sync_done: bool = False
_local_sync_lock = threading.Lock()

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
    headers = {
        "Authorization": f"Bearer {provider_api_key(provider_cfg)}",
        "Content-Type": "application/json",
    }
    for header in _FORWARDED_REQUEST_HEADERS - {"Content-Type"}:
        value = request.headers.get(header)
        if value:
            headers[header] = value
    return headers


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
    headers = {
        "Authorization": f"Bearer {provider_api_key(provider_cfg)}",
        "Content-Type": "application/json",
    }

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


def _rebuild_route_cache(providers_cfg: dict, timeout: int) -> list[dict]:
    """
    Fetch models from all providers concurrently, rebuild _model_route_cache
    atomically, and return the full flat model list.

    The cache is replaced wholesale on each call so that removed or renamed
    upstream models do not linger as stale entries.
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
            new_cache[m["id"]] = route

    with _model_route_cache_lock:
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
    _rebuild_route_cache(providers_cfg, timeout)

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
            try:
                resp = requests.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
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


def _infer_local_reasoning_level(model_id: str) -> str:
    """
    Infer exploratory / standard / deep for a locally-served model.
    Applied during startup sync for Ollama / OpenWebUI models.
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
    providers: dict = config.get("providers", {})
    # Strip reserved names before presence check so a config that contains only
    # reserved providers triggers the "no providers configured" warning rather
    # than returning a silently empty model list.
    providers = {k: v for k, v in providers.items() if k not in RESERVED_PROVIDER_NAMES}
    server_cfg: dict = config.get("server", {})
    timeout: int = server_cfg.get("request_timeout", 120)
    models_ttl: int = server_cfg.get("models_cache_ttl", _DEFAULT_MODELS_CACHE_TTL)

    if not providers:
        return jsonify({
            "object": "list",
            "data": [],
            "_warning": "No providers configured. Run 'llmproxy --setup'.",
        })

    # Return cached model list if still fresh.
    global _models_list_cache
    if models_ttl > 0:
        with _models_list_cache_lock:
            if _models_list_cache is not None:
                cached_data, cached_ts = _models_list_cache
                if time.monotonic() - cached_ts < models_ttl:
                    logger.info("  [models cache] HIT (%.0fs old)", time.monotonic() - cached_ts)
                    return jsonify({"object": "list", "data": cached_data})

    all_models = _rebuild_route_cache(providers, timeout)

    # Prepend synthetic virtual models when their backing candidates exist.
    with _model_route_cache_lock:
        snapshot = dict(_model_route_cache)
    synthetic: list[dict] = []
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

    # Annotate real models with (believed_free) and/or (local) suffixes in name.
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

    full_list = synthetic + all_models
    if models_ttl > 0:
        with _models_list_cache_lock:
            _models_list_cache = (full_list, time.monotonic())

    return jsonify({
        "object": "list",
        "data": full_list,
    })


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
    if _is_virtual_model(model_id):
        candidates = _get_virtual_candidates(model_id)
        return jsonify({
            "id": model_id,
            "object": "model",
            "owned_by": "llmproxy",
            "name": model_id,
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
    new_routes = {m["id"]: m.pop("_route") for m in provider_models if "_route" in m}
    with _model_route_cache_lock:
        _model_route_cache.update(new_routes)

    # Match by proxy display ID or by upstream model ID (clients may use either).
    for m in provider_models:
        if m.get("id") == model_id or m.get("_upstream_id") == upstream_model:
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
) -> Response:
    """
    Forward a non-streaming request to the upstream provider and return its
    response verbatim (status code, body, content-type).

    Parameters
    ----------
    endpoint : str
        The API path suffix, e.g. 'chat/completions'.
    provider_name : str
        Used only for error message attribution.
    provider_cfg : dict
        Provider configuration (base_url, api_key).
    payload : dict
        Request body to forward (with the upstream model ID already set).
    timeout : int
        Request timeout in seconds.
    """
    base_url = provider_base_url(provider_cfg)
    url = f"{base_url}/{endpoint}"
    headers = _upstream_headers(provider_cfg)

    logger.info("  upstream POST %s  model=%s", url, payload.get("model", "?"))
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        logger.info("  upstream %d  %.0fms", resp.status_code, resp.elapsed.total_seconds() * 1000)
        content_type = resp.headers.get("Content-Type", "application/json")
        return Response(resp.content, status=resp.status_code, content_type=content_type)
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

def _proxy_streaming(
    endpoint: str,
    provider_name: str,
    provider_cfg: dict,
    payload: dict,
    timeout: int,
) -> Response:
    """
    Forward a streaming request to the upstream and relay the raw SSE byte
    stream back to the client without buffering.

    The `stream_with_context` wrapper ensures that the generator holds a
    reference to the Flask application context throughout the lifetime of the
    response, which is required when teardown hooks are present.
    """
    base_url = provider_base_url(provider_cfg)
    url = f"{base_url}/{endpoint}"
    headers = _upstream_headers(provider_cfg)

    logger.info("  upstream POST %s  model=%s  [streaming]", url, payload.get("model", "?"))

    @stream_with_context
    def generate():
        try:
            with requests.post(
                url,
                headers=headers,
                json=payload,
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
        except requests.exceptions.Timeout:
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


# ---------------------------------------------------------------------------
# Virtual models — shared cycling logic + per-model candidate selectors
# ---------------------------------------------------------------------------

def _proxy_cycling_non_streaming(
    endpoint: str,
    label: str,
    candidates: list[tuple[str, dict, str]],
    payload: dict,
    timeout: int,
    on_success: Callable[[str, str], None] | None = None,
) -> Response:
    """Try each candidate in order, returning the first success.

    A 200 that fails to deliver a *forced* capability (e.g. ``tool_choice``
    forced a tool call but the body has none) is treated as a failure and the
    next candidate is tried, mirroring HTTP-error failover.  When every
    candidate is exhausted the last response is returned so the client still
    receives the real upstream body rather than a synthesized error.
    """
    candidate_timeout = min(timeout, _VIRTUAL_CANDIDATE_TIMEOUT)
    last: Response | None = None
    for provider_name, provider_cfg, upstream_model in candidates:
        upstream_payload = {**payload, "model": upstream_model}
        logger.info("  [%s] trying %s/%s", label, provider_name, upstream_model)
        resp = _proxy_request(endpoint, provider_name, provider_cfg, upstream_payload, candidate_timeout)
        if resp.status_code < 400:
            if _capability_failed(payload, resp.get_data()):
                logger.warning(
                    "  [%s] %s/%s returned 200 but did not honor a forced capability, trying next",
                    label, provider_name, upstream_model,
                )
                last = resp
                continue
            if on_success is not None:
                on_success(provider_name, upstream_model)
            return resp
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
    on_success: Callable[[str, str], None] | None = None,
) -> Response:
    """
    Try each candidate in order.  Checks the HTTP status code before committing
    to stream the response so failed upstreams are skipped transparently.
    When all candidates fail the last upstream error body is returned so
    clients receive the same diagnostic information as the non-streaming path.
    """
    candidate_timeout = min(timeout, _VIRTUAL_CANDIDATE_TIMEOUT)
    last_error: tuple[bytes, int, str] | None = None

    for provider_name, provider_cfg, upstream_model in candidates:
        upstream_payload = {**payload, "model": upstream_model}
        base_url = provider_base_url(provider_cfg)
        url = f"{base_url}/{endpoint}"
        headers = _upstream_headers(provider_cfg)
        logger.info("  [%s] trying %s/%s  [streaming]", label, provider_name, upstream_model)
        try:
            resp = requests.post(url, headers=headers, json=upstream_payload, stream=True, timeout=candidate_timeout)
            if resp.status_code >= 400:
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

            # Reactive capability detection (forced-tool/json 200-body checks) is
            # intentionally NOT applied to streaming: inspecting delta.tool_calls
            # would require buffering the SSE stream before committing the first
            # chunk, defeating the zero-latency passthrough below.  Proactive
            # capability ordering still steers streaming requests to capable models.
            if on_success is not None:
                on_success(provider_name, upstream_model)

            captured_resp = resp
            captured_provider = provider_name

            @stream_with_context
            def generate(r=captured_resp, pn=captured_provider):
                try:
                    with r:
                        first = True
                        for chunk in r.iter_content(chunk_size=None):
                            if chunk:
                                if first:
                                    logger.info("  upstream %d  first chunk: %s", r.status_code, chunk[:200])
                                    first = False
                                yield chunk
                except requests.exceptions.Timeout:
                    logger.error("[%s] provider=%s timed out mid-stream", label, pn)
                    yield b'data: {"error":{"message":"Upstream stream timed out."}}\n\n'
                except Exception as e:
                    logger.error("[%s] provider=%s mid-stream error: %s", label, pn, e)
                    msg = str(e).replace('"', "'")
                    yield f'data: {{"error":{{"message":"Upstream error: {msg}"}}}}\n\n'.encode()

            return Response(generate(), content_type="text/event-stream")

        except requests.exceptions.Timeout:
            logger.warning("  [%s] %s/%s timed out, trying next", label, provider_name, upstream_model)
            continue
        except Exception as e:
            logger.warning("  [%s] %s/%s error: %s, trying next", label, provider_name, upstream_model, e)
            continue

    if last_error:
        body, status, ct = last_error
        return Response(body, status=status, content_type=ct)
    return _error(f"All '{label}' model candidates failed or are unavailable.", status=503)


def _cycling_candidates(
    candidates: list[tuple[str, dict, str]],
) -> list[tuple[str, dict, str]]:
    """Rotate candidates to a random starting position for load spreading."""
    if not candidates:
        return candidates
    start = random.randrange(len(candidates))
    return candidates[start:] + candidates[:start]


def _capacity_ordered_candidates(
    candidates: list[tuple[str, dict, str]],
    free_limits: dict[str, dict],
) -> list[tuple[str, dict, str]]:
    """
    Order candidates by remaining free-tier capacity using weighted sampling.

    Algorithm:
    - Each candidate is scored via _capacity_score() using its RPM/RPD usage.
    - Candidates with no configured limits score 1.0 (treated as unlimited).
    - Candidates with score > 0 are drawn via weighted reservoir sampling so
      higher-capacity models are preferred while load is still distributed.
    - Candidates with score == 0 (at limit) are appended as last-resort fallbacks;
      they still get tried so a saturated model doesn't cause an avoidable 503.
    - Falls back to random rotation when no candidate has any configured limits.

    Note: tracking is per-worker-process; gunicorn multi-worker deployments
    may undercount usage relative to the provider's actual view.
    """
    if not candidates:
        return candidates

    any_limits = any(
        f"{pn}/{um}".lower() in free_limits for pn, _, um in candidates
    )
    if not any_limits:
        start = random.randrange(len(candidates))
        return candidates[start:] + candidates[:start]

    scored: list[tuple[tuple[str, dict, str], float]] = []
    for pn, pc, um in candidates:
        key = f"{pn}/{um}".lower()
        limits = free_limits.get(key, {})
        used_min, used_day = _get_usage_snapshot(key)
        score = _capacity_score(used_min, used_day, limits)
        logger.debug(
            "[capacity] %s/%s  score=%.3f  used_min=%d  used_day=%d",
            pn, um, score, used_min, used_day,
        )
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


def _get_free_model_candidates() -> list[tuple[str, dict, str]]:
    """(provider_name, provider_cfg, upstream_model) for every model whose upstream ID contains 'free' or appears in config['believed_free'].

    Models served from a localhost / loopback URL are NEVER included — they
    route via the dedicated llmproxy__local family instead. This is a
    defence-in-depth guard against stale configs that still have local models
    in believed_free; the startup local-sync cleans those up too, but this
    runtime filter ensures /free never leaks a local model even before sync runs.
    """
    config = load_config()
    believed_free = _normalized_believed_free(config)
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
        is_free = (
            "free" in upstream_id.lower()
            or upstream_id.lower() in believed_free
            or f"{provider_name}/{upstream_id}".lower() in believed_free
        )
        if is_free:
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


def _capacity_score(used_minute: int, used_day: int, limits: dict) -> float:
    """
    Return a capacity score in [0.0, 1.0]: higher = more remaining headroom.
    Returns 1.0 (neutral) when neither rpm nor rpd is configured.
    Returns 0.0 when any configured limit is at or exceeded.
    Token limits (tpm/tpd) are stored for reference but not enforced here.
    """
    rpm = limits.get("requests_per_minute")
    rpd = limits.get("requests_per_day")
    if not rpm and not rpd:
        return 1.0
    scores: list[float] = []
    if rpm and rpm > 0:
        scores.append(max(0.0, (rpm - used_minute) / rpm))
    if rpd and rpd > 0:
        scores.append(max(0.0, (rpd - used_day) / rpd))
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


def _is_free_virtual_model(model_full: str) -> bool:
    """True for capacity-aware free virtuals (global free set or <provider>/free)."""
    if model_full in _FREE_VIRTUAL_MODELS:
        return True
    split = _split_per_provider_virtual(model_full)
    return split is not None and split[1] == "free"


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


def _proxy_endpoint(endpoint: str) -> Response:
    """
    Generic handler that routes a POST request to the correct upstream provider.

    Reads the 'model' field from the JSON body, resolves the provider, and
    delegates to the streaming or non-streaming proxy helper.  For non-streaming
    requests, successful responses are stored in a short-lived cache so that
    harnesses which replay the same request in quick succession avoid redundant
    upstream round-trips.

    The special model names "free" and "local" cycle through all matching
    cached models until one returns a successful response.
    """
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return _error("Request body must be valid JSON.", status=400)

    model_full: str = payload.get("model", "")
    if not model_full:
        return _error("Request body must include a 'model' field.", status=400)

    config = load_config()
    server_cfg = config.get("server", {})
    is_streaming: bool = payload.get("stream", False)

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
        candidates = _get_virtual_candidates(model_full)
        if not candidates:
            return _error(
                f"No '{model_full}' models are currently available. "
                + _virtual_model_hint(model_full),
                status=503,
            )
        if _is_free_virtual_model(model_full):
            free_limits = _get_normalized_free_limits(config)
            ordered = _capacity_ordered_candidates(candidates, free_limits)
            on_success: Callable[[str, str], None] | None = _record_usage
        else:
            ordered = _cycling_candidates(candidates)
            on_success = None
        # Proactively prefer candidates that support the capabilities this
        # request needs (tools/vision/reasoning/json).  Stable, never drops
        # candidates, and a no-op when nothing is needed or no metadata exists.
        needed = _needed_capabilities(payload)
        if needed:
            ordered = _order_by_capability(ordered, needed, _model_capabilities(config))
        logger.info("  [%s] cycling through %d candidate(s)", model_full, len(ordered))
        if is_streaming:
            timeout = server_cfg.get("stream_timeout", 300)
            return _proxy_cycling_streaming(endpoint, model_full, ordered, payload, timeout, on_success=on_success)
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
            return _proxy_streaming(endpoint, provider_name, provider_cfg, upstream_payload, timeout)
        timeout = server_cfg.get("request_timeout", 120)
        resp = _proxy_request(endpoint, provider_name, provider_cfg, upstream_payload, timeout)

    # Store successful non-streaming responses in the short-lived cache.
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


@app.route("/v1/embeddings", methods=["POST"])
def embeddings() -> Response:
    """Proxy embeddings requests (streaming not applicable)."""
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return _error("Request body must be valid JSON.", status=400)

    model_full: str = payload.get("model", "")
    if not model_full:
        return _error("Request body must include a 'model' field.", status=400)

    provider_name, provider_cfg, upstream_model, err = _resolve_provider(model_full)
    if err is not None:
        return err

    config = load_config()
    timeout = config.get("server", {}).get("request_timeout", 120)
    upstream_payload = {**payload, "model": upstream_model}
    return _proxy_request("embeddings", provider_name, provider_cfg, upstream_payload, timeout)


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
        headers = {
            "Authorization": f"Bearer {provider_api_key(provider_cfg)}",
        }
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

    # Pre-warm the model route cache so routing works even before the first
    # /v1/models call (eliminates the cold-cache edge case for all clients).
    timeout = server_cfg.get("request_timeout", 120)
    _rebuild_route_cache(providers_cfg, timeout)

    app.run(host=host, port=port, threaded=True, debug=False)


# ---------------------------------------------------------------------------
# Web admin UI / config API (/admin, /admin/api/*)
# ---------------------------------------------------------------------------
# Registered at import time so the blueprint is present whether the app is run
# via gunicorn (which imports `app` directly) or the Flask dev server. The
# blueprint's before_request guard enforces the localhost-only / token policy.
from .admin import register_admin  # noqa: E402  (deferred to avoid import cycle)

register_admin(app)
