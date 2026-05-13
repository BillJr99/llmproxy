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
All models exposed by this proxy use the form:
    <provider_name>/<upstream_model_id>

For example, if the provider is named "openrouter" and the upstream model
is "openrouter/free", the proxy model ID is "openrouter/openrouter/free".

The server strips the leading <provider_name>/ prefix before forwarding
each request to the appropriate upstream base URL.
"""

import hashlib
import json
import logging
import random
import threading
import time
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests
from flask import Flask, Response, g, jsonify, make_response, request, stream_with_context

from .config import (
    get_provider,
    load_config,
    model_is_allowed,
    parse_model_string,
)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)
logger = logging.getLogger("llmproxy.server")

# Maps proxy display ID -> (provider_name, upstream_id).
# Always access under _model_route_cache_lock.
_model_route_cache: dict[str, tuple[str, str]] = {}
_model_route_cache_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Short-lived response cache (non-streaming only)
# ---------------------------------------------------------------------------
# Keyed on SHA-256(endpoint + sorted JSON payload).  Only 2xx responses are
# stored.  Entries expire after server.response_cache_ttl seconds (default 120).

_response_cache: dict[str, tuple[bytes, int, str, float]] = {}
_response_cache_lock = threading.Lock()
_DEFAULT_RESPONSE_CACHE_TTL = 120
_RESPONSE_CACHE_MAX_ENTRIES = 512


def _response_cache_key(endpoint: str, payload: dict, auth: str = "") -> str:
    """Stable hash of the request, scoped by caller identity and excluding 'stream'."""
    filtered = {k: v for k, v in payload.items() if k != "stream"}
    raw = json.dumps({"_endpoint": endpoint, "_auth": auth, **filtered}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _response_cache_get(key: str, ttl: int) -> Optional[tuple[bytes, int, str]]:
    with _response_cache_lock:
        entry = _response_cache.get(key)
    if entry is None:
        return None
    content, status, content_type, ts = entry
    if time.monotonic() - ts > ttl:
        with _response_cache_lock:
            _response_cache.pop(key, None)
        return None
    return content, status, content_type


def _response_cache_put(key: str, content: bytes, status: int, content_type: str) -> None:
    with _response_cache_lock:
        _response_cache[key] = (content, status, content_type, time.monotonic())
        if len(_response_cache) > _RESPONSE_CACHE_MAX_ENTRIES:
            oldest = min(_response_cache, key=lambda k: _response_cache[k][3])
            _response_cache.pop(oldest, None)


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
        "Authorization": f"Bearer {provider_cfg['api_key']}",
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
# /v1/models  (GET)
# ---------------------------------------------------------------------------

def _fetch_provider_models(provider_name: str, provider_cfg: dict, timeout: int) -> list[dict]:
    """
    Fetch the model list from a single provider, apply any configured filter,
    and prefix each model ID with '<provider_name>/'.

    Returns an empty list on any failure so that one bad provider does not
    prevent the aggregate response from including all healthy providers.
    """
    base_url = provider_cfg.get("base_url", "").rstrip("/")
    url = f"{base_url}/models"
    headers = {
        "Authorization": f"Bearer {provider_cfg.get('api_key', '')}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(
            "[server:_fetch_provider_models] provider=%s fetch failed: %s",
            provider_name, e,
        )
        return []

    raw_models: list[dict] = data.get("data", [])
    model_filter = provider_cfg.get("model_filter")

    result = []
    for model in raw_models:
        upstream_id: str = model.get("id", "")
        if model_filter is not None and upstream_id not in model_filter:
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
        # Use "model (provider)" as the proxy ID — shows fully in client menus
        # without being silently truncated at a "/" boundary.
        proxy_id = f"{stripped} ({provider_name})"
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
        futures = {
            executor.submit(_fetch_provider_models, name, cfg, timeout): name
            for name, cfg in providers_cfg.items()
        }
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


@app.route("/v1/models", methods=["GET"])
def list_models() -> Response:
    """
    Aggregate model listings from all configured providers.

    Each provider is queried concurrently.  Providers that fail are logged as
    warnings and omitted rather than causing an overall failure.  The route
    cache is rebuilt atomically on each call so stale entries do not linger.

    Two synthetic virtual models are prepended when their backing candidates
    exist: 'free' (cycles through models whose ID contains 'free') and
    'local' (cycles through models on localhost providers).
    """
    config = load_config()
    providers: dict = config.get("providers", {})
    server_cfg: dict = config.get("server", {})
    timeout: int = server_cfg.get("request_timeout", 120)

    if not providers:
        return jsonify({
            "object": "list",
            "data": [],
            "_warning": "No providers configured. Run 'llmproxy --setup'.",
        })

    all_models = _rebuild_route_cache(providers, timeout)

    # Prepend synthetic virtual models when their backing candidates exist.
    with _model_route_cache_lock:
        snapshot = dict(_model_route_cache)
    synthetic: list[dict] = []
    if any("free" in uid.lower() for _, uid in snapshot.values()):
        synthetic.append({
            "id": "free",
            "object": "model",
            "owned_by": "llmproxy",
            "name": "free",
            "_note": "Virtual model: cycles through all models whose ID contains 'free' until one succeeds.",
        })
    if any(
        _is_local_url(get_provider(config, pn).get("base_url", ""))
        for pn, _ in snapshot.values()
        if get_provider(config, pn)
    ):
        synthetic.append({
            "id": "local",
            "object": "model",
            "owned_by": "llmproxy",
            "name": "local",
            "_note": "Virtual model: cycles through all models served on localhost until one succeeds.",
        })

    return jsonify({
        "object": "list",
        "data": synthetic + all_models,
    })


@app.route("/v1/models/free", methods=["GET"])
def get_free_model() -> Response:
    """Return metadata for the synthetic 'free' cycling model."""
    candidates = _get_free_model_candidates()
    return jsonify({
        "id": "free",
        "object": "model",
        "owned_by": "llmproxy",
        "name": "free",
        "_note": "Virtual model: cycles through all models whose ID contains 'free' until one succeeds.",
        "_candidates": [f"{pn}/{um}" for pn, _, um in candidates],
    })


@app.route("/v1/models/local", methods=["GET"])
def get_local_model() -> Response:
    """Return metadata for the synthetic 'local' cycling model."""
    candidates = _get_local_model_candidates()
    return jsonify({
        "id": "local",
        "object": "model",
        "owned_by": "llmproxy",
        "name": "local",
        "_note": "Virtual model: cycles through all models served on localhost until one succeeds.",
        "_candidates": [f"{pn}/{um}" for pn, _, um in candidates],
    })


@app.route("/v1/models/<path:model_id>", methods=["GET"])
def get_model(model_id: str) -> Response:
    """
    Return metadata for a single proxy model ID.

    Accepts both the display format returned by /v1/models ("model (provider)")
    and the slash format ("provider/upstream_model").  The route cache is
    checked first so display-format IDs resolve correctly without parsing.
    """
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
    base_url = provider_cfg.get("base_url", "").rstrip("/")
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
    base_url = provider_cfg.get("base_url", "").rstrip("/")
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
# Virtual models — shared cycling logic + per-model candidate selectors
# ---------------------------------------------------------------------------

def _proxy_cycling_non_streaming(
    endpoint: str,
    label: str,
    candidates: list[tuple[str, dict, str]],
    payload: dict,
    timeout: int,
) -> Response:
    """Try each candidate in order, returning the first success."""
    last: Optional[Response] = None
    for provider_name, provider_cfg, upstream_model in candidates:
        upstream_payload = {**payload, "model": upstream_model}
        logger.info("  [%s] trying %s/%s", label, provider_name, upstream_model)
        resp = _proxy_request(endpoint, provider_name, provider_cfg, upstream_payload, timeout)
        if resp.status_code < 400:
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
) -> Response:
    """
    Try each candidate in order.  Checks the HTTP status code before committing
    to stream the response so failed upstreams are skipped transparently.
    When all candidates fail the last upstream error body is returned so
    clients receive the same diagnostic information as the non-streaming path.
    """
    last_error: Optional[tuple[bytes, int, str]] = None

    for provider_name, provider_cfg, upstream_model in candidates:
        upstream_payload = {**payload, "model": upstream_model}
        base_url = provider_cfg.get("base_url", "").rstrip("/")
        url = f"{base_url}/{endpoint}"
        headers = _upstream_headers(provider_cfg)
        logger.info("  [%s] trying %s/%s  [streaming]", label, provider_name, upstream_model)
        try:
            resp = requests.post(url, headers=headers, json=upstream_payload, stream=True, timeout=timeout)
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


# — "free" candidate selector —

def _get_free_model_candidates() -> list[tuple[str, dict, str]]:
    """(provider_name, provider_cfg, upstream_model) for every model whose upstream ID contains 'free'."""
    config = load_config()
    candidates = []
    for _proxy_id, (provider_name, upstream_id) in _get_route_cache_snapshot().items():
        if "free" in upstream_id.lower():
            provider_cfg = get_provider(config, provider_name)
            if provider_cfg:
                candidates.append((provider_name, provider_cfg, upstream_id))
    return candidates


# — "local" candidate selector —

def _is_local_url(base_url: str) -> bool:
    """Return True when *base_url* points at a loopback address."""
    try:
        hostname = urllib.parse.urlparse(base_url).hostname or ""
    except Exception:
        return False
    hostname = hostname.strip("[]").lower()
    return hostname in ("localhost", "127.0.0.1", "::1") or hostname.startswith("127.")


def _get_local_model_candidates() -> list[tuple[str, dict, str]]:
    """(provider_name, provider_cfg, upstream_model) for every model whose provider base_url is localhost."""
    config = load_config()
    candidates = []
    for _proxy_id, (provider_name, upstream_id) in _get_route_cache_snapshot().items():
        provider_cfg = get_provider(config, provider_name)
        if provider_cfg and _is_local_url(provider_cfg.get("base_url", "")):
            candidates.append((provider_name, provider_cfg, upstream_id))
    return candidates


# ---------------------------------------------------------------------------
# Shared routing logic for all proxied endpoints
# ---------------------------------------------------------------------------

def _resolve_provider(model_full: str) -> tuple[Optional[str], Optional[dict], Optional[str], Optional[Response]]:
    """
    Parse *model_full* into (provider_name, provider_cfg, upstream_model).

    Returns a 4-tuple where the last element is an error Response if
    resolution fails, otherwise None.  Callers should check the last element
    before using the first three.
    """
    config = load_config()

    # Cache-first: the display ID format "model (provider)" is not parseable by
    # parse_model_string, so the cache (populated by /v1/models) is authoritative.
    with _model_route_cache_lock:
        cached_route = _model_route_cache.get(model_full)
    if cached_route:
        provider_name, upstream_model = cached_route
    elif model_full.endswith(")") and " (" in model_full:
        # Cold-cache fallback for "model (provider)" format.
        model_part, _, provider_name = model_full[:-1].rpartition(" (")
        upstream_model = model_part  # may be missing a provider prefix; best-effort
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
    cache_key: Optional[str] = None
    if not is_streaming:
        cache_ttl: int = server_cfg.get("response_cache_ttl", _DEFAULT_RESPONSE_CACHE_TTL)
        if cache_ttl > 0:
            cache_key = _response_cache_key(endpoint, payload, request.headers.get("Authorization", ""))
            cached = _response_cache_get(cache_key, cache_ttl)
            if cached:
                content, status, ct = cached
                logger.info("  [cache] HIT  key=%s…", cache_key[:12])
                return Response(content, status=status, content_type=ct)

    if model_full in ("free", "local"):
        candidates = (
            _get_free_model_candidates() if model_full == "free"
            else _get_local_model_candidates()
        )
        if not candidates:
            hint = (
                "Check that at least one provider exposes a free-tier model."
                if model_full == "free"
                else "Check that at least one provider has a localhost base_url."
            )
            return _error(
                f"No '{model_full}' models are currently available. {hint}",
                status=503,
            )
        candidates = _cycling_candidates(candidates)
        logger.info("  [%s] cycling through %d candidate(s)", model_full, len(candidates))
        if is_streaming:
            timeout = server_cfg.get("stream_timeout", 300)
            return _proxy_cycling_streaming(endpoint, model_full, candidates, payload, timeout)
        timeout = server_cfg.get("request_timeout", 120)
        resp = _proxy_cycling_non_streaming(endpoint, model_full, candidates, payload, timeout)
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
        _response_cache_put(cache_key, resp.get_data(), resp.status_code, resp.content_type)
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

        base_url = provider_cfg.get("base_url", "").rstrip("/")
        url = f"{base_url}/{subpath}"
        headers = {
            "Authorization": f"Bearer {provider_cfg.get('api_key', '')}",
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

def run_server(config_path: Optional[str] = None) -> None:
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
