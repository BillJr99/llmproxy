"""
admin.py — Web admin UI and JSON configuration API for llmproxy.

Exposes a single-page admin frontend at ``/admin`` plus a JSON API under
``/admin/api/*`` that can edit everything in config.json: server settings,
providers (add/edit/delete, add-from-template, live model discovery), the
model categorizations that drive the virtual endpoints (believed_free,
model_reasoning, model_capabilities, free_limits), and a derived preview of the
virtual endpoints those categorizations produce.

Security model
--------------
The UI *shell* (the HTML/CSS/JS at ``/admin`` and ``/admin/static/*``) carries
no secrets and is served without authentication so the browser can load it with
a plain navigation (no custom headers possible on a document/asset request).

Every data endpoint under ``/admin/api/*`` is guarded by ``_require_auth``:

  * If an admin token is configured (``config['admin']['token']`` — which may
    itself be a ``${VAR}`` env reference — or the ``LLMPROXY_ADMIN_TOKEN``
    environment variable), the request must present it via
    ``Authorization: Bearer <token>`` or an ``X-Admin-Token`` header. Any origin
    that presents the correct token is allowed.
  * If no token is configured, the API answers only loopback requests
    (127.0.0.1 / ::1). This is the safe default: an unauthenticated
    secrets-editing panel is never exposed on a non-loopback bind by accident.

Secrets (api_key values) are never returned verbatim by GET endpoints — they are
masked. ``${VAR}`` references are not secret and are returned as-is so the UI can
display and round-trip them.
"""

import contextlib
import hmac
import ipaddress
import os
import threading

from flask import Blueprint, jsonify, request, send_from_directory

from . import providers as _providers
from .config import (
    RESERVED_PROVIDER_NAMES,
    get_config_path,
    get_provider,
    heal_config,
    load_config,
    provider_api_key,
    provider_base_url,
    resolve_env_refs,
    save_config,
    value_has_env_ref,
)

try:
    import fcntl  # POSIX advisory file locking
except ImportError:  # pragma: no cover - non-POSIX (e.g. Windows)
    fcntl = None

# Headers a reverse proxy adds when forwarding a request. Their presence means
# request.remote_addr is the proxy, not the real client, so a 127.0.0.1
# remote_addr can no longer be trusted as "local" for the tokenless gate.
_FORWARDING_HEADERS = ("X-Forwarded-For", "X-Real-IP", "Forwarded", "X-Forwarded-Host")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "admin")

bp = Blueprint(
    "admin",
    __name__,
    static_folder=_STATIC_DIR,
    static_url_path="/admin/static",
)

# Serializes the read-modify-write cycle of config edits. The threading lock
# covers concurrent requests within one gunicorn worker; the fcntl advisory lock
# (see _locked) covers concurrent requests across workers, so two workers cannot
# each load an old snapshot, mutate different subtrees, and clobber each other on
# save (lost updates).
_write_lock = threading.Lock()


@contextlib.contextmanager
def _locked():
    """Acquire the in-process lock and a cross-process advisory file lock around
    a config read-modify-write. Falls back to the thread lock alone where fcntl
    is unavailable (non-POSIX)."""
    _write_lock.acquire()
    try:
        if fcntl is None:
            yield
            return
        lock_path = str(get_config_path()) + ".lock"
        try:
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            handle = open(lock_path, "w")
        except OSError:
            # If the lock file can't be created, degrade to the thread lock only
            # rather than blocking all admin writes.
            yield
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
    finally:
        _write_lock.release()

# Provider fields the admin API accepts/persists. Anything else is ignored on
# write so the UI cannot inject arbitrary keys.
_PROVIDER_FIELDS = (
    "base_url",
    "api_key",
    "model_filter",
    "models_url",
    "models_id_field",
    "models_keep_task",
    "expose_to_virtual_models",
)

_SERVER_INT_FIELDS = (
    "port",
    "request_timeout",
    "stream_timeout",
    "response_cache_ttl",
    "models_cache_ttl",
)
_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})
_VALID_CAPABILITIES = frozenset({"tools", "vision", "reasoning", "json"})


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load() -> dict:
    """Read a fresh copy of config from disk (bypassing the mtime cache)."""
    return load_config(force_reload=True)


def _save(config: dict) -> bool:
    return save_config(config)


def _admin_block(config: dict) -> dict:
    block = config.get("admin")
    return block if isinstance(block, dict) else {}


def _admin_enabled(config: dict) -> bool:
    """Whether the admin UI/API is enabled.

    An ``LLMPROXY_ADMIN_ENABLED`` environment variable (set by the ``--admin`` /
    ``--no-admin`` CLI flags) takes precedence over the config value, mirroring
    how ``LLMPROXY_CONFIG`` propagates the ``--config`` override. This is what
    makes the CLI toggle reach the blueprint, which reads config from disk on
    every request rather than from the in-memory startup config.
    """
    env = os.environ.get("LLMPROXY_ADMIN_ENABLED")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off", "")
    return _admin_block(config).get("enabled", True) is not False


def _admin_token(config: dict) -> str:
    """Resolve the configured admin token, or '' if none.

    Order: LLMPROXY_ADMIN_TOKEN env var (highest), then config['admin']['token']
    (which may itself be a ${VAR} reference).
    """
    env_token = os.environ.get("LLMPROXY_ADMIN_TOKEN", "")
    if env_token:
        return env_token
    return resolve_env_refs(_admin_block(config).get("token")) or ""


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------

def _is_loopback(remote_addr: str | None) -> bool:
    if not remote_addr:
        return False
    try:
        return ipaddress.ip_address(remote_addr).is_loopback
    except ValueError:
        return False


def _presented_token() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer "):].strip()
    return request.headers.get("X-Admin-Token", "").strip()


def enforce_admin_auth():
    """Apply the admin auth policy to the current request.

    Returns ``None`` when the request is authorized, otherwise a
    ``(json_response, status)`` tuple. Shared by the ``/admin/api/*`` guard and
    other administrative mutations (e.g. POST /v1/usage/reset) so they enforce
    the same token / loopback policy.
    """
    config = load_config()
    if not _admin_enabled(config):
        return jsonify({"error": "Admin API is disabled."}), 404

    token = _admin_token(config)
    if token:
        presented = _presented_token()
        if presented and hmac.compare_digest(presented, token):
            return None
        return jsonify({"error": "Missing or invalid admin token."}), 401

    # No token configured: loopback-only. A request that arrived through a
    # reverse proxy carries forwarding headers, in which case remote_addr is the
    # proxy (often 127.0.0.1) and cannot be trusted as "local" — require a token
    # instead of silently exposing the API to forwarded external clients.
    forwarded = any(request.headers.get(h) for h in _FORWARDING_HEADERS)
    if not forwarded and _is_loopback(request.remote_addr):
        return None
    detail = (
        " This request arrived via a reverse proxy (forwarding headers present);"
        " set an admin token to allow proxied/remote access."
        if forwarded else ""
    )
    return (
        jsonify({
            "error": (
                "Admin API is restricted to localhost. Set an admin token "
                "(LLMPROXY_ADMIN_TOKEN env var or config['admin']['token']) to "
                "allow remote access." + detail
            )
        }),
        403,
    )


@bp.before_request
def _require_auth():
    """Gate only the data API (/admin/api/*). The static shell is public."""
    path = request.path or ""
    if not path.startswith("/admin/api/"):
        return None  # UI shell / static assets carry no secrets.
    return enforce_admin_auth()


# ---------------------------------------------------------------------------
# Secret masking
# ---------------------------------------------------------------------------

def _mask_secret(value) -> str:
    """Mask a literal secret for display. ${VAR} references pass through
    verbatim (not secret); short/empty values are fully masked."""
    if not value:
        return ""
    if value_has_env_ref(value):
        return value
    s = str(value)
    if len(s) <= 8:
        return "•" * len(s)
    return f"{s[:3]}…{s[-4:]}"


def _provider_view(cfg: dict) -> dict:
    """Return a provider config copy safe for GET responses: api_key masked,
    with flags telling the UI whether a key exists and whether it's an env ref."""
    view = {k: cfg.get(k) for k in _PROVIDER_FIELDS if k in cfg}
    raw_key = cfg.get("api_key")
    view["api_key"] = _mask_secret(raw_key)
    view["api_key_set"] = bool(raw_key)
    view["api_key_is_env"] = value_has_env_ref(raw_key)
    view["base_url_is_env"] = value_has_env_ref(cfg.get("base_url"))
    return view


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _err(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _clean_provider_payload(payload: dict, existing: dict | None) -> tuple[dict | None, str | None]:
    """Validate and normalize a provider write payload.

    Returns (provider_cfg, error). On the api_key field: when *existing* is
    given (an edit) and the payload omits api_key or sends a blank string, the
    existing raw key is preserved (the UI submits blank to mean "unchanged").
    A non-blank string overwrites; a ``${VAR}`` reference is stored as-is.
    """
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    cfg: dict = dict(existing) if existing else {}

    if "base_url" in payload:
        base_url = payload["base_url"]
        if not isinstance(base_url, str) or not base_url.strip():
            return None, "base_url is required and must be a non-empty string."
        cfg["base_url"] = base_url.strip()
    elif "base_url" not in cfg:
        return None, "base_url is required."

    if "api_key" in payload:
        api_key = payload["api_key"]
        if api_key is None:
            api_key = ""
        if not isinstance(api_key, str):
            return None, "api_key must be a string."
        # Blank on edit => keep existing; blank on create => no key.
        if api_key.strip() == "" and existing is not None:
            pass  # preserve existing cfg['api_key']
        else:
            cfg["api_key"] = api_key.strip()

    if "model_filter" in payload:
        mf = payload["model_filter"]
        if mf is not None and not (isinstance(mf, list) and all(isinstance(x, str) for x in mf)):
            return None, "model_filter must be null or a list of strings."
        cfg["model_filter"] = mf

    for field in ("models_url", "models_id_field", "models_keep_task"):
        if field in payload:
            val = payload[field]
            if val in (None, ""):
                cfg.pop(field, None)
            elif isinstance(val, str):
                cfg[field] = val
            else:
                return None, f"{field} must be a string or null."

    if "expose_to_virtual_models" in payload:
        val = payload["expose_to_virtual_models"]
        if not isinstance(val, bool):
            return None, "expose_to_virtual_models must be a boolean."
        cfg["expose_to_virtual_models"] = val

    return cfg, None


# ---------------------------------------------------------------------------
# UI shell
# ---------------------------------------------------------------------------

@bp.route("/admin")
@bp.route("/admin/")
def admin_index():
    config = load_config()
    if not _admin_enabled(config):
        return jsonify({"error": "Admin UI is disabled."}), 404
    return send_from_directory(_STATIC_DIR, "index.html")


# ---------------------------------------------------------------------------
# Config (read) + server settings
# ---------------------------------------------------------------------------

@bp.route("/admin/api/config", methods=["GET"])
def api_get_config():
    config = _load()
    providers = {
        name: _provider_view(cfg)
        for name, cfg in config.get("providers", {}).items()
        if isinstance(cfg, dict)
    }
    admin = _admin_block(config)
    return jsonify({
        "providers": providers,
        "believed_free": config.get("believed_free", []),
        "model_reasoning": config.get("model_reasoning", {}),
        "model_capabilities": config.get("model_capabilities", {}),
        "free_limits": config.get("free_limits", {}),
        "server": config.get("server", {}),
        "admin": {
            "enabled": admin.get("enabled", True) is not False,
            "token_set": bool(_admin_token(config)),
        },
        "reserved_provider_names": sorted(RESERVED_PROVIDER_NAMES),
        "valid_reasoning_levels": sorted(_providers.VALID_REASONING_LEVELS),
        "valid_capabilities": sorted(_VALID_CAPABILITIES),
        "free_limit_keys": list(_providers.FREE_LIMIT_KEYS),
    })


@bp.route("/admin/api/server", methods=["PUT"])
def api_put_server():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _err("Request body must be a JSON object.")

    with _locked():
        config = _load()
        server = dict(config.get("server", {}))

        if "host" in payload:
            host = payload["host"]
            if not isinstance(host, str) or not host.strip():
                return _err("host must be a non-empty string.")
            server["host"] = host.strip()

        if "log_level" in payload:
            lvl = str(payload["log_level"]).upper()
            if lvl not in _VALID_LOG_LEVELS:
                return _err(f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}.")
            server["log_level"] = lvl

        for field in _SERVER_INT_FIELDS:
            if field not in payload or payload[field] is None:
                continue
            try:
                val = int(payload[field])
            except (TypeError, ValueError):
                return _err(f"{field} must be an integer.")
            if field == "port" and not (1 <= val <= 65535):
                return _err("port must be between 1 and 65535.")
            if field != "port" and val < 0:
                return _err(f"{field} must be >= 0.")
            server[field] = val

        config["server"] = server
        if not _save(config):
            return _err("Failed to persist configuration.", 500)
    return jsonify({"server": server})


# ---------------------------------------------------------------------------
# Providers CRUD
# ---------------------------------------------------------------------------

@bp.route("/admin/api/providers", methods=["GET"])
def api_list_providers():
    config = _load()
    return jsonify({
        name: _provider_view(cfg)
        for name, cfg in config.get("providers", {}).items()
        if isinstance(cfg, dict)
    })


@bp.route("/admin/api/providers/<name>", methods=["GET"])
def api_get_provider(name: str):
    config = _load()
    cfg = get_provider(config, name)
    if not cfg:
        return _err(f"Unknown provider '{name}'.", 404)
    return jsonify(_provider_view(cfg))


@bp.route("/admin/api/providers", methods=["POST"])
def api_create_provider():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _err("Request body must be a JSON object.")
    name = (payload.get("name") or "").strip()
    if not name:
        return _err("Provider name is required.")
    if name in RESERVED_PROVIDER_NAMES:
        return _err(f"'{name}' is a reserved provider name.", 409)

    with _locked():
        config = _load()
        if get_provider(config, name) is not None:
            return _err(f"Provider '{name}' already exists.", 409)
        cfg, error = _clean_provider_payload(payload, existing=None)
        if error:
            return _err(error)
        config.setdefault("providers", {})[name] = cfg
        if not _save(config):
            return _err("Failed to persist configuration.", 500)
    return jsonify({"name": name, "provider": _provider_view(cfg)}), 201


@bp.route("/admin/api/providers/<name>", methods=["PUT"])
def api_update_provider(name: str):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _err("Request body must be a JSON object.")
    if name in RESERVED_PROVIDER_NAMES:
        return _err(f"'{name}' is a reserved provider name.", 409)

    with _locked():
        config = _load()
        existing = get_provider(config, name)
        if existing is None:
            return _err(f"Unknown provider '{name}'.", 404)
        cfg, error = _clean_provider_payload(payload, existing=existing)
        if error:
            return _err(error)
        config["providers"][name] = cfg
        if not _save(config):
            return _err("Failed to persist configuration.", 500)
    return jsonify({"name": name, "provider": _provider_view(cfg)})


@bp.route("/admin/api/providers/<name>", methods=["DELETE"])
def api_delete_provider(name: str):
    with _locked():
        config = _load()
        if get_provider(config, name) is None:
            return _err(f"Unknown provider '{name}'.", 404)
        del config["providers"][name]
        if not _save(config):
            return _err("Failed to persist configuration.", 500)
    return jsonify({"deleted": name})


# ---------------------------------------------------------------------------
# Provider templates + add-from-template
# ---------------------------------------------------------------------------

@bp.route("/admin/api/provider-templates", methods=["GET"])
def api_provider_templates():
    return jsonify({"templates": _providers.get_provider_templates()})


def _substitute_placeholders(value: str, subs: dict) -> str:
    for key, val in subs.items():
        value = value.replace("{" + key + "}", val)
    return value


@bp.route("/admin/api/providers/from-template", methods=["POST"])
def api_provider_from_template():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _err("Request body must be a JSON object.")
    template_key = payload.get("template_key")
    templates = {t["key"]: t for t in _providers.get_provider_templates()}
    template = templates.get(template_key)
    if template is None:
        return _err(f"Unknown template '{template_key}'.", 404)

    name = (payload.get("name") or template_key).strip()
    if name in RESERVED_PROVIDER_NAMES:
        return _err(f"'{name}' is a reserved provider name.", 409)

    subs: dict = {}
    if template.get("account_id_required"):
        acct = (payload.get("account_id") or "").strip()
        if not acct:
            return _err("This template requires an account_id.")
        subs["account_id"] = acct
    if template.get("gateway_id_required"):
        gw = (payload.get("gateway_id") or "").strip()
        if not gw:
            return _err("This template requires a gateway_id.")
        subs["gateway_id"] = gw

    cfg: dict = {
        "base_url": _substitute_placeholders(template.get("base_url", ""), subs),
        "model_filter": None,
    }
    api_key = (payload.get("api_key") or "").strip()
    if api_key:
        cfg["api_key"] = api_key
    elif template.get("key_required"):
        cfg["api_key"] = ""
    for field in ("models_url", "models_id_field", "models_keep_task"):
        if template.get(field):
            cfg[field] = _substitute_placeholders(template[field], subs)

    with _locked():
        config = _load()
        if get_provider(config, name) is not None:
            return _err(f"Provider '{name}' already exists.", 409)
        config.setdefault("providers", {})[name] = cfg
        if not _save(config):
            return _err("Failed to persist configuration.", 500)
    return jsonify({"name": name, "provider": _provider_view(cfg)}), 201


# ---------------------------------------------------------------------------
# Model discovery (live)
# ---------------------------------------------------------------------------

def _discover(provider_name: str, cfg: dict, timeout: int) -> list[str]:
    """Discover a provider's model display IDs via the server's existing
    /models fetch (handles all the upstream response shapes + filters)."""
    from . import server  # lazy import avoids any import-cycle fragility
    models = server._fetch_provider_models(provider_name, cfg, timeout)
    return [m["id"] for m in models]


@bp.route("/admin/api/providers/<name>/models", methods=["GET"])
def api_provider_models(name: str):
    config = _load()
    cfg = get_provider(config, name)
    if not cfg:
        return _err(f"Unknown provider '{name}'.", 404)
    timeout = int(config.get("server", {}).get("request_timeout", 30))
    ids = _discover(name, cfg, min(timeout, 30))
    body: dict = {"provider": name, "models": ids}
    if not ids:
        body["_warning"] = "No models discovered (provider unreachable or empty)."
    return jsonify(body)


@bp.route("/admin/api/models", methods=["GET"])
def api_all_models():
    config = _load()
    timeout = min(int(config.get("server", {}).get("request_timeout", 30)), 30)
    out: list[str] = []
    for name, cfg in config.get("providers", {}).items():
        if name in RESERVED_PROVIDER_NAMES or not isinstance(cfg, dict):
            continue
        out.extend(_discover(name, cfg, timeout))
    return jsonify({"models": sorted(set(out))})


@bp.route("/admin/api/providers/<name>/test", methods=["POST"])
def api_test_provider(name: str):
    config = _load()
    cfg = get_provider(config, name)
    if not cfg:
        return _err(f"Unknown provider '{name}'.", 404)
    ids = _discover(name, cfg, 15)
    return jsonify({
        "ok": bool(ids),
        "model_count": len(ids),
        "base_url": provider_base_url(cfg),
        "api_key_resolved": bool(provider_api_key(cfg)),
    })


# ---------------------------------------------------------------------------
# Categorizations (believed_free / reasoning / capabilities / free_limits)
# ---------------------------------------------------------------------------

def _put_section(key: str, validate):
    payload = request.get_json(silent=True)
    error = validate(payload)
    if error:
        return _err(error)
    with _locked():
        config = _load()
        config[key] = payload
        if not _save(config):
            return _err("Failed to persist configuration.", 500)
    return jsonify({key: payload})


@bp.route("/admin/api/believed-free", methods=["GET", "PUT"])
def api_believed_free():
    if request.method == "GET":
        return jsonify({"believed_free": _load().get("believed_free", [])})

    def validate(p):
        if not (isinstance(p, list) and all(isinstance(x, str) for x in p)):
            return "believed_free must be a list of strings."
        return None
    return _put_section("believed_free", validate)


@bp.route("/admin/api/model-reasoning", methods=["GET", "PUT"])
def api_model_reasoning():
    if request.method == "GET":
        return jsonify({"model_reasoning": _load().get("model_reasoning", {})})

    def validate(p):
        if not isinstance(p, dict):
            return "model_reasoning must be an object of model -> level."
        for model, level in p.items():
            if level not in _providers.VALID_REASONING_LEVELS:
                return (
                    f"Invalid reasoning level '{level}' for '{model}'. "
                    f"Valid: {sorted(_providers.VALID_REASONING_LEVELS)}."
                )
        return None
    return _put_section("model_reasoning", validate)


@bp.route("/admin/api/model-capabilities", methods=["GET", "PUT"])
def api_model_capabilities():
    if request.method == "GET":
        return jsonify({"model_capabilities": _load().get("model_capabilities", {})})

    def validate(p):
        if not isinstance(p, dict):
            return "model_capabilities must be an object of model -> [capabilities]."
        for model, caps in p.items():
            if not (isinstance(caps, list) and all(isinstance(c, str) for c in caps)):
                return f"Capabilities for '{model}' must be a list of strings."
            bad = set(caps) - _VALID_CAPABILITIES
            if bad:
                return (
                    f"Invalid capabilities {sorted(bad)} for '{model}'. "
                    f"Valid: {sorted(_VALID_CAPABILITIES)}."
                )
        return None
    return _put_section("model_capabilities", validate)


@bp.route("/admin/api/free-limits", methods=["GET", "PUT"])
def api_free_limits():
    if request.method == "GET":
        return jsonify({"free_limits": _load().get("free_limits", {})})

    def validate(p):
        if not isinstance(p, dict):
            return "free_limits must be an object of model -> limits."
        for model, limits in p.items():
            if model == "_note":
                continue
            if not isinstance(limits, dict):
                return f"Limits for '{model}' must be an object."
            for k, v in limits.items():
                if k not in _providers.FREE_LIMIT_KEYS:
                    return (
                        f"Invalid limit key '{k}' for '{model}'. "
                        f"Valid: {list(_providers.FREE_LIMIT_KEYS)}."
                    )
                if v is not None and not isinstance(v, int):
                    return f"Limit '{k}' for '{model}' must be an integer or null."
        return None
    return _put_section("free_limits", validate)


# ---------------------------------------------------------------------------
# Virtual-endpoint preview (derived from categorizations)
# ---------------------------------------------------------------------------

@bp.route("/admin/api/virtual-models", methods=["GET"])
def api_virtual_models():
    """Preview the virtual endpoints the current categorizations produce.

    Derived statically from config (no live discovery), so it reflects exactly
    what the user has tagged. Local-provider models are summarized by provider
    since enumerating them requires discovery.
    """
    from . import server  # for the canonical level/capability names
    config = _load()
    believed_free = [s.lower() for s in config.get("believed_free", [])]
    reasoning = config.get("model_reasoning", {})
    capabilities = config.get("model_capabilities", {})
    providers = {
        n: c for n, c in config.get("providers", {}).items()
        if isinstance(c, dict) and n not in RESERVED_PROVIDER_NAMES
    }

    virtuals: list[dict] = []

    free_models = sorted({
        m for m in set(believed_free) | set(reasoning) | _capability_models(capabilities)
        if "free" in m.lower() or m.lower() in believed_free
    })
    if free_models or believed_free:
        virtuals.append({
            "id": "llmproxy__free",
            "description": "Free-tier models (ID contains 'free' or listed in believed_free).",
            "backing": free_models or sorted(believed_free),
        })

    local_providers = [n for n, c in providers.items() if server._is_local_url(provider_base_url(c))]
    if local_providers:
        virtuals.append({
            "id": "llmproxy__local",
            "description": "All models served by localhost providers.",
            "backing": [f"{n}/* (all models)" for n in sorted(local_providers)],
        })

    for level in server._REASONING_LEVELS:
        backing = sorted(m for m, lvl in reasoning.items() if lvl == level)
        if backing:
            virtuals.append({
                "id": f"llmproxy__{level}",
                "description": f"Models tagged '{level}' reasoning.",
                "backing": backing,
            })

    for cap in server._CAPABILITY_VIRTUALS:
        backing = sorted(m for m, caps in capabilities.items() if cap in (caps or []))
        if backing:
            virtuals.append({
                "id": f"llmproxy__{cap}",
                "description": f"Models tagged '{cap}' in model_capabilities.",
                "backing": backing,
            })

    return jsonify({"virtual_models": virtuals})


def _capability_models(capabilities: dict) -> set:
    out: set = set()
    for model in capabilities:
        out.add(model)
    return out


# ---------------------------------------------------------------------------
# Heal / validate
# ---------------------------------------------------------------------------

@bp.route("/admin/api/heal", methods=["POST"])
def api_heal():
    with _locked():
        config = _load()
        healed, changed, messages = heal_config(config)
        if changed:
            if not _save(healed):
                return _err("Failed to persist healed configuration.", 500)
    return jsonify({
        "changed": changed,
        "messages": [{"level": lvl, "text": txt} for lvl, txt in messages],
    })


@bp.route("/admin/api/validate", methods=["POST", "GET"])
def api_validate():
    config = _load()
    problems: list[str] = []
    for name, cfg in config.get("providers", {}).items():
        if name in RESERVED_PROVIDER_NAMES:
            problems.append(f"Provider '{name}' uses a reserved name.")
        if not isinstance(cfg, dict) or not cfg.get("base_url"):
            problems.append(f"Provider '{name}' is missing base_url.")
    return jsonify({"ok": not problems, "problems": problems})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_admin(app) -> None:
    """Register the admin blueprint onto *app* (idempotent)."""
    if "admin" in app.blueprints:
        return
    app.register_blueprint(bp)
