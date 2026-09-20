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
import traceback

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
    "protocol",
)

# Upstream dialects llmproxy can translate to (see llmproxy/dialects/).
_PROVIDER_PROTOCOLS = ("openai", "anthropic", "gemini")

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


def _accounts_view(cfg: dict) -> list | None:
    """Masked view of a provider's multiple-account credentials, or None.

    Never returns raw key material: each account exposes only a masked key plus
    key_set / key_is_env flags, alongside its (non-secret) label and priority.
    Recognizes both the ``accounts`` and ``api_keys`` shapes.
    """
    raw = cfg.get("accounts")
    entries: list[tuple] = []
    if isinstance(raw, list) and raw:
        for item in raw:
            if isinstance(item, dict):
                entries.append((item.get("key"), item.get("label"), item.get("priority")))
            elif isinstance(item, str):
                entries.append((item, None, None))
    elif isinstance(cfg.get("api_keys"), list) and cfg["api_keys"]:
        entries = [(k, None, None) for k in cfg["api_keys"]]
    else:
        return None

    out: list[dict] = []
    for key, label, priority in entries:
        item = {
            "key": _mask_secret(key),
            "key_set": bool(key),
            "key_is_env": value_has_env_ref(key),
        }
        if label is not None:
            item["label"] = label
        if priority is not None:
            item["priority"] = priority
        out.append(item)
    return out


def _provider_view(cfg: dict) -> dict:
    """Return a provider config copy safe for GET responses: api_key masked,
    with flags telling the UI whether a key exists and whether it's an env ref."""
    view = {k: cfg.get(k) for k in _PROVIDER_FIELDS if k in cfg}
    raw_key = cfg.get("api_key")
    view["api_key"] = _mask_secret(raw_key)
    view["api_key_set"] = bool(raw_key)
    view["api_key_is_env"] = value_has_env_ref(raw_key)
    view["base_url_is_env"] = value_has_env_ref(cfg.get("base_url"))
    if "account_strategy" in cfg:
        view["account_strategy"] = cfg.get("account_strategy")
    accounts = _accounts_view(cfg)
    if accounts is not None:
        view["accounts"] = accounts  # keys masked; raw material never leaves here
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

    if "protocol" in payload:
        val = payload["protocol"]
        if val in (None, "", "openai"):
            cfg.pop("protocol", None)  # openai is the default; keep configs clean
        elif val in _PROVIDER_PROTOCOLS:
            cfg["protocol"] = val
        else:
            return None, f"protocol must be one of {_PROVIDER_PROTOCOLS}."

    if "account_strategy" in payload:
        val = payload["account_strategy"]
        if val in (None, "", "round_robin"):
            cfg.pop("account_strategy", None)  # round_robin is the default
        elif val == "priority":
            cfg["account_strategy"] = "priority"
        else:
            return None, "account_strategy must be 'round_robin' or 'priority'."

    if "api_keys" in payload:
        val = payload["api_keys"]
        if val in (None, []):
            cfg.pop("api_keys", None)
        elif isinstance(val, list) and all(isinstance(x, str) and x.strip() for x in val):
            cfg["api_keys"] = [x.strip() for x in val]
        else:
            return None, "api_keys must be a list of non-empty strings."

    if "accounts" in payload:
        cleaned, err = _clean_accounts(payload["accounts"], existing)
        if err is not None:
            return None, err
        if cleaned:
            cfg["accounts"] = cleaned
        else:
            cfg.pop("accounts", None)

    return cfg, None


def _clean_accounts(val, existing: dict | None) -> tuple[list | None, str | None]:
    """Validate a provider's ``accounts`` list.

    Each entry is ``{"key", "label"?, "priority"?}``. A blank/omitted key on an
    edit preserves the existing account's key matched by label (the same
    "blank means unchanged" convention as the single api_key field), so the
    masked GET view can be re-submitted without re-entering secrets.
    """
    if val in (None, []):
        return None, None
    if not isinstance(val, list):
        return None, "accounts must be a list of objects."

    existing_by_label: dict = {}
    for a in (existing or {}).get("accounts", []) or []:
        if isinstance(a, dict) and a.get("label"):
            existing_by_label[a["label"]] = a.get("key")

    cleaned: list[dict] = []
    for item in val:
        if not isinstance(item, dict):
            return None, "each account must be an object."
        key = item.get("key")
        label = item.get("label")
        if key is None or (isinstance(key, str) and key.strip() == ""):
            key = existing_by_label.get(label)  # blank => keep existing by label
            if not key:
                return None, "each account requires a key."
        elif not isinstance(key, str):
            return None, "account key must be a string."
        else:
            key = key.strip()
        entry: dict = {"key": key}
        if label not in (None, ""):
            entry["label"] = label
        if item.get("priority") is not None:
            try:
                entry["priority"] = int(item["priority"])
            except (TypeError, ValueError):
                return None, "account priority must be an integer."
        cleaned.append(entry)
    return cleaned, None


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

_MAINTENANCE_BOOL_FLAGS = (
    "probe_cost",
    "autoremove_believed_free",
    "update_believed_free_on_startup",
    "pr_providers_list",
)
# Maintenance booleans that default to True when absent (vs False above).
_MAINTENANCE_BOOL_FLAGS_DEFAULT_TRUE = (
    "sync_believed_free_on_startup",
    # The two learning cadences and the inference switches. None of these had an
    # admin surface, so a deployment could only change them by hand-editing the
    # config the UI is meant to replace.
    "routing_metadata_enabled",
    "flagship_enabled",
    "infer_reasoning",
    "infer_family_capabilities",
)
_MAINTENANCE_STR_FIELDS = ("pr_providers_repo", "pr_providers_base", "pr_providers_branch")
# Integer fields, with the default applied when the key is absent from config.
_MAINTENANCE_INT_FIELDS: dict[str, int] = {
    "probe_frequency_days": 0,
    "update_frequency_days": 7,
    "probe_timeout_sec": 10,
    "routing_metadata_frequency_days": 7,
    "flagship_frequency_days": 7,
    "min_family_members": 3,
    # Enforced by _maybe_fire_pr_if_due all along, with no way to set it.
    # 0, matching what the server reads for a missing key
    # (pr_cfg.get("frequency_days", 0) = no throttle). Showing 7 here would mean
    # saving the form once silently turned "PR on every update" into "weekly".
    "pr_providers_frequency_days": 0,
}

# The admin API and frontend keep the historical flat field names; storage maps
# each into the reorganized nested config (free_tier / providers_pr). This keeps
# the single-page admin UI unchanged while the on-disk config uses the grouped
# objects (and the config loader's migration shim accepts either form on input).
_MAINTENANCE_PATHS: dict[str, tuple[str, ...]] = {
    "probe_cost": ("free_tier", "cost_probe", "enabled"),
    "autoremove_believed_free": ("free_tier", "cost_probe", "autoremove"),
    "update_believed_free_on_startup": ("free_tier", "update_on_startup"),
    "pr_providers_list": ("providers_pr", "enabled"),
    "sync_believed_free_on_startup": ("free_tier", "sync_on_startup"),
    "probe_frequency_days": ("free_tier", "cost_probe", "frequency_days"),
    # The sweep's own cadence. Distinct from probe_frequency_days, which only
    # throttles the cost probe *within* a sweep.
    "update_frequency_days": ("free_tier", "update_frequency_days"),
    # Read timeout shared by both probes.
    "probe_timeout_sec": ("free_tier", "probe_timeout_sec"),
    "pr_providers_repo": ("providers_pr", "repo"),
    "pr_providers_base": ("providers_pr", "base"),
    "pr_providers_branch": ("providers_pr", "branch"),
    "pr_providers_token": ("providers_pr", "token"),
    "pr_providers_frequency_days": ("providers_pr", "frequency_days"),
    "routing_metadata_enabled": ("routing_metadata", "enabled"),
    "routing_metadata_frequency_days": ("routing_metadata", "refresh_frequency_days"),
    "infer_reasoning": ("routing_metadata", "infer_reasoning"),
    "infer_family_capabilities": ("routing_metadata", "infer_family_capabilities"),
    "min_family_members": ("routing_metadata", "min_family_members"),
    "flagship_enabled": ("flagship_tier", "enabled"),
    "flagship_frequency_days": ("flagship_tier", "refresh_frequency_days"),
}


def _cfg_get(config: dict, path: tuple[str, ...], default=None):
    """Read a value at a nested *path* in *config*, returning *default* if absent."""
    cur = config
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _cfg_set(config: dict, path: tuple[str, ...], value) -> None:
    """Set *value* at a nested *path* in *config*, creating intermediate dicts."""
    cur = config
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _maintenance_view(config: dict) -> dict:
    """The automation/maintenance flags, with the PR token masked like api_key.

    Field names are the historical flat names; values are read from the nested
    free_tier / providers_pr objects via _MAINTENANCE_PATHS.
    """
    view: dict = {
        k: bool(_cfg_get(config, _MAINTENANCE_PATHS[k], False)) for k in _MAINTENANCE_BOOL_FLAGS
    }
    view.update(
        {k: bool(_cfg_get(config, _MAINTENANCE_PATHS[k], True)) for k in _MAINTENANCE_BOOL_FLAGS_DEFAULT_TRUE}
    )
    for key, default in _MAINTENANCE_INT_FIELDS.items():
        raw = _cfg_get(config, _MAINTENANCE_PATHS[key], default)
        try:
            view[key] = int(default if raw is None else raw)
        except (TypeError, ValueError):
            view[key] = default
    for k in _MAINTENANCE_STR_FIELDS:
        view[k] = _cfg_get(config, _MAINTENANCE_PATHS[k]) or ""
    tok = _cfg_get(config, _MAINTENANCE_PATHS["pr_providers_token"])
    view["pr_providers_token"] = _mask_secret(tok)
    view["pr_providers_token_set"] = bool(tok)
    view["pr_providers_token_is_env"] = value_has_env_ref(tok)
    return view


@bp.route("/admin/api/config", methods=["GET"])
def api_get_config():
    config = _load()
    providers = {
        name: _provider_view(cfg)
        for name, cfg in config.get("providers", {}).items()
        if isinstance(cfg, dict)
    }
    admin = _admin_block(config)
    # The EFFECTIVE values, not config.json's. Reading config alone showed every
    # model as un-free, untagged and incapable once the five keys migrated out,
    # and saving that form wrote the blanks back as real overrides.
    eff = _effective_routing(config)
    return jsonify({
        "providers": providers,
        "believed_free": eff.get("believed_free") or [],
        "favorite_free_models": config.get("favorite_free_models", []),
        "model_reasoning": eff.get("model_reasoning") or {},
        "model_capabilities": eff.get("model_capabilities") or {},
        "free_limits": eff.get("free_limits") or {},
        "server": config.get("server", {}),
        "admin": {
            "enabled": admin.get("enabled", True) is not False,
            "token_set": bool(_admin_token(config)),
        },
        "maintenance": _maintenance_view(config),
        "reserved_provider_names": sorted(RESERVED_PROVIDER_NAMES),
        # Hand-assignable levels, in canonical weakest-to-strongest order (not
        # alphabetical — the order is the routing rank).
        "valid_reasoning_levels": [
            lvl for lvl in _providers.REASONING_LEVELS
            if lvl in _providers.VALID_REASONING_LEVELS
        ],
        # Every tier including computed overlays, for display purposes.
        "reasoning_levels": list(_providers.REASONING_LEVELS),
        "overlay_reasoning_levels": sorted(_providers.OVERLAY_REASONING_LEVELS),
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


@bp.route("/admin/api/maintenance", methods=["PUT"])
def api_put_maintenance():
    """Edit the top-level automation flags: the free-models updater / cost probe
    (probe_cost, autoremove_believed_free, update_believed_free_on_startup,
    probe_frequency_days, update_frequency_days, probe_timeout_sec) and the
    providers-PR settings (pr_providers_*).

    The PR token is write-only: send a new value to set it, or omit/blank to keep
    the current one (mirrors the api_key edit convention)."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _err("Request body must be a JSON object.")

    with _locked():
        config = _load()

        for key in _MAINTENANCE_BOOL_FLAGS + _MAINTENANCE_BOOL_FLAGS_DEFAULT_TRUE:
            if key in payload:
                if not isinstance(payload[key], bool):
                    return _err(f"{key} must be a boolean.")
                _cfg_set(config, _MAINTENANCE_PATHS[key], payload[key])

        for key in _MAINTENANCE_INT_FIELDS:
            if key not in payload:
                continue
            try:
                v = int(payload[key])
            except (TypeError, ValueError):
                return _err(f"{key} must be an integer.")
            if v < 0:
                return _err(f"{key} must be >= 0.")
            # A zero timeout would mean "give up immediately", never "no limit".
            if key == "probe_timeout_sec" and v == 0:
                return _err("probe_timeout_sec must be >= 1.")
            _cfg_set(config, _MAINTENANCE_PATHS[key], v)

        for key in _MAINTENANCE_STR_FIELDS:
            if key in payload:
                val = payload[key]
                if val is None:
                    val = ""
                if not isinstance(val, str):
                    return _err(f"{key} must be a string.")
                _cfg_set(config, _MAINTENANCE_PATHS[key], val.strip())

        if "pr_providers_token" in payload:
            tok = payload["pr_providers_token"]
            if tok is None:
                tok = ""
            if not isinstance(tok, str):
                return _err("pr_providers_token must be a string.")
            if tok.strip():
                _cfg_set(config, _MAINTENANCE_PATHS["pr_providers_token"], tok.strip())
            # blank -> keep the existing token

        if not _save(config):
            return _err("Failed to persist configuration.", 500)
    return jsonify({"maintenance": _maintenance_view(config)})


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

    # Same as the wizard: a template's curated model list is the only thing
    # standing between a catalog-less provider and an inert config entry, so
    # seed model_filter from it rather than always writing null.
    template_filter = template.get("example_model_filter")
    cfg: dict = {
        "base_url": _substitute_placeholders(template.get("base_url", ""), subs),
        "model_filter": list(template_filter) if template_filter else None,
    }
    api_key = (payload.get("api_key") or "").strip()
    if api_key:
        cfg["api_key"] = api_key
    elif template.get("key_required"):
        cfg["api_key"] = ""
    for field in ("models_url", "models_id_field", "models_keep_task"):
        if template.get(field):
            cfg[field] = _substitute_placeholders(template[field], subs)
    # Native (non-OpenAI) upstreams carry a protocol so the proxy translates for
    # them — without this, an Anthropic/Gemini template would be saved as openai.
    if template.get("protocol") and template["protocol"] != "openai":
        cfg["protocol"] = template["protocol"]
    # Same reason the wizard copies it: the loadbalanced virtual reads
    # free_allowance off the provider block, not off the template sidecar.
    if template.get("free_allowance"):
        cfg["free_allowance"] = dict(template["free_allowance"])

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
    if key in _ROUTING_SECTIONS:
        return _put_routing_section(key, payload)
    with _locked():
        config = _load()
        config[key] = payload
        if not _save(config):
            return _err("Failed to persist configuration.", 500)
    return jsonify({key: payload})


# The sections that live in the sidecar's curated layer rather than config.json.
_ROUTING_SECTIONS = frozenset({
    "believed_free", "cost_observed_free_tier",
    "model_reasoning", "model_capabilities", "free_limits",
})


def _put_routing_section(key: str, payload):
    """Save a whole section, recording only what actually differs as curated.

    The UI loads every model, lets you toggle a few, and saves the lot. If that
    wrote the whole payload into the curated layer it would freeze every
    inferred and observed fact as a permanent hand correction, and no later
    refresh could ever improve them — the opposite of what the layering is for.

    So this diffs against what the layers BELOW curated say and stores only the
    entries that disagree. Saving a form you did not touch is a no-op, and
    clearing an override restores whatever the machine had learned.
    """
    from . import server

    baseline = server._merged_routing_config(_load(), include_curated=False)
    below = baseline.get(key)

    def _mutate(curated: dict) -> None:
        if key in server._ROUTING_LIST_KEYS:
            inherited = {e.lower() for e in (below or []) if isinstance(e, str)}
            wanted = {e.lower() for e in (payload or []) if isinstance(e, str)}
            # Only additions are recordable: the curated layer is additive for
            # lists, so an entry inherited from below cannot be removed here.
            curated[key] = sorted(wanted - inherited)
        else:
            inherited = below if isinstance(below, dict) else {}
            kept = {}
            for k, v in (payload or {}).items():
                if not isinstance(k, str):
                    continue
                if inherited.get(k.lower()) != v:
                    kept[k.lower()] = v
            curated[key] = kept

    if not _curated_write(_mutate):
        return _err("Failed to persist routing metadata.", 500)
    return jsonify({key: _effective_routing().get(key)})


@bp.route("/admin/api/favorite-free-models", methods=["GET", "PUT"])
def api_favorite_free_models():
    if request.method == "GET":
        return jsonify({"favorite_free_models": _load().get("favorite_free_models", [])})

    def validate(p):
        if not (isinstance(p, list) and all(isinstance(x, str) for x in p)):
            return "favorite_free_models must be a list of strings."
        return None
    return _put_section("favorite_free_models", validate)


@bp.route("/admin/api/believed-free", methods=["GET", "PUT"])
def api_believed_free():
    if request.method == "GET":
        return jsonify({"believed_free": _effective_routing().get("believed_free") or []})

    def validate(p):
        if not (isinstance(p, list) and all(isinstance(x, str) for x in p)):
            return "believed_free must be a list of strings."
        return None
    return _put_section("believed_free", validate)


@bp.route("/admin/api/model-reasoning", methods=["GET", "PUT"])
def api_model_reasoning():
    if request.method == "GET":
        return jsonify({"model_reasoning": _effective_routing().get("model_reasoning") or {}})

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
        return jsonify({"model_capabilities": _effective_routing().get("model_capabilities") or {}})

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
        return jsonify({"free_limits": _effective_routing().get("free_limits") or {}})

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

# ---------------------------------------------------------------------------
# Routing metadata — the effective view, and the hand-set layer
# ---------------------------------------------------------------------------

def _curated_write(mutate) -> bool:
    """Apply *mutate* to the sidecar's curated section under the sidecar lock.

    The admin UI writes here rather than to config.json. config.json is drained
    into this section at startup and is no longer where hand-set routing facts
    live, so an editor pointed at it would be saving into a file the router
    stops reading the moment the next restart migrates it.
    """
    from . import server
    try:
        with server._routing_sidecar_txn() as state:
            mutate(server._curated_facts(state))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[admin:_curated_write] {e}")
        traceback.print_exc()
        return False


def _effective_routing(config: dict | None = None) -> dict:
    """The merged routing metadata the ROUTER actually uses.

    The editors used to read config.json alone. Once its five keys were
    migrated out that showed every model as un-free, untagged and incapable —
    and saving the form wrote those blanks back as real overrides. Reading the
    merged view is what makes the page show the truth instead.
    """
    from . import server
    merged = server._merged_routing_config(config if config is not None else _load())
    return {k: merged.get(k) for k in server._ROUTING_CONFIG_KEYS}


def layers_for_ids(config: dict):
    """The routing layers, assembled once.

    Once, not once per row: four layer rebuilds per model was most of the cost
    of listing a few thousand of them.
    """
    from . import server
    return server.routing_layers(config)


def _routing_rows(config: dict, q: str = "") -> list[dict]:
    """One row per known model: what is in effect, and which layers said so."""
    from . import server

    layers = layers_for_ids(config)

    eff = _effective_routing(config)
    believed = {m.lower() for m in eff.get("believed_free") or []}
    observed_cost = {m.lower() for m in eff.get("cost_observed_free_tier") or []}
    reasoning = eff.get("model_reasoning") or {}
    caps = eff.get("model_capabilities") or {}
    limits = eff.get("free_limits") or {}

    # Rows are ROUTING TARGETS and ids a person actually typed — never the
    # learned layer's keys. Those are normalized join keys by design
    # ("aionlabsaion30mini"), not anything callable, and listing them showed
    # thousands of phantom models that no request could ever address.
    ids: set[str] = set()
    try:
        for provider_name, upstream in server._get_distinct_routes():
            ids.add(f"{provider_name}/{upstream}".lower())
    except Exception as e:  # noqa: BLE001 — a cold route cache is not an error
        print(f"[admin:_routing_rows] {e}")
        traceback.print_exc()
    for layer_name, layer in layers_for_ids(config):
        if layer_name == "learned":
            continue
        for key in server._ROUTING_LIST_KEYS:
            ids |= {m.lower() for m in layer.get(key) or [] if isinstance(m, str)}
        for key in server._ROUTING_DICT_KEYS:
            ids |= {m.lower() for m in (layer.get(key) or {})
                    if isinstance(m, str) and m != "_note"}

    needle = q.strip().lower()
    rows: list[dict] = []
    for model_id in sorted(ids):
        if needle and needle not in model_id:
            continue
        provider = model_id.split("/", 1)[0] if "/" in model_id else ""
        upstream = model_id.split("/", 1)[1] if "/" in model_id else model_id
        try:
            from .flagship import normalize_model_id
            key = normalize_model_id(upstream)
        except Exception as e:  # noqa: BLE001
            print(f"[admin:_routing_rows] {e}")
            traceback.print_exc()
            key = upstream
        rows.append({
            "id": model_id,
            "free": model_id in believed,
            "cost_observed": model_id in observed_cost,
            "reasoning": server._lookup_model_fact(reasoning, provider, upstream),
            "capabilities": sorted(
                server._lookup_capabilities(
                    {k: set(v or ()) for k, v in caps.items()}, provider, upstream)),
            "free_limits": server._lookup_model_fact(limits, provider, upstream) or {},
            "layers": server.routing_fact_sources(model_id, provider, config,
                                                  layers=layers),
            "grades": server.learned_fact_grades(key),
        })
    return rows


@bp.route("/admin/api/routing-metadata", methods=["GET"])
def api_routing_metadata():
    """Paged, filtered view of the effective routing metadata.

    Paged server-side because a deployment with a couple of dozen providers
    sees thousands of models, and the grid used to render every one of them —
    rebuilding the whole table, with six listeners per row, on every keystroke.
    """
    from . import server

    config = _load()
    q = request.args.get("q", "")
    try:
        offset = max(0, int(request.args.get("offset", 0)))
        limit = min(500, max(1, int(request.args.get("limit", 100))))
    except (TypeError, ValueError):
        return _err("offset and limit must be integers.")
    rows = _routing_rows(config, q)
    return jsonify({
        "models": rows[offset:offset + limit],
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "layers": list(server.ROUTING_LAYER_NAMES),
    })


@bp.route("/admin/api/routing-metadata", methods=["PUT"])
def api_put_routing_metadata():
    """Set one model's facts by hand, recorded as curated.

    Per model rather than whole-section, so editing one row cannot blank the
    rest — which is exactly how the old whole-section PUT could wipe the
    learned layer in a single click.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _err("Body must be an object.")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        return _err("'model' is required.")
    model = model.strip().lower()

    caps = payload.get("capabilities")
    if caps is not None:
        if not (isinstance(caps, list) and all(isinstance(c, str) for c in caps)):
            return _err("'capabilities' must be a list of strings.")
        bad = {c.lower() for c in caps} - _VALID_CAPABILITIES
        if bad:
            return _err(f"Invalid capabilities {sorted(bad)}. "
                        f"Valid: {sorted(_VALID_CAPABILITIES)}.")
    level = payload.get("reasoning")
    if level not in (None, "") and level not in _providers.VALID_REASONING_LEVELS:
        return _err(f"Invalid reasoning level '{level}'. "
                    f"Valid: {sorted(_providers.VALID_REASONING_LEVELS)}.")
    limits = payload.get("free_limits")
    if limits is not None:
        if not isinstance(limits, dict):
            return _err("'free_limits' must be an object.")
        for k, v in limits.items():
            if k not in _providers.FREE_LIMIT_KEYS:
                return _err(f"Invalid limit key '{k}'. "
                            f"Valid: {list(_providers.FREE_LIMIT_KEYS)}.")
            if v is not None and not isinstance(v, int):
                return _err(f"Limit '{k}' must be an integer or null.")
    free = payload.get("free")
    if free is not None and not isinstance(free, bool):
        return _err("'free' must be a boolean.")

    def _mutate(curated: dict) -> None:
        if caps is not None:
            target = curated.setdefault("model_capabilities", {})
            if caps:
                target[model] = sorted({c.lower() for c in caps})
            else:
                target.pop(model, None)
        if level is not None:
            target = curated.setdefault("model_reasoning", {})
            if level:
                target[model] = level
            else:
                target.pop(model, None)
        if limits is not None:
            target = curated.setdefault("free_limits", {})
            if limits:
                target[model] = limits
            else:
                target.pop(model, None)
        if free is not None:
            entries = curated.setdefault("believed_free", [])
            present = model in {e.lower() for e in entries if isinstance(e, str)}
            if free and not present:
                entries.append(model)
            elif not free and present:
                curated["believed_free"] = [
                    e for e in entries
                    if not (isinstance(e, str) and e.lower() == model)
                ]

    if not _curated_write(_mutate):
        return _err("Failed to persist routing metadata.", 500)
    rows = _routing_rows(_load(), model)
    return jsonify({"model": next((r for r in rows if r["id"] == model), {"id": model})})


@bp.route("/admin/api/refresh", methods=["POST"])
def api_refresh():
    """Run a maintenance pass now instead of waiting for its cadence.

    Every refresh was cadence-driven with no way to ask for one, so a user who
    corrected a provider or added a key had to wait out the interval to see the
    routing change.
    """
    payload = request.get_json(silent=True) or {}
    which = payload.get("what", "routing_metadata")
    from . import server
    try:
        if which == "routing_metadata":
            config = _load()
            from .config import routing_metadata_cfg
            if not routing_metadata_cfg(config).get("enabled", True):
                # Matches the flagship branch, which goes through a fire helper
                # that checks its own gate. "Refresh now" changes the timing,
                # never the decision to maintain this at all.
                return _err("routing_metadata.enabled is false; "
                            "turn it on before refreshing.", 409)
            state = server._recompute_routing_metadata(config, None)
            if state is None:
                return _err("Refresh learned nothing; previous state kept.", 409)
            return jsonify({
                "refreshed": which,
                "models": len(state.get("by_model") or {}),
                "last_refresh_at": state.get("last_refresh_at"),
            })
        if which == "flagship":
            server._maybe_fire_flagship_refresh(
                _load(), _flagship_cfg_forced(), None)
            return jsonify({"refreshed": which, "started": True})
    except Exception as e:  # noqa: BLE001
        print(f"[admin:api_refresh] {e}")
        traceback.print_exc()
        return _err(f"Refresh failed: {e}", 500)
    return _err(f"Unknown refresh target '{which}'. "
                "Valid: routing_metadata, flagship.")


def _flagship_cfg_forced() -> dict:
    """Flagship config with the cadence gate removed, for an on-demand refresh."""
    from .config import flagship_tier_cfg
    cfg = dict(flagship_tier_cfg(_load()))
    cfg["refresh_frequency_days"] = 0
    return cfg


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
            "id": "llmproxy/free",
            "description": "Free-tier models (ID contains 'free' or listed in believed_free).",
            "backing": free_models or sorted(believed_free),
        })

    local_providers = [n for n, c in providers.items() if server._is_local_url(provider_base_url(c))]
    if local_providers:
        virtuals.append({
            "id": "llmproxy/local",
            "description": "All models served by localhost providers.",
            "backing": [f"{n}/* (all models)" for n in sorted(local_providers)],
        })

    for level in server._REASONING_LEVELS:
        backing = sorted(m for m, lvl in reasoning.items() if lvl == level)
        if backing:
            virtuals.append({
                "id": f"llmproxy/{level}",
                "description": f"Models tagged '{level}' reasoning.",
                "backing": backing,
            })

    for cap in server._CAPABILITY_VIRTUALS:
        backing = sorted(m for m, caps in capabilities.items() if cap in (caps or []))
        if backing:
            virtuals.append({
                "id": f"llmproxy/{cap}",
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
