"""
config.py — Configuration loading, saving, and schema validation for llmproxy.

Config is stored at ~/.config/llmproxy/config.json (overridable via
LLMPROXY_CONFIG environment variable or the --config CLI flag).

Schema:
{
  "providers": {
    "<provider_name>": {
      "base_url": "https://...",
      "api_key": "sk-...",
      "model_filter": ["model-a", "model-b"],  // null or absent = allow all
      "expose_to_virtual_models": false         // optional; default true. Set
                                               // false to hide this provider
                                               // from ALL virtual endpoints
                                               // (free/local/deep/tools/etc.)
                                               // Models still appear in the
                                               // flat /v1/models list and can
                                               // be called directly.
    }
  },
  "believed_free": ["model-a", "provider/model-b"],  // models the 'free' virtual
                                                  // model should include even
                                                  // when their ID lacks 'free'
  "model_reasoning": {                            // optional; tag individual
    "<upstream_model_id>": "exploratory",         // models with a reasoning
    "<provider>/<upstream_model_id>": "standard", // level so they appear under
    "another-model": "deep"                       // the exploratory/standard/deep
  },                                              // virtual endpoints
  "model_capabilities": {                         // optional; tag individual models
    "<upstream_model_id>": ["tools", "vision"],   // with the capabilities they
    "<provider>/<upstream_model_id>": ["json"]    // support. Drives capability-aware
  },                                              // routing/failover and the
                                                  // llmproxy__tools / __vision virtual
                                                  // endpoints. Valid values:
                                                  // tools, vision, reasoning, json
  "free_limits": {                                // optional; per-model rate limits
    "<provider>/<upstream_model_id>": {           // used for capacity-aware ordering
      "requests_per_minute": 15,                  // on llmproxy__free and */free
      "requests_per_day": 1500,                   // endpoints; null = not tracked
      "tokens_per_minute": null,                  // token limits; null = not tracked.
      "tokens_per_day": null                      // Enforced in capacity ordering from
    }                                             // recorded usage, like the request limits.
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "log_level": "INFO",
    "request_timeout": 120,
    "stream_timeout": 300
  },
  "admin": {                                      // optional; web admin UI at /admin
    "enabled": true,                              // default true; serve the UI/API
    "token": "${LLMPROXY_ADMIN_TOKEN}"            // optional bearer token. When unset,
  }                                               // /admin is reachable from loopback
}                                                 // only; when set, any origin that
                                                  // presents the token is allowed.

Environment-variable references
-------------------------------
The string fields ``api_key`` and ``base_url`` (and the admin ``token``) may
contain ``${VAR}`` references, e.g. ``"api_key": "${OPENAI_API_KEY}"`` or
``"base_url": "http://${OLLAMA_HOST}:11434/v1"``. References are resolved from
the process environment at request time (see ``resolve_env_refs`` and the
``provider_api_key`` / ``provider_base_url`` accessors), so secrets never need to
be written literally into config.json.
"""

import copy
import json
import os
import re
import tempfile
import traceback
from pathlib import Path

from . import providers as _providers

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "llmproxy"
_DEFAULT_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "config.json"


def get_config_path(override: str | None = None) -> Path:
    """
    Return the resolved config file path.

    Resolution order (highest to lowest priority):
      1. *override* argument (from --config CLI flag)
      2. LLMPROXY_CONFIG environment variable (read at call time)
      3. ~/.config/llmproxy/config.json
    """
    if override:
        return Path(override)
    env_path = os.environ.get("LLMPROXY_CONFIG")
    if env_path:
        return Path(env_path)
    return _DEFAULT_CONFIG_FILE


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Provider names in this set are reserved by the proxy itself and must not be
# used in config['providers'].  The setup wizard enforces this interactively;
# the server enforces it at model-list build time.
RESERVED_PROVIDER_NAMES: frozenset[str] = frozenset({"llmproxy"})

DEFAULT_SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8080,
    "log_level": "INFO",
    "request_timeout": 120,
    "stream_timeout": 300,
}

# Web admin UI defaults. The UI is enabled by default but, with no token set, is
# reachable only from loopback (see llmproxy/admin.py). Setting a token allows
# remote access for callers that present it.
DEFAULT_ADMIN_CONFIG = {
    "enabled": True,
    "token": "",
}

# Free-tier maintenance defaults. Groups the startup sync/update switches with
# the cost-probe controls they drive. Replaces the former flat top-level keys
# sync_believed_free_on_startup / update_believed_free_on_startup / probe_cost /
# autoremove_believed_free / probe_frequency_days (still accepted on input via
# _normalize_config below for backward compatibility).
DEFAULT_FREE_TIER_CONFIG = {
    "sync_on_startup": True,
    "update_on_startup": False,
    "probe": {
        "enabled": False,
        "autoremove": False,
        "frequency_days": 0,
    },
}

# Providers-PR defaults. Groups the auto-PR switch with its repo/branch/token.
# Replaces the former flat pr_providers_list / pr_providers_repo /
# pr_providers_base / pr_providers_branch / pr_providers_token keys (still
# accepted on input via _normalize_config below).
DEFAULT_PROVIDERS_PR_CONFIG = {
    "enabled": False,
    "repo": None,
    "base": "main",
    "branch": "llmproxy-auto/providers",
    "token": None,
}

# Fusion (multi-model deliberation) defaults. See llmproxy/fusion.py and the
# llmproxy__fusion / llmproxy__fusion/free virtual models in server.py.
DEFAULT_FUSION_CONFIG = {
    "enabled": True,
    "panel": None,            # explicit model list for bare fusion; None -> full pool
    "panel_size": 4,
    "diversity": "provider",  # "provider" -> prefer distinct providers/families; "none"
    "judge_model": None,      # None -> auto-pick a capable model
    "synthesizer_model": None,  # None -> auto-pick a capable model
    "allow_paid": True,       # bare fusion may use paid models; fusion/free never does
    "report": {"metadata": True},
    "forced_capability": "restrict",  # "restrict" -> panel+synth must be capable; "bypass"
}

DEFAULT_CONFIG: dict = {
    "providers": {},
    "believed_free": [],
    # Qualified provider/model ids that served a request reporting a non-zero cost
    # while marked believed_free. The proxy appends to this set at runtime (see
    # server._persist_cost_observed); the updater treats membership as a hard
    # "not free" signal — such models are never re-added to believed_free and are
    # removed if present. Operator-editable.
    "cost_observed_free_tier": [],
    "model_reasoning": {},
    "model_capabilities": {},
    "free_limits": {},
    "free_tier": dict(DEFAULT_FREE_TIER_CONFIG),
    "providers_pr": dict(DEFAULT_PROVIDERS_PR_CONFIG),
    "fusion": dict(DEFAULT_FUSION_CONFIG),
    "server": dict(DEFAULT_SERVER_CONFIG),
    "admin": dict(DEFAULT_ADMIN_CONFIG),
}

# Mapping from each legacy flat top-level key to its new nested location, used by
# _normalize_config to migrate configs written before the reorganization. Each
# value is the tuple path into the nested config.
_LEGACY_KEY_MIGRATIONS: dict[str, tuple[str, ...]] = {
    "sync_believed_free_on_startup": ("free_tier", "sync_on_startup"),
    "update_believed_free_on_startup": ("free_tier", "update_on_startup"),
    "probe_cost": ("free_tier", "probe", "enabled"),
    "autoremove_believed_free": ("free_tier", "probe", "autoremove"),
    "probe_frequency_days": ("free_tier", "probe", "frequency_days"),
    "pr_providers_list": ("providers_pr", "enabled"),
    "pr_providers_repo": ("providers_pr", "repo"),
    "pr_providers_base": ("providers_pr", "base"),
    "pr_providers_branch": ("providers_pr", "branch"),
    "pr_providers_token": ("providers_pr", "token"),
}


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

# Simple mtime-based hot-reload cache.
_cache: dict = {}
_cache_mtime: float = 0.0


def load_config(config_path: str | None = None, force_reload: bool = False) -> dict:
    """
    Load configuration from disk.

    Uses a modification-time cache so that repeated reads within a single
    request cycle do not hit disk, while still picking up changes between
    requests without a server restart.

    Parameters
    ----------
    config_path : str, optional
        Explicit path override; falls back to get_config_path().
    force_reload : bool
        Bypass the cache and re-read from disk unconditionally.

    Returns
    -------
    dict
        Merged configuration (file values overlaid on defaults).
    """
    global _cache, _cache_mtime

    path = get_config_path(config_path)

    if not path.exists():
        return _deep_merge(DEFAULT_CONFIG, {})

    try:
        mtime = path.stat().st_mtime
        if not force_reload and _cache and mtime == _cache_mtime:
            return _cache

        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

        merged = _deep_merge(DEFAULT_CONFIG, _normalize_config(raw))
        _cache = merged
        _cache_mtime = mtime
        return merged

    except Exception as e:
        print(f"[config:load_config] Failed to load {path}: {e}")
        traceback.print_exc()
        return _deep_merge(DEFAULT_CONFIG, {})


def save_config(config: dict, config_path: str | None = None) -> bool:
    """
    Persist configuration to disk, creating parent directories as needed.

    Parameters
    ----------
    config : dict
        Full configuration dictionary to serialize.
    config_path : str, optional
        Explicit path override.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    global _cache, _cache_mtime

    path = get_config_path(config_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then atomically replace the
        # target. This prevents a crash or concurrent admin-UI/wizard write from
        # truncating config.json and leaving an unparseable file behind.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except Exception:
            # Best-effort cleanup of the temp file on any failure.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        # Invalidate cache
        _cache = {}
        _cache_mtime = 0.0
        print(f"Configuration saved to {path}")
        return True
    except Exception as e:
        print(f"[config:save_config] Failed to write {path}: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Probe state (machine-managed cache, kept out of the user-edited config)
# ---------------------------------------------------------------------------
#
# The cost probe (scripts/sources/probe.py) can be throttled to at most once
# every ``probe_frequency_days``. The last-run timestamp is stored in a small
# sibling cache file rather than in config.json so we don't churn the
# hand-edited config (or sweep the timestamp into the providers-PR / sync flow).

def get_probe_state_path(config_path: str | None = None) -> Path:
    """Return the path to the probe-state cache file, a sibling of config.json."""
    return get_config_path(config_path).parent / "probe_state.json"


def load_probe_state(config_path: str | None = None) -> dict:
    """Load the probe-state cache, returning {} when absent or unreadable."""
    path = get_probe_state_path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001 — a corrupt cache must never break a run
        print(f"[config:load_probe_state] Failed to load {path}: {e}")
        return {}


def save_probe_state(state: dict, config_path: str | None = None) -> bool:
    """Persist the probe-state cache atomically (tempfile + rename)."""
    path = get_probe_state_path(config_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[config:save_probe_state] Failed to write {path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def get_provider(config: dict, provider_name: str) -> dict | None:
    """Return the provider config dict for *provider_name*, or None if absent."""
    return config.get("providers", {}).get(provider_name)


def model_is_allowed(provider_cfg: dict, upstream_model: str) -> bool:
    """
    Return True if *upstream_model* passes the provider's model filter.

    Semantics:
      model_filter = None  → no filter configured; all models permitted.
      model_filter = []    → explicit empty allowlist; no models permitted.
      model_filter = [..] → only models in the list are permitted.

    The distinction between None and [] matters: None means "I haven't set
    a filter", while [] would mean "allow nothing" (unusual but unambiguous).
    We use `is None` rather than truthiness so that an empty list is not
    silently treated as "allow all".
    """
    model_filter = provider_cfg.get("model_filter")
    if model_filter is None:
        return True
    return upstream_model in model_filter


def parse_model_string(model_full: str) -> tuple[str, str]:
    """
    Split a proxy model string into (provider_name, upstream_model).

    The proxy convention is:  <provider_name>/<upstream_model_id>
    where <upstream_model_id> may itself contain slashes.

    Example
    -------
    >>> parse_model_string("openrouter/openrouter/free")
    ('openrouter', 'openrouter/free')

    Raises
    ------
    ValueError
        If the string contains no '/' separator.
    """
    sep = model_full.find("/")
    if sep == -1:
        raise ValueError(
            f"Model '{model_full}' does not follow the required "
            f"'<provider>/<model>' convention."
        )
    return model_full[:sep], model_full[sep + 1:]


# ---------------------------------------------------------------------------
# Environment-variable references — runtime resolution for secrets/endpoints
# ---------------------------------------------------------------------------

# A ${VAR} reference inside a string field (currently api_key and base_url).
# References are resolved from os.environ at *consumption* time (see
# provider_api_key / provider_base_url and their call sites in server.py), never
# at load_config() time. This keeps the on-disk config — and everything the admin
# UI / setup wizard read back — as the raw reference, so secrets never need to
# live literally in config.json (set e.g. "api_key": "${OPENAI_API_KEY}").
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_env_refs(value):
    """Substitute every ``${VAR}`` in *value* with ``os.environ[VAR]``.

    Resolution happens at call time, so the same config picks up environment
    changes without a rewrite. An unset variable resolves to the empty string.
    Non-string values (and strings without a ``${`` marker) pass through
    unchanged, so this is cheap and safe to call on any field.
    """
    if not isinstance(value, str) or "${" not in value:
        return value
    return _ENV_REF_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


def provider_base_url(provider_cfg: dict) -> str:
    """Return the provider's base_url with ``${VAR}`` refs resolved and no
    trailing slash. Use this everywhere a request URL is built or a base_url is
    inspected (e.g. localhost detection)."""
    return (resolve_env_refs(provider_cfg.get("base_url")) or "").rstrip("/")


def provider_api_key(provider_cfg: dict) -> str:
    """Return the provider's api_key with ``${VAR}`` refs resolved. Use this
    wherever the Authorization bearer token is built."""
    return resolve_env_refs(provider_cfg.get("api_key")) or ""


def value_has_env_ref(value) -> bool:
    """True if *value* is a string containing at least one ``${VAR}`` reference.

    The admin UI uses this to decide whether a field is a (non-secret) env
    reference that can be shown verbatim, versus a literal secret that must be
    masked.
    """
    return isinstance(value, str) and bool(_ENV_REF_RE.search(value))


# ---------------------------------------------------------------------------
# Auto-heal — backfill template-derived fields missing from older configs
# ---------------------------------------------------------------------------

# Provider fields that carry a canonical value in the provider template and
# whose absence breaks model discovery. These were added after the initial
# release, so configs created earlier lack them. base_url / api_key are user
# secrets and deliberately out of scope.
_HEALABLE_FIELDS = ("models_url", "models_id_field", "models_keep_task")


def _template_base_url_regex(template_base_url: str) -> re.Pattern:
    """Compile a regex that matches a resolved base_url against a template.

    Each ``{placeholder}`` in the template becomes a named capture group, so a
    match both confirms the template and recovers the substituted values
    (e.g. ``{account_id}``). Literal segments are escaped.
    """
    parts = re.split(r"(\{[a-zA-Z_]+\})", template_base_url)
    pattern = ""
    for part in parts:
        m = re.fullmatch(r"\{([a-zA-Z_]+)\}", part)
        if m:
            pattern += f"(?P<{m.group(1)}>[^/]+)"
        else:
            pattern += re.escape(part)
    return re.compile(f"^{pattern}/?$")


def _match_template(provider_name: str, provider_cfg: dict, templates: dict) -> tuple[dict | None, dict]:
    """Resolve which template a configured provider came from.

    Returns ``(template, placeholder_values)``. Matching is two-tier:
      1. By name — the config provider name equals a template key (the common
         case; the wizard defaults the name to the template key).
      2. By base_url — the provider's resolved base_url matches a template's
         base_url pattern, which also recovers any ``{placeholder}`` values.

    ``placeholder_values`` is empty for a name match (no recovery needed unless
    base_url also matches, in which case it is populated).
    """
    base_url = (provider_cfg.get("base_url") or "").rstrip("/")

    tmpl = templates.get(provider_name)
    if tmpl is not None:
        placeholders: dict = {}
        tmpl_base = (tmpl.get("base_url") or "").rstrip("/")
        if tmpl_base:
            m = _template_base_url_regex(tmpl_base).match(base_url)
            if m:
                placeholders = m.groupdict()
        return tmpl, placeholders

    # Fallback: identify a renamed provider by its base_url shape.
    if base_url:
        for tmpl in templates.values():
            tmpl_base = (tmpl.get("base_url") or "").rstrip("/")
            if not tmpl_base:
                continue
            m = _template_base_url_regex(tmpl_base).match(base_url)
            if m:
                return tmpl, m.groupdict()
    return None, {}


def _reconstruct_field(field: str, template: dict, placeholders: dict) -> str | None:
    """Reconstruct a healable field value from the template, or None if it
    requires information we cannot recover without user input."""
    value = template.get(field)
    if not value:
        return None
    if field != "models_url":
        # models_id_field / models_keep_task are static literals.
        return value
    # models_url may carry {account_id} / {gateway_id} placeholders that must be
    # substituted with the same values resolved into the provider's base_url.
    missing = re.findall(r"\{([a-zA-Z_]+)\}", value)
    for name in missing:
        if name not in placeholders:
            return None  # can't fabricate the id; caller will warn.
        value = value.replace(f"{{{name}}}", placeholders[name])
    return value


def heal_config(config: dict) -> tuple[dict, bool, list[tuple[str, str]]]:
    """Backfill missing template-derived provider fields in *config*.

    For each configured provider that matches a known provider template, fill
    in any of the model-discovery fields (models_url / models_id_field /
    models_keep_task) the template defines but the provider lacks. Existing
    keys are never overwritten, so this is idempotent and safe.

    Returns ``(config, changed, messages)`` where *changed* is True if any
    field was added and *messages* is a list of ``(level, text)`` pairs with
    *level* in {"info", "warning"} for the caller to log.
    """
    messages: list[tuple[str, str]] = []
    changed = False

    try:
        templates = {t["key"]: t for t in _providers.get_provider_templates()}
    except Exception as e:  # pragma: no cover - templates ship with the package
        messages.append((
            "warning",
            f"Could not load provider templates; skipping config auto-heal: {e}",
        ))
        return config, False, messages

    for name, provider_cfg in config.get("providers", {}).items():
        if not isinstance(provider_cfg, dict):
            continue
        template, placeholders = _match_template(name, provider_cfg, templates)
        if template is None:
            continue
        for field in _HEALABLE_FIELDS:
            if field in provider_cfg:
                continue
            # Only heal when the template actually defines a usable value.
            # providers.get_provider_templates() copies fields verbatim from
            # providers.json, so a null/empty entry must be skipped silently
            # rather than treated as a healable-but-unrecoverable field.
            template_value = template.get(field)
            if not isinstance(template_value, str) or not template_value:
                continue
            value = _reconstruct_field(field, template, placeholders)
            if value is None:
                messages.append((
                    "warning",
                    f"Provider '{name}' is missing '{field}' and it cannot be "
                    f"auto-healed; re-run 'llmproxy --setup' to repair it.",
                ))
                continue
            provider_cfg[field] = value
            changed = True
            messages.append((
                "info",
                f"Auto-healed provider '{name}': added {field}={value}",
            ))

    return config, changed, messages


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _nested_present(d: dict, path: tuple[str, ...]) -> bool:
    """True if *path* is explicitly present (to any depth) in nested dict *d*."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def _nested_set(d: dict, path: tuple[str, ...], value) -> None:
    """Set *value* at *path* in *d*, creating intermediate dicts as needed."""
    cur = d
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _normalize_config(raw: dict) -> dict:
    """Migrate legacy flat top-level keys into their new nested homes.

    The free-tier maintenance and providers-PR switches were originally flat
    top-level keys (probe_cost, pr_providers_list, ...). They now live under the
    ``free_tier`` and ``providers_pr`` objects. To keep configs written before
    the reorganization working unchanged, any legacy flat key still present is
    lifted into its nested location here, unless the user has *also* explicitly
    set the nested form (in which case the nested form wins and the legacy key is
    simply dropped). The returned dict is a shallow-safe copy with the legacy
    keys removed, so all downstream readers see only the canonical nested shape.

    A no-op for configs that already use the nested shape.
    """
    if not isinstance(raw, dict):
        return raw
    if not any(k in raw for k in _LEGACY_KEY_MIGRATIONS):
        return raw

    normalized = copy.deepcopy(raw)
    for legacy_key, path in _LEGACY_KEY_MIGRATIONS.items():
        if legacy_key not in normalized:
            continue
        value = normalized.pop(legacy_key)
        # Nested form set by the user takes precedence over the legacy value.
        if not _nested_present(raw, path):
            try:
                _nested_set(normalized, path, value)
            except Exception as e:  # noqa: BLE001
                print(f"[config:_normalize_config] failed migrating {legacy_key}: {e}")
                traceback.print_exc()
    return normalized


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into a copy of *base*.

    Scalar and list values in *override* replace those in *base*.
    Dict values are merged recursively.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
