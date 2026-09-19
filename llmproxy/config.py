"""
config.py — Configuration loading, saving, and schema validation for llmproxy.

Config is stored at ~/.config/llmproxy/config.json (overridable via
LLMPROXY_CONFIG environment variable or the --config CLI flag).

Schema:
{
  "providers": {
    "<provider_name>": {
      "base_url": "https://...",
      "api_key": "sk-...",                      // single credential (legacy)
      "accounts": [                             // optional; multiple credentials
        {"key": "sk-a", "label": "team-a"},     // for this provider. The proxy
        {"key": "${KEY_B}", "priority": 1}      // rotates across them to multiply
      ],                                        // free-tier headroom, keying quota
                                                // + saturation state per account.
                                                // "api_keys": ["sk-a","sk-b"] is a
                                                // shorthand. api_key stays the
                                                // fallback when neither is set.
      "account_strategy": "round_robin",        // optional; round_robin (default)
                                                // spreads load, priority prefers
                                                // the lowest-priority account first
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
  },                                              // virtual endpoints.
                                                  // 'flagship' is NOT settable
                                                  // here: it is a computed
                                                  // overlay above deep — see
                                                  // flagship_tier below
  "flagship_tier": { ... },                       // optional; policy for the
                                                  // computed llmproxy/flagship
                                                  // tier. The membership list
                                                  // itself is deployment-
                                                  // specific and cached in
                                                  // flagship_models.json, not
                                                  // stored here. See the README
                                                  // section "The flagship tier"
  "model_capabilities": {                         // optional; tag individual models
    "<upstream_model_id>": ["tools", "vision"],   // with the capabilities they
    "<provider>/<upstream_model_id>": ["json"]    // support. Drives capability-aware
  },                                              // routing/failover and the
                                                  // llmproxy/tools / vision virtual
                                                  // endpoints. Valid values:
                                                  // tools, vision, reasoning, json
  "free_limits": {                                // optional; per-model rate limits
    "<provider>/<upstream_model_id>": {           // used for capacity-aware ordering
      "requests_per_minute": 15,                  // on llmproxy/free and /*__free
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
from collections import namedtuple
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
# _normalize_config below for backward compatibility), and the former
# endpoint_probe block (flattened to probe_timeout_sec, also migrated there).
DEFAULT_FREE_TIER_CONFIG = {
    "sync_on_startup": True,
    "update_on_startup": False,
    # How often the free-models sweep runs at all. Everything else in this block
    # is a per-source throttle subordinate to it: a source runs only as part of
    # a sweep, so it can never run more often than this.
    "update_frequency_days": 7,
    # Read timeout for BOTH probes. They differ in what they spend, not in how
    # long to wait for a slow provider, so one setting covers both. Replaces the
    # former free_tier.endpoint_probe block, which held nothing else once the
    # endpoint probe stopped needing a frequency of its own.
    "probe_timeout_sec": 10,
    "cost_probe": {
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
# llmproxy/fusion / llmproxy/fusion__free virtual models in server.py.
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
    "probe_cost": ("free_tier", "cost_probe", "enabled"),
    "autoremove_believed_free": ("free_tier", "cost_probe", "autoremove"),
    "probe_frequency_days": ("free_tier", "cost_probe", "frequency_days"),
    "pr_providers_list": ("providers_pr", "enabled"),
    "pr_providers_repo": ("providers_pr", "repo"),
    "pr_providers_base": ("providers_pr", "base"),
    "pr_providers_branch": ("providers_pr", "branch"),
    "pr_providers_token": ("providers_pr", "token"),
}


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

# Hot-reload cache keyed on a (st_mtime_ns, st_size) fingerprint rather than a
# bare float mtime. Integer-nanosecond mtimes avoid float-equality fuzz, and the
# size tiebreaker catches a rewrite that lands within the same mtime tick — which
# is realistic on Docker volume filesystems with coarse (1s) mtime granularity
# (e.g. 9p on Docker Desktop, some bind-mount/network volume drivers). A bare
# mtime cache could otherwise pin a stale snapshot for the life of the process.
_cache: dict = {}
_cache_stat: tuple[int, int] = (0, 0)


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
    global _cache, _cache_stat

    path = get_config_path(config_path)

    if not path.exists():
        return _deep_merge(DEFAULT_CONFIG, {})

    try:
        st = path.stat()
        fingerprint = (st.st_mtime_ns, st.st_size)
        if not force_reload and _cache and fingerprint == _cache_stat:
            return _cache

        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

        merged = _deep_merge(DEFAULT_CONFIG, _normalize_config(raw))
        _cache = merged
        _cache_stat = fingerprint
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
    global _cache, _cache_stat

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
        _cache_stat = (0, 0)
        print(f"Configuration saved to {path}")
        return True
    except Exception as e:
        print(f"[config:save_config] Failed to write {path}: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Probe state (machine-managed caches, kept out of the user-edited config)
# ---------------------------------------------------------------------------
#
# The cost probe and PR creation are each throttled via their own frequency
# setting; the endpoint probe has none, because it spends no quota and runs on
# every sweep. The last-run timestamps live in small sibling cache files rather
# than in config.json so we don't churn the hand-edited config.

def _load_state_file(path: Path, label: str) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001 — a corrupt cache must never break a run
        print(f"[config:{label}] Failed to load {path}: {e}")
        return {}


def _save_state_file(state: dict, path: Path, label: str) -> bool:
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
        print(f"[config:{label}] Failed to write {path}: {e}")
        return False


# --- Cost probe state (cost_probe_state.json) ---

def get_cost_probe_state_path(config_path: str | None = None) -> Path:
    return get_config_path(config_path).parent / "cost_probe_state.json"


def load_cost_probe_state(config_path: str | None = None) -> dict:
    """Load cost probe state, migrating from the old probe_state.json if needed."""
    path = get_cost_probe_state_path(config_path)
    if not path.exists():
        old = path.parent / "probe_state.json"
        if old.exists():
            data = _load_state_file(old, "load_cost_probe_state")
            if data:
                _save_state_file(data, path, "load_cost_probe_state")
                return data
    return _load_state_file(path, "load_cost_probe_state")


def save_cost_probe_state(state: dict, config_path: str | None = None) -> bool:
    return _save_state_file(
        state, get_cost_probe_state_path(config_path), "save_cost_probe_state"
    )


# Back-compat aliases so any external callers of the old names still work.
def get_probe_state_path(config_path: str | None = None) -> Path:
    return get_cost_probe_state_path(config_path)


def load_probe_state(config_path: str | None = None) -> dict:
    return load_cost_probe_state(config_path)


def save_probe_state(state: dict, config_path: str | None = None) -> bool:
    return save_cost_probe_state(state, config_path)


# NOTE: there is no endpoint_probe_state.json. The endpoint probe has no
# throttle to remember a last-run time for — it runs on every sweep. An
# endpoint_probe_state.json left over from an older version is inert and can be
# deleted.


# --- Full-refresh state (update_state.json) ---
#
# Throttles the full free-models scrape (free_tier.update_frequency_days) so a
# restart-heavy deployment does not re-scrape every provider on every boot, and
# a long-lived process still refreshes on its configured cadence.

def get_update_state_path(config_path: str | None = None) -> Path:
    return get_config_path(config_path).parent / "update_state.json"


def load_update_state(config_path: str | None = None) -> dict:
    return _load_state_file(get_update_state_path(config_path), "load_update_state")


def save_update_state(state: dict, config_path: str | None = None) -> bool:
    return _save_state_file(
        state, get_update_state_path(config_path), "save_update_state"
    )


# Defaults for the top-level `flagship_tier` config block. Defined once here so
# the values the server falls back to for a config that predates the block, and
# the values scripts/update_free_models.py writes into config.example.json,
# cannot drift apart. See the README for what each key does.
FLAGSHIP_TIER_DEFAULTS: dict = {
    "enabled": True,
    "min_flagship_free_models": 5,
    "start_percentile": 0.9,
    "min_context": 200000,
    "require_tools": True,
    "max_models": None,
    "pin": [],
    "exclude": [],
    "sources": ["openrouter_aa"],
    "refresh_frequency_days": 7,
}


def flagship_tier_cfg(config: dict | None = None) -> dict:
    """The `flagship_tier` block with defaults filled in.

    A config written before the block existed simply gets every default, so the
    tier behaves identically whether or not the user has pasted the block in.
    """
    raw = (config or {}).get("flagship_tier")
    merged = dict(FLAGSHIP_TIER_DEFAULTS)
    if isinstance(raw, dict):
        merged.update({k: v for k, v in raw.items() if v is not None
                       or k in ("max_models",)})
    return merged


# --- Flagship tier membership + refresh state (flagship_models.json) ---
#
# Membership is DEPLOYMENT-SPECIFIC: it depends on which providers are
# configured and what each of them currently serves, so it is computed locally
# and cached here rather than shipped in providers.json or written into the
# hand-edited config.json. Nothing about it is committed to the repo.
#
# Shape: {"last_refresh_at": iso8601, "bar": float,
#         "members": ["provider/model", ...],
#         "scores": {"provider/model": {"combined": float, "model_key": str}},
#         "model_scores": {model_key: float},
#         "distinct_models": [model_key, ...], "free_models": [model_key, ...],
#         "candidates_considered": int}
#
# `members` says who is in the tier; `scores` and `model_scores` say how strong
# each one is, which is what lets the router walk a flagship pool strongest-first
# rather than in the arbitrary order `members` happens to carry. `model_scores`
# is keyed by normalised model rather than by routing target, because a score
# belongs to the weights and not to the provider serving them.

def get_flagship_state_path(config_path: str | None = None) -> Path:
    return get_config_path(config_path).parent / "flagship_models.json"


def load_flagship_state(config_path: str | None = None) -> dict:
    return _load_state_file(get_flagship_state_path(config_path), "load_flagship_state")


def save_flagship_state(state: dict, config_path: str | None = None) -> bool:
    return _save_state_file(
        state, get_flagship_state_path(config_path), "save_flagship_state"
    )


# --- Learned routing metadata (routing_metadata.json) ---
#
# The middle of three layers. Routing metadata — which models are free, what
# they cost, what they can do, how fast you may call them — is DEPLOYMENT
# specific and changes as providers add and withdraw models, so it cannot live
# in the repo; and it is MACHINE maintained, so it must not live in the file the
# user hand-edits. It sits here instead, refreshed on its own cadence:
#
#   config.json           overrides    (yours, and the admin UI's; always wins)
#   routing_metadata.json learned      (this file; rewritten by the refresh)
#   providers.json        defaults     (shipped with the repo, PR-able)
#
# Split by what each fact actually describes. A capability belongs to the
# weights, so it is keyed by normalized model and one entry covers every
# provider serving them. A rate limit belongs to the deployment, so it is keyed
# per provider. Keying them the same way would either lose the cross-provider
# join or invent per-provider capabilities that do not exist.
#
# Shape: {"last_refresh_at": iso8601,
#         "by_model":    {model_key: {"capabilities": [...], "reasoning": str}},
#         "by_provider": {provider: {"believed_free": [...],
#                                    "cost_observed_free_tier": [...],
#                                    "free_limits": {model: {...}}}},
#         "models_considered": int}

ROUTING_METADATA_DEFAULTS: dict = {
    "enabled": True,
    "refresh_frequency_days": 7,
}


def routing_metadata_cfg(config: dict | None = None) -> dict:
    """The `routing_metadata` block with defaults filled in."""
    raw = (config or {}).get("routing_metadata")
    merged = dict(ROUTING_METADATA_DEFAULTS)
    if isinstance(raw, dict):
        merged.update({k: v for k, v in raw.items() if v is not None})
    return merged


def get_routing_metadata_path(config_path: str | None = None) -> Path:
    return get_config_path(config_path).parent / "routing_metadata.json"


def load_routing_metadata(config_path: str | None = None) -> dict:
    return _load_state_file(
        get_routing_metadata_path(config_path), "load_routing_metadata"
    )


def save_routing_metadata(state: dict, config_path: str | None = None) -> bool:
    return _save_state_file(
        state, get_routing_metadata_path(config_path), "save_routing_metadata"
    )


# --- PR state (pr_state.json) ---

def get_pr_state_path(config_path: str | None = None) -> Path:
    return get_config_path(config_path).parent / "pr_state.json"


def load_pr_state(config_path: str | None = None) -> dict:
    return _load_state_file(get_pr_state_path(config_path), "load_pr_state")


def save_pr_state(state: dict, config_path: str | None = None) -> bool:
    return _save_state_file(
        state, get_pr_state_path(config_path), "save_pr_state"
    )


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


# ---------------------------------------------------------------------------
# Multiple accounts per provider — free-tier headroom via credential rotation
# ---------------------------------------------------------------------------

# A single provider may carry several credentials ("accounts") so the proxy can
# rotate across them and multiply that provider's free-tier headroom. An account
# is a resolved key plus display metadata; ``id`` is the stable handle used to
# key per-account usage/saturation state. A provider with exactly one account
# (the common case — a lone ``api_key``) uses ``id=None`` so every downstream
# usage key stays byte-identical to the historical ``provider/model`` form.
Account = namedtuple("Account", ["id", "key", "key_raw", "label", "priority"])


def _account_priority(entry: dict) -> int:
    """Coerce an account's optional ``priority`` to an int (default 0)."""
    try:
        return int(entry.get("priority"))
    except (TypeError, ValueError):
        return 0


def _account_id_from_label(label, idx: int) -> str:
    """Derive a stable, key-safe account id from a label, falling back to ``kN``.

    The id becomes part of ``provider#<id>/model`` usage keys, so any character
    that would confuse that form (``/``, ``#``, whitespace) is collapsed to
    ``_``. An empty/blank label falls back to the account's declared index.
    """
    if label:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_")
        if safe:
            return safe
    return f"k{idx}"


def _normalize_account_entries(provider_cfg: dict) -> list:
    """Return the raw account dicts declared on a provider, in declared order.

    Recognized shapes, in precedence order (first non-empty wins):
      * ``accounts`` — list of ``{"key": ..., "label"?: ..., "priority"?: ...}``
        (a bare string entry is also accepted as shorthand for ``{"key": ...}``)
      * ``api_keys`` — list of key strings, each becoming an account
      * ``api_key``  — the legacy single-key field (one account)

    The legacy ``api_key`` is always the fallback, so every existing config —
    including keyless local providers (``api_key`` absent) — keeps working.
    """
    entries: list = []
    raw = provider_cfg.get("accounts")
    if isinstance(raw, list) and raw:
        for item in raw:
            if isinstance(item, dict) and item.get("key") is not None:
                entries.append({
                    "key": item.get("key"),
                    "label": item.get("label"),
                    "priority": item.get("priority"),
                })
            elif isinstance(item, str):
                entries.append({"key": item, "label": None, "priority": None})
        if entries:
            return entries
    keys = provider_cfg.get("api_keys")
    if isinstance(keys, list) and keys:
        for item in keys:
            if isinstance(item, str):
                entries.append({"key": item, "label": None, "priority": None})
        if entries:
            return entries
    return [{"key": provider_cfg.get("api_key"), "label": None, "priority": None}]


def provider_account_strategy(provider_cfg: dict) -> str:
    """Return the provider's account-rotation strategy.

    ``round_robin`` (default) spreads load across accounts; ``priority`` always
    prefers the lowest-``priority`` account first, falling through to the next
    only when it is exhausted. Unknown values fall back to ``round_robin``.
    """
    strat = str(provider_cfg.get("account_strategy") or "round_robin").strip().lower()
    return strat if strat in ("round_robin", "priority") else "round_robin"


def provider_accounts(provider_cfg: dict) -> list:
    """Return the provider's credentials as an ordered list of :class:`Account`.

    Each account's ``key`` is resolved from ``${VAR}`` refs at call time (exactly
    like :func:`provider_api_key`), while ``key_raw`` preserves the unresolved
    form for admin display/masking. With the ``priority`` strategy the list is
    ordered lowest-priority-first (stable for ties); otherwise it is in declared
    order and callers apply round-robin rotation.

    A single-account provider yields one ``Account`` with ``id=None`` so per-
    account usage/saturation keys collapse to today's ``provider/model`` form.
    """
    entries = _normalize_account_entries(provider_cfg)
    indexed = list(enumerate(entries))  # keep declared index for stable ids/ties
    if provider_account_strategy(provider_cfg) == "priority":
        indexed.sort(key=lambda t: (_account_priority(t[1]), t[0]))
    single = len(indexed) <= 1
    accounts: list = []
    seen: set = set()
    for orig_idx, entry in indexed:
        label = entry.get("label")
        acct_id = None if single else _account_id_from_label(label, orig_idx)
        if acct_id is not None and acct_id in seen:  # guard duplicate labels
            acct_id = f"{acct_id}_{orig_idx}"
        seen.add(acct_id)
        accounts.append(Account(
            id=acct_id,
            key=resolve_env_refs(entry.get("key")) or "",
            key_raw=entry.get("key"),
            label=(label or "") if acct_id is None else (label or acct_id),
            priority=_account_priority(entry),
        ))
    return accounts


def account_bound_cfg(provider_cfg: dict, account: Account) -> dict:
    """Return a shallow copy of *provider_cfg* bound to *account*'s credential.

    The account's (unresolved) key is written into ``api_key`` and its id into
    ``_account_id`` so that (a) every existing ``provider_api_key(cfg)`` call
    site transparently uses the chosen account's key with **no signature
    change**, and (b) usage/saturation recording can recover the account id. The
    copy is runtime-only and never persisted.
    """
    bound = dict(provider_cfg)
    bound["api_key"] = account.key_raw
    bound["_account_id"] = account.id
    return bound


def provider_account_id(provider_cfg: dict):
    """Return the bound account id on a runtime cfg copy (``None`` if unbound)."""
    return provider_cfg.get("_account_id")


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

    normalized = copy.deepcopy(raw)

    # Migrate free_tier.probe → free_tier.cost_probe (renamed in this release).
    ft = normalized.get("free_tier")
    if isinstance(ft, dict) and "probe" in ft and "cost_probe" not in ft:
        ft["cost_probe"] = ft.pop("probe")

    # Migrate free_tier.endpoint_probe.{timeout_sec} → free_tier.probe_timeout_sec.
    # The endpoint_probe block was flattened once it held nothing but a timeout,
    # and that timeout is now shared with the cost probe. frequency_minutes is
    # dropped rather than migrated: the endpoint probe no longer throttles
    # itself, so there is no new key for it to become.
    if isinstance(ft, dict) and isinstance(ft.get("endpoint_probe"), dict):
        ep = ft["endpoint_probe"]
        if "timeout_sec" in ep and "probe_timeout_sec" not in ft:
            ft["probe_timeout_sec"] = ep["timeout_sec"]
        ep.pop("timeout_sec", None)
        ep.pop("frequency_minutes", None)
        if not ep:
            ft.pop("endpoint_probe")

    if not any(k in normalized for k in _LEGACY_KEY_MIGRATIONS):
        return normalized
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
