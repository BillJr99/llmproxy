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
      "model_filter": ["model-a", "model-b"]   // null or absent = allow all
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
      "tokens_per_minute": null,                  // token limits stored for reference
      "tokens_per_day": null                      // but not yet enforced (no token
    }                                             // counting without response parsing)
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "log_level": "INFO",
    "request_timeout": 120,
    "stream_timeout": 300
  }
}
"""

import json
import os
import traceback
from pathlib import Path

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

DEFAULT_CONFIG: dict = {
    "providers": {},
    "believed_free": [],
    "model_reasoning": {},
    "model_capabilities": {},
    "free_limits": {},
    "server": dict(DEFAULT_SERVER_CONFIG),
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

        merged = _deep_merge(DEFAULT_CONFIG, raw)
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
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
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
# Internal utilities
# ---------------------------------------------------------------------------

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
