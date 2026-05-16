"""
setup_wizard.py — Interactive terminal wizard for configuring llmproxy.

Run via:  llmproxy --setup
or:       python -m llmproxy --setup

In Docker:  docker run -it --rm -v llmproxy_config:/root/.config/llmproxy llmproxy --setup
"""

import getpass
import json
import sys
import traceback
from typing import Optional

import requests as _requests

from .config import (
    DEFAULT_SERVER_CONFIG,
    RESERVED_PROVIDER_NAMES,
    get_config_path,
    load_config,
    save_config,
)

# ---------------------------------------------------------------------------
# Provider templates
# ---------------------------------------------------------------------------

# Each entry is a dict with keys:
#   display           – human-readable label shown in the menu
#   key               – default provider name / model-ID prefix
#   base_url          – upstream base URL; may contain "{account_id}" as a placeholder
#   account_id_required – (optional) True when the URL contains "{account_id}"
#   account_id_label  – (optional) prompt label for the account ID (default "Account ID")
#   account_id_hint   – (optional) one-line hint for finding the account ID
PROVIDER_TEMPLATES: list[dict] = [
    {
        "display": "Nous Research (Hermes)",
        "key": "nous",
        "base_url": "https://inference-api.nousresearch.com/v1",
    },
    {
        "display": "Nvidia NIM",
        "key": "nvidia",
        "base_url": "https://integrate.api.nvidia.com/v1",
    },
    {
        "display": "Google Gemini (via OpenAI-compat endpoint)",
        "key": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    },
    {
        "display": "Cerebras",
        "key": "cerebras",
        "base_url": "https://api.cerebras.ai/v1",
    },
    {
        "display": "GitHub Models",
        "key": "github",
        "base_url": "https://models.inference.ai.azure.com",
    },
    {
        "display": "SambaNova Cloud",
        "key": "sambanova",
        "base_url": "https://api.sambanova.ai/v1",
    },
    {
        "display": "Mistral AI",
        "key": "mistral",
        "base_url": "https://api.mistral.ai/v1",
    },
    {
        "display": "Groq",
        "key": "groq",
        "base_url": "https://api.groq.com/openai/v1",
    },
    {
        "display": "Cloudflare Workers AI",
        "key": "cloudflare",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "account_id_required": True,
        "account_id_label": "Cloudflare Account ID",
        "account_id_hint": "Find your account ID at dash.cloudflare.com (top-right corner)",
    },
    {
        "display": "Zhipu AI (Z.ai / BigModel)",
        "key": "zhipu",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    {
        "display": "Cohere",
        "key": "cohere",
        "base_url": "https://api.cohere.com/compatibility/v1",
    },
    {
        "display": "OpenRouter",
        "key": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
    },
    {
        "display": "Ollama Cloud",
        "key": "ollama-cloud",
        "base_url": "https://ollama.com/v1",
    },
]

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
DIM     = "\033[2m"


def _h(text: str) -> str:
    return f"{BOLD}{CYAN}{text}{RESET}"


def _ok(text: str) -> str:
    return f"{GREEN}{text}{RESET}"


def _warn(text: str) -> str:
    return f"{YELLOW}{text}{RESET}"


def _err(text: str) -> str:
    return f"{RED}{text}{RESET}"


def _dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


def _mask_api_key(key: str) -> str:
    return key[:8] + "..." + key[-4:] if len(key) > 12 else "****"


def _prompt(label: str, default: Optional[str] = None, secret: bool = False) -> str:
    """
    Display a prompt and return stripped user input.

    If *secret* is True, use getpass so the value is not echoed.
    If the user presses Enter without input and *default* is provided,
    the default value is returned.
    """
    hint = f" [{_dim(default)}]" if default is not None else ""
    full_label = f"  {label}{hint}: "
    while True:
        try:
            if secret:
                value = getpass.getpass(full_label)
            else:
                value = input(full_label)
        except (EOFError, KeyboardInterrupt):
            print()
            return default or ""
        value = value.strip()
        if value:
            return value
        if default is not None:
            return default
        print(_warn("  This field is required. Please enter a value."))


def _confirm(question: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer, returning a bool."""
    yn = "Y/n" if default else "y/N"
    try:
        answer = input(f"  {question} [{yn}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer.startswith("y")


def _pick(options: list[str], label: str = "Choice") -> Optional[int]:
    """
    Present a numbered menu and return the 0-based index of the selection,
    or None if the user cancels.
    """
    for i, opt in enumerate(options, 1):
        print(f"    {_bold_num(i)}. {opt}")
    try:
        raw = input(f"  {label} (or Enter to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not raw:
        return None
    try:
        idx = int(raw)
        if 1 <= idx <= len(options):
            return idx - 1
    except ValueError:
        pass
    print(_warn("  Invalid selection."))
    return None


def _bold_num(n: int) -> str:
    return f"{BOLD}{n}{RESET}"


def _divider(char: str = "─", width: int = 60) -> str:
    return _dim(char * width)


# ---------------------------------------------------------------------------
# Provider editing
# ---------------------------------------------------------------------------

def _edit_provider(name: str, existing: Optional[dict] = None) -> dict:
    """
    Interactively collect or update a provider configuration.

    Parameters
    ----------
    name : str
        The provider's key/prefix name.
    existing : dict, optional
        Pre-existing values to show as defaults.

    Returns
    -------
    dict
        Provider configuration with keys: base_url, api_key, model_filter.
    """
    ex = existing or {}
    print()
    print(_h(f"  Provider: {name}"))
    print()

    base_url = _prompt(
        "Base URL",
        default=ex.get("base_url", "https://api.openai.com/v1"),
    )

    # API key: optional (e.g. Ollama does not require one)
    existing_key = ex.get("api_key", "")
    print()
    print(_dim("  API Key: required for cloud providers (OpenAI, OpenRouter, etc.)."))
    print(_dim("  Leave blank for providers that do not need authentication (e.g. Ollama)."))
    if existing_key:
        masked = _mask_api_key(existing_key)
        print(f"  API Key  [{_dim(masked)}] (press Enter to keep, or type new key):")
        try:
            new_key = getpass.getpass("  API Key (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            new_key = ""
        api_key = new_key if new_key else existing_key
    else:
        print(f"  API Key (optional): ", end="", flush=True)
        try:
            api_key = getpass.getpass("").strip()
        except (EOFError, KeyboardInterrupt):
            api_key = ""

    # Model filter
    existing_filter = ex.get("model_filter")
    current_filter_str = ", ".join(existing_filter) if existing_filter else ""
    print()
    print(_dim("  Model filter: comma-separated list of upstream model IDs to allow."))
    print(_dim("  Leave blank to permit all models from this provider."))
    filter_raw = _prompt(
        "Model filter",
        default=current_filter_str if current_filter_str else None or "",
    )
    if filter_raw.strip():
        model_filter = [m.strip() for m in filter_raw.split(",") if m.strip()]
    else:
        model_filter = None

    return {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model_filter": model_filter,
    }


# ---------------------------------------------------------------------------
# Server settings
# ---------------------------------------------------------------------------

def _edit_server(existing: dict) -> dict:
    """Interactively update server-level settings."""
    print()
    print(_h("  Server Settings"))
    print()

    host = _prompt("Bind host", default=existing.get("host", DEFAULT_SERVER_CONFIG["host"]))
    port_str = _prompt("Port", default=str(existing.get("port", DEFAULT_SERVER_CONFIG["port"])))
    try:
        port = int(port_str)
    except ValueError:
        print(_warn("  Invalid port; using 8080."))
        port = 8080

    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    current_level = existing.get("log_level", "INFO")
    print()
    print(f"  Log level (current: {_ok(current_level)}):")
    for i, lvl in enumerate(log_levels, 1):
        marker = " <--" if lvl == current_level else ""
        print(f"    {_bold_num(i)}. {lvl}{_dim(marker)}")
    level_raw = input("  Choice (Enter to keep current): ").strip()
    if level_raw.isdigit() and 1 <= int(level_raw) <= len(log_levels):
        log_level = log_levels[int(level_raw) - 1]
    else:
        log_level = current_level

    req_timeout = _prompt(
        "Request timeout (seconds, non-streaming)",
        default=str(existing.get("request_timeout", DEFAULT_SERVER_CONFIG["request_timeout"])),
    )
    stream_timeout = _prompt(
        "Stream timeout (seconds)",
        default=str(existing.get("stream_timeout", DEFAULT_SERVER_CONFIG["stream_timeout"])),
    )

    try:
        req_timeout = int(req_timeout)
    except ValueError:
        req_timeout = DEFAULT_SERVER_CONFIG["request_timeout"]
    try:
        stream_timeout = int(stream_timeout)
    except ValueError:
        stream_timeout = DEFAULT_SERVER_CONFIG["stream_timeout"]

    return {
        "host": host,
        "port": port,
        "log_level": log_level,
        "request_timeout": req_timeout,
        "stream_timeout": stream_timeout,
    }


# ---------------------------------------------------------------------------
# Template-based provider setup
# ---------------------------------------------------------------------------

def _setup_from_template(providers: dict) -> tuple[str, dict] | None:
    """
    Interactively guide the user through adding a provider from a template.

    Returns (provider_key, provider_cfg) on success, or None if the user
    cancels.
    """
    print()
    print(_h("  Quick setup — choose a provider template:"))
    print()
    names = [t["display"] for t in PROVIDER_TEMPLATES]
    idx = _pick(names, label="Template")
    if idx is None:
        return None

    tmpl = PROVIDER_TEMPLATES[idx]
    print()
    print(_h(f"  Template: {tmpl['display']}"))
    print()
    print(_dim(f"  Base URL : {tmpl['base_url']}"))
    print()

    # Confirm provider name first so we can look up any existing key below.
    provider_key = _prompt("Provider name (used as prefix in model IDs)", default=tmpl["key"])
    if provider_key in RESERVED_PROVIDER_NAMES:
        print(_warn(f"  {provider_key!r} is a reserved provider namespace. Choose a different name."))
        return None

    existing = providers.get(provider_key)
    if existing:
        print(_warn(f"\n  Provider '{provider_key}' already exists. It will be overwritten."))
        if not _confirm("Overwrite?", default=False):
            return None

    base_url = tmpl["base_url"]

    # Some templates require an account ID to be substituted into the URL.
    if tmpl.get("account_id_required"):
        acct_hint = tmpl.get("account_id_hint", "")
        acct_label = tmpl.get("account_id_label", "Account ID")
        print()
        print(_dim("  The base URL contains a placeholder for an account ID."))
        if acct_hint:
            print(_dim(f"  {acct_hint}"))
        account_id = _prompt(acct_label)
        base_url = base_url.replace("{account_id}", account_id)
        print(_dim(f"  Resolved URL: {base_url}"))
        print()

    # API key — preserve the existing key if the user submits an empty value.
    existing_key = (existing or {}).get("api_key", "")
    if existing_key:
        masked = _mask_api_key(existing_key)
        print(f"  API Key [{_dim(masked)}] (press Enter to keep): ", end="", flush=True)
    else:
        print(f"  API Key (optional): ", end="", flush=True)
    try:
        new_key = getpass.getpass("").strip()
    except (EOFError, KeyboardInterrupt):
        new_key = ""
    api_key = new_key if new_key else existing_key

    cfg = {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model_filter": None,
    }
    return provider_key, cfg


# ---------------------------------------------------------------------------
# Scrollable model picker (queries providers directly)
# ---------------------------------------------------------------------------

_PAGE_SIZE = 20
_REASONING_LEVELS = ("exploratory", "standard", "deep")


def _fetch_provider_models_direct(providers: dict) -> list[tuple[str, str]]:
    """
    Query every configured provider's /models endpoint directly and return a
    flat list of (provider_name, upstream_model_id) tuples.  Providers that
    are unreachable are skipped with a warning.
    """
    results: list[tuple[str, str]] = []
    for provider_name, provider_cfg in providers.items():
        base_url = provider_cfg.get("base_url", "").rstrip("/")
        api_key = provider_cfg.get("api_key", "")
        model_filter = provider_cfg.get("model_filter")
        try:
            resp = _requests.get(
                f"{base_url}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            for m in resp.json().get("data", []):
                uid = m.get("id", "")
                if uid and (model_filter is None or uid in model_filter):
                    results.append((provider_name, uid))
        except Exception as exc:
            print(_warn(f"  Could not reach '{provider_name}': {exc}"))
    return results


def _pick_model_scrollable(
    providers: dict,
    prompt: str = "Select a model",
    exclude: Optional[set] = None,
) -> Optional[str]:
    """
    Fetch models from all configured providers and show a paginated, filterable
    numbered list.

    Returns the selected entry as "provider_name/upstream_model_id", or None if
    the user cancels.

    Navigation:
      <number>   — select that entry
      n / p      — next / previous page
      <text>     — narrow the list to entries whose provider or model ID contains
                   the text (re-type or press Enter on empty to reset)
      Enter      — cancel (when input is blank and no filter is active)
    """
    exclude = exclude or set()

    print()
    print(_dim("  Fetching models from configured providers…"))
    raw_models = _fetch_provider_models_direct(providers)
    if not raw_models:
        print(_warn("  No models found. Check that your providers are reachable."))
        return None

    # Remove already-listed entries so the picker shows only new additions.
    all_models = [(pn, uid) for pn, uid in raw_models
                  if f"{pn}/{uid}" not in exclude and uid not in exclude]

    if not all_models:
        print(_warn("  All models are already in the list."))
        return None

    active = list(all_models)
    filter_str = ""
    page = 0

    while True:
        total = len(active)
        total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
        page = max(0, min(page, total_pages - 1))
        start = page * _PAGE_SIZE
        page_slice = active[start: start + _PAGE_SIZE]

        print()
        header = f"  {prompt}"
        if filter_str:
            header += f"  {_dim(f'[filter: {filter_str!r}]')}"
        header += f"  {_dim(f'({total} entries, page {page+1}/{total_pages})')}"
        print(_h(header))
        print()

        for i, (pn, uid) in enumerate(page_slice, start + 1):
            print(f"    {_bold_num(i):>4}.  {pn}/{uid}")

        print()
        nav_hints = []
        if page < total_pages - 1:
            nav_hints.append("n=next")
        if page > 0:
            nav_hints.append("p=prev")
        if filter_str:
            nav_hints.append("r=reset filter")
        nav = ("  [" + "  " .join(nav_hints) + "]  ") if nav_hints else "  "

        try:
            raw = input(f"{nav}Number, text to filter, or Enter to cancel: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if not raw:
            if filter_str:
                # Reset filter
                active = list(all_models)
                filter_str = ""
                page = 0
                continue
            return None

        low = raw.lower()

        if low == "n" and page < total_pages - 1:
            page += 1
            continue
        if low == "p" and page > 0:
            page -= 1
            continue
        if low == "r":
            active = list(all_models)
            filter_str = ""
            page = 0
            continue

        # Numeric selection
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(active):
                pn, uid = active[idx - 1]
                chosen = f"{pn}/{uid}"
                print(_ok(f"  Selected: {chosen}"))
                return chosen
            print(_warn(f"  Number out of range (1–{len(active)})."))
            continue

        # Text filter / narrow
        narrowed = [(pn, uid) for pn, uid in active
                    if low in uid.lower() or low in pn.lower()]
        if not narrowed:
            print(_warn(f"  No entries match '{raw}'."))
            continue
        if len(narrowed) == 1:
            pn, uid = narrowed[0]
            chosen = f"{pn}/{uid}"
            print(_ok(f"  Selected: {chosen}"))
            return chosen
        # Narrow the list
        active = narrowed
        filter_str = raw
        page = 0


# ---------------------------------------------------------------------------
# Model-tags menu (known_free + model_reasoning)
# ---------------------------------------------------------------------------

def _edit_model_tags(config: dict, providers: dict) -> bool:
    """
    Interactive sub-menu for managing known_free and model_reasoning.
    Returns True if the config was modified.
    """
    modified = False

    while True:
        known_free: list = config.setdefault("known_free", [])
        reasoning: dict = config.setdefault("model_reasoning", {})

        print()
        print(_divider())
        print(_h("  Model Tags"))
        print()
        print(f"  {_dim('known_free:')}     {len(known_free)} entr{'y' if len(known_free)==1 else 'ies'}")
        print(f"  {_dim('model_reasoning:')} {len(reasoning)} entr{'y' if len(reasoning)==1 else 'ies'}")
        print()

        options = [
            "Add model to known_free",
            "Remove model from known_free",
            "Tag model with reasoning level  (exploratory / standard / deep)",
            "Remove reasoning tag from model",
            "View current tags",
            "Back to main menu",
        ]
        choice = _pick(options, label="Option")
        if choice is None or choice == 5:
            break

        # ── Add to known_free ──────────────────────────────────────────────
        if choice == 0:
            if not providers:
                print(_warn("  No providers configured — add a provider first."))
                continue
            entry = _pick_model_scrollable(
                providers,
                prompt="Add to known_free — pick a model",
                exclude=set(known_free),
            )
            if entry:
                known_free.append(entry)
                config["known_free"] = known_free
                print(_ok(f"  Added '{entry}' to known_free."))
                modified = True

        # ── Remove from known_free ─────────────────────────────────────────
        elif choice == 1:
            if not known_free:
                print(_warn("  known_free is empty."))
                continue
            print()
            print(_h("  Remove from known_free:"))
            idx = _pick(known_free)
            if idx is None:
                continue
            removed = known_free.pop(idx)
            config["known_free"] = known_free
            print(_ok(f"  Removed '{removed}' from known_free."))
            modified = True

        # ── Tag model with reasoning level ────────────────────────────────
        elif choice == 2:
            if not providers:
                print(_warn("  No providers configured — add a provider first."))
                continue
            entry = _pick_model_scrollable(
                providers,
                prompt="Tag with reasoning level — pick a model",
                exclude=set(reasoning.keys()),
            )
            if not entry:
                continue
            print()
            print(f"  Reasoning level for {_ok(entry)}:")
            level_idx = _pick(list(_REASONING_LEVELS), label="Level")
            if level_idx is None:
                continue
            level = _REASONING_LEVELS[level_idx]
            reasoning[entry] = level
            config["model_reasoning"] = reasoning
            print(_ok(f"  Tagged '{entry}' as '{level}'."))
            modified = True

        # ── Remove reasoning tag ──────────────────────────────────────────
        elif choice == 3:
            if not reasoning:
                print(_warn("  model_reasoning is empty."))
                continue
            print()
            print(_h("  Remove reasoning tag:"))
            entries = [f"{k}  →  {v}" for k, v in reasoning.items()]
            idx = _pick(entries)
            if idx is None:
                continue
            key = list(reasoning.keys())[idx]
            del reasoning[key]
            config["model_reasoning"] = reasoning
            print(_ok(f"  Removed reasoning tag for '{key}'."))
            modified = True

        # ── View current tags ─────────────────────────────────────────────
        elif choice == 4:
            print()
            print(_h("  known_free:"))
            if known_free:
                for entry in known_free:
                    print(f"    • {entry}")
            else:
                print(_dim("    (empty)"))
            print()
            print(_h("  model_reasoning:"))
            if reasoning:
                for k, v in reasoning.items():
                    print(f"    {v:<14}  {k}")
            else:
                print(_dim("    (empty)"))
            print()

    return modified


# ---------------------------------------------------------------------------
# Main wizard entry point
# ---------------------------------------------------------------------------

def run_setup(config_path: Optional[str] = None) -> None:
    """
    Launch the interactive setup wizard.

    Loads any existing configuration as defaults, then writes the updated
    configuration back to disk on exit.

    Parameters
    ----------
    config_path : str, optional
        Explicit path override for the config file location.
    """
    resolved_path = get_config_path(config_path)
    config = load_config(config_path, force_reload=True)
    providers: dict = config.setdefault("providers", {})
    server_cfg: dict = config.setdefault("server", dict(DEFAULT_SERVER_CONFIG))

    print()
    print(_h("╔══════════════════════════════════════════╗"))
    print(_h("║          llmproxy  Setup Wizard          ║"))
    print(_h("╚══════════════════════════════════════════╝"))
    print()
    print(f"  Config file: {_ok(str(resolved_path))}")
    print()

    _show_summary(providers, server_cfg)

    modified = False

    while True:
        print()
        print(_divider())
        print(_h("  Main Menu"))
        print()
        options = [
            "Quick setup from template",
            "Add / edit a provider (manual)",
            "Remove a provider",
            "Configure server settings",
            "Manage model tags  (known_free / reasoning levels)",
            "View current configuration (JSON)",
            "Save and exit",
            "Exit without saving",
        ]
        choice = _pick(options, label="Option")
        if choice is None:
            continue

        if choice == 0:
            # Quick setup from template
            result = _setup_from_template(providers)
            if result is not None:
                name, cfg = result
                providers[name] = cfg
                print()
                print(_ok(f"  Provider '{name}' configured from template."))
                modified = True

        elif choice == 1:
            # Add / edit provider (manual)
            print()
            provider_names = list(providers.keys())
            if provider_names:
                print(_h("  Existing providers:"))
                for p in provider_names:
                    print(f"    • {p}")
                print()
            name = _prompt("Provider name (e.g. openrouter, openai, ollama)")
            if not name:
                continue
            if name in RESERVED_PROVIDER_NAMES:
                print(_warn(f"  {name!r} is a reserved provider namespace. Choose a different name."))
                continue
            existing = providers.get(name)
            if existing:
                print(_warn(f"\n  Provider '{name}' already exists. Editing."))
            providers[name] = _edit_provider(name, existing)
            print()
            print(_ok(f"  Provider '{name}' configured."))
            modified = True

        elif choice == 2:
            # Remove provider
            provider_names = list(providers.keys())
            if not provider_names:
                print(_warn("  No providers configured yet."))
                continue
            print()
            print(_h("  Select provider to remove:"))
            idx = _pick(provider_names)
            if idx is None:
                continue
            name = provider_names[idx]
            if _confirm(f"Remove provider '{name}'?", default=False):
                del providers[name]
                print(_ok(f"  Provider '{name}' removed."))
                modified = True

        elif choice == 3:
            # Server settings
            config["server"] = _edit_server(server_cfg)
            server_cfg = config["server"]
            print(_ok("  Server settings updated."))
            modified = True

        elif choice == 4:
            # Model tags
            if _edit_model_tags(config, providers):
                modified = True

        elif choice == 5:
            # View config
            print()
            print(_h("  Current configuration:"))
            print()
            display = _redact_config(config)
            print(json.dumps(display, indent=2))

        elif choice == 6:
            # Save and exit
            if save_config(config, config_path):
                print(_ok("  Configuration saved. Exiting setup."))
            else:
                print(_err("  Failed to save configuration. Check permissions."))
            break

        elif choice == 7:
            # Exit without saving
            if modified and not _confirm("You have unsaved changes. Exit anyway?", default=False):
                continue
            print("  Exiting without saving.")
            break

    print()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _show_summary(providers: dict, server_cfg: dict) -> None:
    if not providers:
        print(_warn("  No providers configured yet."))
    else:
        print(_h("  Configured providers:"))
        for name, cfg in providers.items():
            base = cfg.get("base_url", "(none)")
            key  = cfg.get("api_key", "")
            filt = cfg.get("model_filter")
            key_display = (key[:8] + "...") if key else _warn("(no key)")
            filt_display = ", ".join(filt) if filt else _dim("(all models)")
            print(f"    {BOLD}{name}{RESET}")
            print(f"       base_url:     {base}")
            print(f"       api_key:      {key_display}")
            print(f"       model_filter: {filt_display}")
    print()
    print(_h("  Server:"))
    print(f"    host:      {server_cfg.get('host', '0.0.0.0')}")
    print(f"    port:      {server_cfg.get('port', 8080)}")
    print(f"    log_level: {server_cfg.get('log_level', 'INFO')}")


def _redact_config(config: dict) -> dict:
    """Return a copy of config with API keys masked for display."""
    import copy
    display = copy.deepcopy(config)
    for name, cfg in display.get("providers", {}).items():
        key = cfg.get("api_key", "")
        if key:
            cfg["api_key"] = _mask_api_key(key)
    return display
