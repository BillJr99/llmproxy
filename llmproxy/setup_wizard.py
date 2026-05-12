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

from .config import (
    DEFAULT_SERVER_CONFIG,
    get_config_path,
    load_config,
    save_config,
)

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
        masked = existing_key[:8] + "..." + existing_key[-4:] if len(existing_key) > 12 else "****"
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
            api_key = 

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
            "Add / edit a provider",
            "Remove a provider",
            "Configure server settings",
            "View current configuration (JSON)",
            "Save and exit",
            "Exit without saving",
        ]
        choice = _pick(options, label="Option")
        if choice is None:
            continue

        if choice == 0:
            # Add / edit provider
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
            existing = providers.get(name)
            if existing:
                print(_warn(f"\n  Provider '{name}' already exists. Editing."))
            providers[name] = _edit_provider(name, existing)
            print()
            print(_ok(f"  Provider '{name}' configured."))
            modified = True

        elif choice == 1:
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

        elif choice == 2:
            # Server settings
            config["server"] = _edit_server(server_cfg)
            server_cfg = config["server"]
            print(_ok("  Server settings updated."))
            modified = True

        elif choice == 3:
            # View config
            print()
            print(_h("  Current configuration:"))
            print()
            display = _redact_config(config)
            print(json.dumps(display, indent=2))

        elif choice == 4:
            # Save and exit
            if save_config(config, config_path):
                print(_ok("  Configuration saved. Exiting setup."))
            else:
                print(_err("  Failed to save configuration. Check permissions."))
            break

        elif choice == 5:
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
            cfg["api_key"] = key[:8] + "..." + key[-4:] if len(key) > 12 else "****"
    return display
