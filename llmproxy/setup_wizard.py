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
# Provider templates
# ---------------------------------------------------------------------------

# Each entry: (display_name, provider_key, base_url, model_filter, api_key_hint, notes)
# base_url may contain the placeholder "{account_id}" — the wizard will substitute it.
PROVIDER_TEMPLATES: list[dict] = [
    {
        "display": "Nous Research (Hermes)",
        "key": "nous",
        "base_url": "https://inference-api.nousresearch.com/v1",
        "model_filter": ["Hermes-3-Llama-3.1-8B"],
        "api_key_hint": "Get free key at nousresearch.com",
    },
    {
        "display": "Nvidia NIM",
        "key": "nvidia",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model_filter": None,
        "api_key_hint": "Get free key at build.nvidia.com",
    },
    {
        "display": "Google Gemini (via OpenAI-compat endpoint)",
        "key": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model_filter": [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3.1-pro-preview",
        ],
        "api_key_hint": "Free key from Google AI Studio (aistudio.google.com) — free Google account, no credit card",
    },
    {
        "display": "Cerebras",
        "key": "cerebras",
        "base_url": "https://api.cerebras.ai/v1",
        "model_filter": ["qwen3-235b"],
        "api_key_hint": "Free account at cerebras.ai, no credit card",
    },
    {
        "display": "GitHub Models",
        "key": "github",
        "base_url": "https://models.inference.ai.azure.com",
        "model_filter": ["gpt-4o", "openai/gpt-4.1"],
        "api_key_hint": "GitHub Personal Access Token (any free GitHub account, no special scopes needed)",
    },
    {
        "display": "SambaNova Cloud",
        "key": "sambanova",
        "base_url": "https://api.sambanova.ai/v1",
        "model_filter": [
            "Meta-Llama-3.3-70B-Instruct",
            "DeepSeek-V3.1",
            "DeepSeek-V3.2",
            "gpt-oss-120b",
            "Llama-4-Maverick-17B-128E-Instruct",
            "gemma-3-12b-it",
        ],
        "api_key_hint": "Free account at cloud.sambanova.ai, no credit card",
    },
    {
        "display": "Mistral AI",
        "key": "mistral",
        "base_url": "https://api.mistral.ai/v1",
        "model_filter": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "magistral-medium-latest",
            "codestral-latest",
            "devstral-latest",
        ],
        "api_key_hint": 'Free "Experiment" tier account at console.mistral.ai, no credit card',
    },
    {
        "display": "Groq",
        "key": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model_filter": [
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
            "llama-3.1-8b-instant",
        ],
        "api_key_hint": "Free account at console.groq.com, no credit card",
    },
    {
        "display": "Cloudflare Workers AI",
        "key": "cloudflare",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "model_filter": [
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/meta/llama-4-scout-17b-16e-instruct",
            "@cf/openai/gpt-oss-120b",
            "@cf/zai-org/glm-4.7-flash",
            "@cf/moonshotai/kimi-k2.5",
            "@cf/moonshotai/kimi-k2.6",
            "@cf/qwen/qwen3-30b-a3b-fp8",
            "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
            "@cf/ibm-granite/granite-4.0-h-micro",
        ],
        "api_key_hint": 'Free Cloudflare account — create API Token with "Workers AI Read" permission',
        "account_id_required": True,
    },
    {
        "display": "Zhipu AI (Z.ai / BigModel)",
        "key": "zhipu",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model_filter": ["glm-4.5-flash", "glm-4.7-flash"],
        "api_key_hint": "Free account at open.bigmodel.cn, no credit card",
    },
    {
        "display": "Cohere",
        "key": "cohere",
        "base_url": "https://api.cohere.com/compatibility/v1",
        "model_filter": None,
        "api_key_hint": "Get key at dashboard.cohere.com",
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
    mf = tmpl["model_filter"]
    if mf:
        print(_dim(f"  Models   : {', '.join(mf)}"))
    else:
        print(_dim("  Models   : (all models)"))
    print()

    base_url = tmpl["base_url"]

    # Cloudflare (and any future template) requires account ID substitution
    if tmpl.get("account_id_required"):
        print(_dim("  The base URL contains a placeholder for your account ID."))
        print(_dim("  Find your account ID at dash.cloudflare.com (top-right corner)."))
        account_id = _prompt("Cloudflare Account ID")
        base_url = base_url.replace("{account_id}", account_id)
        print(_dim(f"  Resolved URL: {base_url}"))
        print()

    # API key
    hint = tmpl.get("api_key_hint", "")
    if hint:
        print(_dim(f"  API key: {hint}"))
    print(f"  API Key (optional): ", end="", flush=True)
    try:
        api_key = getpass.getpass("").strip()
    except (EOFError, KeyboardInterrupt):
        api_key = ""

    # Allow the user to override the default provider key
    print()
    provider_key = _prompt("Provider name (used as prefix in model IDs)", default=tmpl["key"])

    if provider_key in providers:
        print(_warn(f"\n  Provider '{provider_key}' already exists. It will be overwritten."))
        if not _confirm("Overwrite?", default=False):
            return None

    cfg = {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model_filter": tmpl["model_filter"],
    }
    return provider_key, cfg


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
            # View config
            print()
            print(_h("  Current configuration:"))
            print()
            display = _redact_config(config)
            print(json.dumps(display, indent=2))

        elif choice == 5:
            # Save and exit
            if save_config(config, config_path):
                print(_ok("  Configuration saved. Exiting setup."))
            else:
                print(_err("  Failed to save configuration. Check permissions."))
            break

        elif choice == 6:
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
