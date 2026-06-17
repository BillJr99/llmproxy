#!/usr/bin/env python3
"""
test_tui.py — Interactive chat TUI for llmproxy.

Connects to an llmproxy server and provides a conversational interface with
streaming responses.  Supports model switching including all virtual endpoints
(llmproxy/free, llmproxy/local, llmproxy/exploratory, llmproxy/standard,
llmproxy/deep, and their free/local combinations).  The earlier "llmproxy__..."
forms are also accepted as input.

Usage
-----
  python test_tui.py
  python test_tui.py --base-url http://localhost:8080/v1
  python test_tui.py --model llmproxy/standard --system "You are a concise assistant."

Commands (type inside the chat)
--------------------------------
  /model [filter]   browse/pick a model; optional filter narrows the list
  /models [filter]  alias for /model
  /clear            clear conversation history (keeps system prompt)
  /history          print the current conversation so far
  /system [text]    show or replace the system prompt
  /url <url>        switch to a different proxy base URL
  /help             show this help
  /quit  /exit  /q  exit   (or press Ctrl-D)

Press Ctrl-C during a response to cancel the stream without exiting.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from typing import Optional

import requests

try:
    import readline  # noqa: F401 — imported for side-effect (line editing / history)
    _HAS_READLINE = True
except ImportError:
    _HAS_READLINE = False

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
CYAN    = "\033[36m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
RED     = "\033[31m"

_W = 72   # display width

_VIRTUAL_IDS = frozenset({
    # Advertised "llmproxy/..." form (current /v1/models output)
    "llmproxy/free", "llmproxy/local",
    "llmproxy/exploratory", "llmproxy/standard", "llmproxy/deep",
    "llmproxy/exploratory__free", "llmproxy/exploratory__local",
    "llmproxy/standard__free", "llmproxy/standard__local",
    "llmproxy/deep__free", "llmproxy/deep__local",
    # Earlier "llmproxy__..." form (still accepted as input)
    "llmproxy__free", "llmproxy__local",
    "llmproxy__exploratory", "llmproxy__standard", "llmproxy__deep",
    "llmproxy__exploratory/free", "llmproxy__exploratory/local",
    "llmproxy__standard/free", "llmproxy__standard/local",
    "llmproxy__deep/free", "llmproxy__deep/local",
    # Legacy "llmproxy/<name>/<dimension>" slash form (still accepted as input)
    "llmproxy/exploratory/free", "llmproxy/exploratory/local",
    "llmproxy/standard/free", "llmproxy/standard/local",
    "llmproxy/deep/free", "llmproxy/deep/local",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "━") -> str:
    return char * _W


def _print_banner(base_url: str, model: Optional[str]) -> None:
    m = f"{CYAN}{BOLD}{model}{RESET}" if model else f"{YELLOW}(none — type /model to choose){RESET}"
    print(f"\n{BOLD}{CYAN}{_hr()}{RESET}")
    print(f"{BOLD}  llmproxy TUI{RESET}")
    print(f"  Server : {DIM}{base_url}{RESET}")
    print(f"  Model  : {m}")
    print(f"  {DIM}/model  /clear  /history  /system  /url  /help  /quit{RESET}")
    print(f"{BOLD}{CYAN}{_hr()}{RESET}\n")


def _fetch_models(base_url: str) -> Optional[list[dict]]:
    """Return model list on success, or None if the server is unreachable."""
    try:
        resp = requests.get(f"{base_url}/models", timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except requests.exceptions.ConnectionError:
        print(f"{RED}Cannot connect to {base_url} — is llmproxy running?{RESET}")
        return None
    except Exception as e:
        print(f"{YELLOW}Warning: could not fetch models: {e}{RESET}")
        return None


def _auto_pick(models: list[dict]) -> Optional[str]:
    """Pick a sensible default model from the list."""
    ids = {m["id"] for m in models}
    for pref in (
        # Advertised form (current /v1/models output).
        "llmproxy/standard", "llmproxy/free", "llmproxy/local",
        "llmproxy/exploratory", "llmproxy/deep",
        # Earlier "__" form, in case the server hasn't been upgraded yet.
        "llmproxy__standard", "llmproxy__free", "llmproxy__local",
        "llmproxy__exploratory", "llmproxy__deep",
    ):
        if pref in ids:
            return pref
    if models:
        return models[0]["id"]
    return None


# ---------------------------------------------------------------------------
# Model picker
# ---------------------------------------------------------------------------

def _pick_model(models: list[dict], current: Optional[str], initial_filter: str = "") -> Optional[str]:
    """
    Interactive model picker.  Displays a numbered list grouped by virtual /
    provider, lets the user type a number or a partial name to select.
    Typing additional text narrows the visible list.  Enter with no input
    keeps the current model.
    """
    if not models:
        print(f"{YELLOW}No models available from the proxy.{RESET}")
        return current

    def _display(filter_str: str) -> list[dict]:
        """Build the ordered, filtered display list."""
        f = filter_str.lower()
        ordered: list[dict] = []
        # Virtual models first
        for m in models:
            if m["id"] in _VIRTUAL_IDS:
                if not f or f in m["id"].lower():
                    ordered.append(m)
        # Real models grouped by provider
        by_prov: dict[str, list[dict]] = defaultdict(list)
        for m in models:
            if m["id"] not in _VIRTUAL_IDS:
                prov = m.get("_provider") or (m["id"].split("/")[0] if "/" in m["id"] else "?")
                if not f or f in m["id"].lower() or f in prov.lower():
                    by_prov[prov].append(m)
        for prov in sorted(by_prov):
            ordered.extend(by_prov[prov])
        return ordered

    filter_str = initial_filter
    while True:
        display = _display(filter_str)
        print(f"\n{BOLD}Models{' — filter: ' + repr(filter_str) if filter_str else ''}{RESET}")
        print(f"{DIM}{_hr('─')}{RESET}")

        if not display:
            print(f"  {YELLOW}No models match '{filter_str}'.{RESET}")
        else:
            # Print with section headers
            in_virtual = False
            cur_prov = None
            for idx, m in enumerate(display, 1):
                mid = m["id"]
                if mid in _VIRTUAL_IDS:
                    if not in_virtual:
                        print(f"\n  {BOLD}{CYAN}Virtual (cycling){RESET}")
                        in_virtual = True
                else:
                    prov = m.get("_provider") or (mid.split("/")[0] if "/" in mid else "?")
                    if prov != cur_prov:
                        print(f"\n  {BOLD}{BLUE}{prov}{RESET}")
                        cur_prov = prov
                    in_virtual = False

                marker = f"{GREEN}▸{RESET} " if mid == current else "  "
                if mid in _VIRTUAL_IDS:
                    # For virtual models show a short description from _note
                    note = m.get("_note", "")
                    short_note = note.split(":")[1].strip()[:55] if ": " in note else ""
                    suffix = f"  {DIM}{short_note}{RESET}" if short_note else ""
                    label = f"{CYAN}{mid}{RESET}{suffix}"
                else:
                    # Strip provider prefix for real models to save space
                    prov = m.get("_provider") or (mid.split("/")[0] if "/" in mid else "?")
                    label = mid[len(prov) + 1:] if mid.startswith(prov + "/") else mid
                print(f"  {marker}{idx:>3}.  {label}")

        print(f"\n{DIM}{_hr('─')}{RESET}")
        hint = f"current: {CYAN}{current}{RESET}" if current else "no model selected"
        print(
            f"Enter number, partial name to filter, or blank to keep ({hint}): ",
            end="", flush=True,
        )

        try:
            raw = input().strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return current

        if not raw:
            return current

        # Try numeric selection
        try:
            n = int(raw)
            if display and 1 <= n <= len(display):
                chosen = display[n - 1]["id"]
                print(f"{GREEN}✓ Selected: {BOLD}{chosen}{RESET}\n")
                return chosen
            print(f"{YELLOW}Number {n} out of range (1–{len(display)}).{RESET}")
            continue
        except ValueError:
            pass

        # Try exact match first, then partial
        exact = [m for m in display if m["id"].lower() == raw.lower()]
        if exact:
            chosen = exact[0]["id"]
            print(f"{GREEN}✓ Selected: {BOLD}{chosen}{RESET}\n")
            return chosen

        partial = [m for m in display if raw.lower() in m["id"].lower()]
        if len(partial) == 1:
            chosen = partial[0]["id"]
            print(f"{GREEN}✓ Selected: {BOLD}{chosen}{RESET}\n")
            return chosen
        if len(partial) > 1:
            # Narrow the filter
            filter_str = raw
            continue

        # No match — treat as a new filter
        filter_str = raw


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------

def _stream_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: int = 120,
) -> str:
    """
    Send a streaming chat completion and print tokens as they arrive.
    Returns the full assembled response text.
    Ctrl-C cancels the stream without exiting the program.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    parts: list[str] = []

    print(f"\n{BOLD}{BLUE}Assistant{RESET}  {DIM}({model}){RESET}\n", flush=True)

    try:
        with requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                try:
                    err = resp.json().get("error", {}).get("message", resp.text[:300])
                except Exception:
                    err = resp.text[:300]
                print(f"{RED}Error {resp.status_code}: {err}{RESET}\n")
                return ""

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content") or delta.get("reasoning_content") or ""
                if token:
                    parts.append(token)
                    print(token, end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}[stream cancelled]{RESET}", end="", flush=True)
    except requests.exceptions.Timeout:
        print(f"\n{YELLOW}[timed out after {timeout}s]{RESET}", end="")
    except Exception as e:
        print(f"\n{RED}[error: {e}]{RESET}", end="")

    print("\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_help() -> None:
    print(f"""
{BOLD}Commands:{RESET}
  {CYAN}/model [filter]{RESET}    browse and pick a model; optional text pre-filters the list
  {CYAN}/models [filter]{RESET}   alias for /model
  {CYAN}/clear{RESET}             clear conversation history (keep system prompt)
  {CYAN}/history{RESET}           print conversation so far
  {CYAN}/system [text]{RESET}     show or replace the system prompt
  {CYAN}/url <url>{RESET}         switch to a different proxy base URL
  {CYAN}/help{RESET}              show this help
  {CYAN}/quit{RESET}  {CYAN}/exit{RESET}  {CYAN}/q{RESET}    exit   (or press Ctrl-D)

{DIM}Press Ctrl-C during a response to cancel without exiting.{RESET}
""")


def _cmd_history(conversation: list[dict]) -> None:
    if not conversation:
        print(f"{DIM}(no conversation history yet){RESET}\n")
        return
    print()
    for msg in conversation:
        role, content = msg["role"], msg["content"]
        if role == "user":
            label = f"{BOLD}{GREEN}You{RESET}"
        else:
            label = f"{BOLD}{BLUE}Assistant{RESET}"
        # Wrap long content
        preview = content[:400] + ("…" if len(content) > 400 else "")
        print(f"  {label}: {preview}")
    print()


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="test_tui",
        description="Interactive chat TUI for llmproxy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        metavar="URL",
        help="Proxy base URL (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Starting model (default: auto-select from proxy)",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        metavar="TEXT",
        help="Initial system prompt.",
    )
    parser.add_argument(
        "--timeout",
        default=120,
        type=int,
        metavar="SECS",
        help="Streaming request timeout in seconds (default: 120)",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    system_prompt: str = args.system
    timeout: int = args.timeout
    conversation: list[dict] = []

    # Startup: fetch model list
    print(f"{DIM}Connecting to {base_url}…{RESET}", end="", flush=True)
    models: list[dict] = _fetch_models(base_url) or []
    print(f"\r{' ' * (_W)}\r", end="", flush=True)

    model: Optional[str] = args.model or _auto_pick(models)

    _print_banner(base_url, model)

    if not models:
        print(f"{YELLOW}No models found. The proxy may not be running or no providers are configured.{RESET}\n")

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        m_display = f"{CYAN}{model}{RESET}" if model else f"{YELLOW}?{RESET}"
        prompt_str = f"{BOLD}{GREEN}You{RESET} [{m_display}]: "

        try:
            user_input = input(prompt_str)
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────

        low = user_input.lower()

        if low in ("/quit", "/exit", "/q"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        if low in ("/help", "/?"):
            _cmd_help()
            continue

        if low.startswith(("/model", "/models")):
            # Optional inline filter after the command word
            parts = user_input.split(None, 1)
            inline_filter = parts[1] if len(parts) > 1 else ""
            chosen = _pick_model(models, model, inline_filter)
            if chosen and chosen != model:
                model = chosen
            continue

        if low == "/clear":
            conversation = []
            print(f"{DIM}Conversation history cleared.{RESET}\n")
            continue

        if low == "/history":
            _cmd_history(conversation)
            continue

        if low.startswith("/system"):
            rest = user_input[7:].strip()
            if rest:
                system_prompt = rest
                print(f"{DIM}System prompt updated: {repr(system_prompt)}{RESET}\n")
            else:
                print(f"{DIM}Current system prompt: {repr(system_prompt)}{RESET}\n")
            continue

        if low.startswith("/url "):
            new_url = user_input[5:].strip().rstrip("/")
            print(f"{DIM}Fetching models from {new_url}…{RESET}", end="", flush=True)
            fetched = _fetch_models(new_url)
            if fetched is not None:
                base_url = new_url
                models = fetched
                print(f"\r{' ' * _W}\r{DIM}Switched to {base_url} — {len(models)} model(s) found.{RESET}\n")
                if not model or not any(m["id"] == model for m in models):
                    model = _auto_pick(models)
                    if model:
                        print(f"{DIM}Auto-selected model: {model}{RESET}\n")
            else:
                print(f"\r{' ' * _W}\r{RED}Could not connect to {new_url} — staying on {base_url}.{RESET}\n")
            continue

        if user_input.startswith("/"):
            print(f"{YELLOW}Unknown command: {user_input!r}   (type /help for commands){RESET}\n")
            continue

        # ── Chat ──────────────────────────────────────────────────────────────

        if not model:
            print(f"{YELLOW}No model selected. Type /model to choose one.{RESET}\n")
            continue

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(conversation)
        messages.append({"role": "user", "content": user_input})

        response_text = _stream_chat(base_url, model, messages, timeout=timeout)

        if response_text:
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
