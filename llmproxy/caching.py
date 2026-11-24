"""Prompt caching: settings, per-provider request shaping, and usage counters.

Every provider that caches prompts does it differently, and only some of them
need the request to ask for it:

* **Anthropic Messages** (``"protocol": "anthropic"``) caches only what the
  request marks with ``cache_control`` breakpoints (at most four). Unmarked
  requests are never cached. A marker may carry ``"ttl": "1h"`` for the
  one-hour cache; omitting ``ttl`` gives the default five minutes.
* **OpenRouter** forwards ``cache_control`` content-part markers to the
  Anthropic models it serves (and to the other explicit-cache families), and
  caches the automatic-cache families (OpenAI, DeepSeek, Grok, Moonshot, Groq,
  Gemini 2.5+) on its own.
* **OpenAI** caches every prompt of 1024 tokens or more automatically.
  ``prompt_cache_key`` steers related requests to the same cache and
  ``prompt_cache_retention`` (``"in_memory"`` or ``"24h"``) picks the lifetime
  on the models that support extended retention.
* **Gemini** (2.5 and newer) caches implicitly with no request change.
  Explicit Gemini caching needs a separately created ``cachedContents``
  resource, which a stateless proxy does not manage, so it is not offered.
* **DeepSeek**, Grok, Moonshot and the other automatic caches need nothing.

So this module does three things:

1. Resolves the ``prompt_caching`` setting for one provider from the defaults,
   the global config block, the ``LLMPROXY_PROMPT_CACHING*`` environment
   variables and the provider's own entry, in that order.
2. Shapes one outbound request when caching is on: Anthropic breakpoints on
   the tool list, the system prompt and the newest turn; the same markers as
   content parts for Claude models behind OpenRouter (or any OpenAI-compatible
   provider that opts in with ``"inject_markers": true``); and
   ``prompt_cache_key`` / ``prompt_cache_retention`` for OpenAI itself. A
   request that already carries its own markers is left exactly as the client
   wrote it. With ``"passthrough": false`` a provider instead has every
   client-supplied cache field removed, for an upstream that rejects them.
3. Normalises the cache counters each provider reports into one shape
   (``cache_read_tokens`` / ``cache_creation_tokens``) for ``/v1/usage``.

Caching is on by default with the five-minute lifetime (``ttl: "5m"``).
``"1h"`` keeps the cache for an hour at a higher write price (2x the input
price instead of 1.25x; reads cost 0.1x either way), and ``"off"`` (or
``enabled: false``) adds nothing. llmproxy never warms a cache by itself.
"""

from __future__ import annotations

import copy
import hashlib
import os
from urllib.parse import urlparse

# Values accepted for ``ttl``: "off" disables caching, "5m" and "1h" are the
# Anthropic-style breakpoint lifetimes.
TTL_VALUES = ("off", "5m", "1h")
# Values accepted for ``warming``. llmproxy never sends a request of its own to
# keep a cache warm, so "off" is the only value; a client that warms (pi, for
# one) is relayed like any other request.
WARMING_VALUES = ("off",)
# Values accepted for ``retention`` (OpenAI ``prompt_cache_retention``).
RETENTION_VALUES = ("in_memory", "24h")

DEFAULTS: dict = {
    # Master switch for everything llmproxy ADDS to a request. Client-supplied
    # cache fields are relayed whether or not this is on (see ``passthrough``).
    # Kept for compatibility: ``"ttl": "off"`` is the same as ``false``.
    "enabled": True,
    # Lifetime of the breakpoints llmproxy places: "off", "5m" (default), "1h".
    "ttl": "5m",
    # Cache warming: always "off" (see WARMING_VALUES).
    "warming": "off",
    # OpenAI prompt_cache_retention to send when the client did not: None
    # (send nothing, so the upstream default applies), "in_memory" or "24h".
    "retention": None,
    # "auto": place markers where the provider documents them (Anthropic
    # protocol, Claude models on OpenRouter). True: also on this
    # OpenAI-compatible provider. False: never place markers here.
    "inject_markers": "auto",
    # Derive a prompt_cache_key for OpenAI when the client sent none.
    "cache_key": True,
    # Relay client-supplied cache fields. "auto": relay everything to the
    # providers that document caching fields (Anthropic protocol, OpenRouter,
    # api.openai.com, or inject_markers true); elsewhere drop the fields an
    # OpenAI-compatible server is likely to reject (top-level cache_control,
    # prompt_cache_retention, prompt_cache_options, markers on tool results)
    # and keep the rest. True relays everything everywhere; False strips all.
    "passthrough": "auto",
}

_ENV_ENABLED = "LLMPROXY_PROMPT_CACHING"
_ENV_TTL = "LLMPROXY_PROMPT_CACHE_TTL"
_ENV_RETENTION = "LLMPROXY_PROMPT_CACHE_RETENTION"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

# Anthropic allows four breakpoints per request; llmproxy uses at most three.
MAX_BREAKPOINTS = 4

# Request fields that carry a client's caching intent, stripped when a
# provider has ``passthrough: false``.
_CLIENT_TOP_LEVEL_FIELDS = (
    "cache_control", "prompt_cache_key", "prompt_cache_retention", "prompt_cache_options",
)

# Content blocks the Anthropic API refuses a cache_control on.
_UNMARKABLE_BLOCKS = {"thinking", "redacted_thinking"}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _overlay(base: dict, raw) -> dict:
    """Merge one settings layer: a bool toggles ``enabled``, a dict merges.

    ``ttl`` and ``enabled`` are two spellings of one switch, reconciled per
    layer: a layer that sets ``ttl`` to "5m" or "1h" turns caching on (unless
    the same layer says ``enabled: false``), ``ttl: "off"`` turns it off, and
    turning caching on over an inherited "off" restores the 5-minute default.
    """
    out = dict(base)
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    if not isinstance(raw, dict):
        return out
    for key in DEFAULTS:
        if key in raw:
            out[key] = raw[key]
    ttl = raw.get("ttl")
    if ttl == "off":
        out["enabled"] = False
    elif ttl in TTL_VALUES and "enabled" not in raw:
        out["enabled"] = True
    if out.get("enabled") is True and out.get("ttl") == "off":
        out["ttl"] = DEFAULTS["ttl"]
    return out


def _env_layer(environ) -> dict:
    layer: dict = {}
    enabled = (environ.get(_ENV_ENABLED) or "").strip().lower()
    if enabled in _TRUE:
        layer["enabled"] = True
    elif enabled in _FALSE:
        layer["enabled"] = False
    ttl = (environ.get(_ENV_TTL) or "").strip()
    if ttl:
        layer["ttl"] = ttl
    retention = (environ.get(_ENV_RETENTION) or "").strip()
    if retention:
        layer["retention"] = None if retention.lower() in ("none", "default") else retention
    return layer


def _validated(settings: dict) -> dict:
    out = dict(settings)
    out["enabled"] = bool(out.get("enabled"))
    if out.get("ttl") not in TTL_VALUES:
        out["ttl"] = DEFAULTS["ttl"]
    if out["ttl"] == "off":
        out["enabled"] = False
    if not out["enabled"]:
        out["ttl"] = "off"
    out["warming"] = "off"
    if out.get("retention") not in RETENTION_VALUES:
        out["retention"] = None
    if out.get("inject_markers") not in (True, False, "auto"):
        out["inject_markers"] = "auto"
    out["cache_key"] = bool(out.get("cache_key", True))
    if out.get("passthrough") not in (True, False, "auto"):
        out["passthrough"] = "auto"
    return out


def resolve_settings(config: dict | None, provider_cfg: dict | None, environ=None) -> dict:
    """Effective prompt-caching settings for one provider.

    The result always has ``enabled`` and ``ttl`` in agreement: ``enabled`` is
    false exactly when ``ttl`` is "off".

    Layers, later wins: :data:`DEFAULTS`, the top-level ``prompt_caching``
    block (``true``/``false`` or an object), the ``LLMPROXY_PROMPT_CACHING``,
    ``LLMPROXY_PROMPT_CACHE_TTL`` and ``LLMPROXY_PROMPT_CACHE_RETENTION``
    environment variables, then the provider entry's own ``prompt_caching``.
    Unknown or invalid values fall back to the defaults, so a typo can never
    produce a request the upstream rejects.
    """
    environ = os.environ if environ is None else environ
    settings = _overlay(dict(DEFAULTS), (config or {}).get("prompt_caching"))
    settings = _overlay(settings, _env_layer(environ))
    settings = _overlay(settings, (provider_cfg or {}).get("prompt_caching"))
    return _validated(settings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _host(base_url: str | None) -> str:
    try:
        return (urlparse(base_url or "").hostname or "").lower()
    except ValueError:
        return ""


def _is_openrouter(base_url: str | None) -> bool:
    host = _host(base_url)
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _is_openai(base_url: str | None) -> bool:
    return _host(base_url) == "api.openai.com"


def _is_claude_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("anthropic/") or "claude" in m


def _marker(ttl: str) -> dict:
    marker: dict = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def _blocks_have_marker(blocks) -> bool:
    if not isinstance(blocks, list):
        return False
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("cache_control"):
            return True
        # tool_result blocks nest their own content list.
        if _blocks_have_marker(b.get("content")):
            return True
    return False


def has_client_markers(body: dict) -> bool:
    """True when the request already says how it wants to be cached."""
    if not isinstance(body, dict):
        return False
    if body.get("cache_control"):
        return True
    if _blocks_have_marker(body.get("tools")):
        return True
    if _blocks_have_marker(body.get("system")):
        return True
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and _blocks_have_marker(msg.get("content")):
            return True
    return False


def _strip_markers(value):
    """Return *value* with every nested ``cache_control`` removed."""
    if isinstance(value, dict):
        return {k: _strip_markers(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_markers(v) for v in value]
    return value


def strip_client_cache_fields(body: dict) -> dict:
    """A copy of *body* without any client-supplied caching field."""
    out = {k: v for k, v in body.items() if k not in _CLIENT_TOP_LEVEL_FIELDS}
    for key in ("messages", "system", "tools", "input"):
        if key in out:
            out[key] = _strip_markers(out[key])
    return out


# Fields the "auto" passthrough drops for an OpenAI-compatible server that does
# not document caching: they are the ones such servers reject as unknown.
_AUTO_STRIPPED_TOP_LEVEL = ("cache_control", "prompt_cache_retention", "prompt_cache_options")


def _strip_for_generic_server(body: dict) -> dict:
    """Drop the caching fields a generic OpenAI-compatible server may reject.

    Markers on user, system and assistant content parts and a
    ``prompt_cache_key`` are kept (llmproxy has always relayed those), but a
    tool result marked as a parts list is flattened back to the plain string
    tool content most servers require.
    """
    if not any(k in body for k in _AUTO_STRIPPED_TOP_LEVEL) and not any(
        isinstance(m, dict) and m.get("role") == "tool" and isinstance(m.get("content"), list)
        and _blocks_have_marker(m.get("content"))
        for m in body.get("messages") or []
    ):
        return body
    out = {k: v for k, v in body.items() if k not in _AUTO_STRIPPED_TOP_LEVEL}
    messages = out.get("messages")
    if isinstance(messages, list):
        fixed = []
        for m in messages:
            if (isinstance(m, dict) and m.get("role") == "tool"
                    and isinstance(m.get("content"), list) and _blocks_have_marker(m["content"])):
                text = "".join(p.get("text", "") for p in m["content"]
                               if isinstance(p, dict) and p.get("type") == "text")
                m = {**m, "content": text}
            fixed.append(m)
        out["messages"] = fixed
    return out


def _mark_last_eligible(blocks: list, marker: dict) -> bool:
    for block in reversed(blocks):
        if not isinstance(block, dict) or block.get("type") in _UNMARKABLE_BLOCKS:
            continue
        if block.get("type") == "text" and not block.get("text"):
            continue  # empty text blocks cannot be cached
        block["cache_control"] = dict(marker)
        return True
    return False


# ---------------------------------------------------------------------------
# Request shaping, per wire format
# ---------------------------------------------------------------------------

def add_anthropic_breakpoints(body: dict, ttl: str = "5m") -> dict:
    """Anthropic Messages body -> a copy with up to three breakpoints.

    Marks the last tool definition, the last system block and the last
    markable block of the newest turn. Each later breakpoint caches everything
    before it, so the tools-and-system prefix stays cached across
    conversations while the newest turn extends the cache one turn at a time
    (the API looks back up to 20 blocks for an earlier hit). Requests that
    already carry a marker anywhere are returned unchanged.
    """
    if has_client_markers(body):
        return body
    out = dict(body)
    marker = _marker(ttl)
    tools = out.get("tools")
    if isinstance(tools, list) and tools:
        tools = copy.deepcopy(tools)
        _mark_last_eligible(tools, marker)
        out["tools"] = tools
    system = out.get("system")
    if isinstance(system, str) and system:
        out["system"] = [{"type": "text", "text": system, "cache_control": dict(marker)}]
    elif isinstance(system, list) and system:
        system = copy.deepcopy(system)
        _mark_last_eligible(system, marker)
        out["system"] = system
    messages = out.get("messages")
    if isinstance(messages, list) and messages:
        messages = copy.deepcopy(messages)
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                if content:
                    msg["content"] = [{"type": "text", "text": content,
                                       "cache_control": dict(marker)}]
                    break
                continue
            if isinstance(content, list) and _mark_last_eligible(content, marker):
                break
        out["messages"] = messages
    return out


def _mark_chat_message(msg: dict, marker: dict) -> bool:
    """Mark one OpenAI chat message's last text part, converting a string."""
    content = msg.get("content")
    if isinstance(content, str):
        if not content:
            return False
        msg["content"] = [{"type": "text", "text": content, "cache_control": dict(marker)}]
        return True
    if isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                part["cache_control"] = dict(marker)
                return True
    return False


def add_chat_breakpoints(body: dict, ttl: str = "5m") -> dict:
    """OpenAI chat body -> a copy with content-part ``cache_control`` markers.

    The OpenRouter form: a marker on the last text part of the last system
    message and of the newest user or tool-result message, so an agent loop
    keeps extending the cache through its tool iterations. Assistant turns are
    left alone.
    """
    if has_client_markers(body):
        return body
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body
    out = dict(body)
    messages = copy.deepcopy(messages)
    marker = _marker(ttl)
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "system":
            _mark_chat_message(msg, marker)
            break
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") in ("user", "tool"):
            if _mark_chat_message(msg, marker):
                break
    out["messages"] = messages
    return out


def derive_cache_key(affinity_key: str | None) -> str | None:
    """A short, stable ``prompt_cache_key`` from a conversation identity."""
    if not affinity_key:
        return None
    return "llmproxy-" + hashlib.sha256(affinity_key.encode("utf-8")).hexdigest()[:32]


def apply(
    protocol: str | None,
    provider_cfg: dict,
    body,
    headers: dict,
    settings: dict,
    *,
    endpoint: str = "chat/completions",
    affinity_key: str | None = None,
):
    """Shape one built upstream request for prompt caching.

    *protocol* is the outbound adapter's name, *body* the provider-native
    request it built. Returns ``(body, headers)``; *body* is never mutated in
    place (the caller may be about to retry the same payload elsewhere).
    """
    if not isinstance(body, dict):
        return body, headers
    if endpoint not in ("chat/completions", "messages", "responses"):
        return body, headers
    protocol = protocol or "openai"
    base_url = provider_cfg.get("base_url") if isinstance(provider_cfg, dict) else None
    passthrough = settings.get("passthrough", "auto")
    if passthrough is False:
        body = strip_client_cache_fields(body)
    elif passthrough == "auto" and protocol in ("openai", "responses") and not (
        _is_openrouter(base_url) or _is_openai(base_url)
        or settings.get("inject_markers") is True
    ):
        body = _strip_for_generic_server(body)
    if not settings.get("enabled"):
        return body, headers

    model = str(body.get("model") or "")
    ttl = settings.get("ttl", "5m")
    inject = settings.get("inject_markers", "auto")

    if protocol == "anthropic":
        if inject is not False:
            body = add_anthropic_breakpoints(body, ttl)
        return body, headers

    if protocol == "openai":
        wants_markers = inject is True or (
            inject == "auto" and _is_openrouter(base_url) and _is_claude_model(model)
        )
        if wants_markers:
            body = add_chat_breakpoints(body, ttl)

    if protocol in ("openai", "responses") and _is_openai(base_url):
        extra: dict = {}
        if settings.get("cache_key") and not body.get("prompt_cache_key"):
            key = derive_cache_key(affinity_key)
            if key:
                extra["prompt_cache_key"] = key
        if settings.get("retention") and not body.get("prompt_cache_retention"):
            extra["prompt_cache_retention"] = settings["retention"]
        if extra:
            body = {**body, **extra}

    return body, headers


# ---------------------------------------------------------------------------
# Usage counters
# ---------------------------------------------------------------------------

def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def cache_counters(usage: dict | None) -> tuple[int, int]:
    """``(cache_read_tokens, cache_creation_tokens)`` from any usage shape.

    Reads the OpenAI form (``prompt_tokens_details.cached_tokens`` and
    ``cache_write_tokens``, or the Responses ``input_tokens_details``), the
    Anthropic form (``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``) and DeepSeek's
    ``prompt_cache_hit_tokens``.
    """
    if not isinstance(usage, dict):
        return 0, 0
    read = 0
    write = 0
    for key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict):
            read = read or _int(details.get("cached_tokens"))
            write = write or _int(details.get("cache_write_tokens"))
    read = read or _int(usage.get("cache_read_input_tokens")) or _int(
        usage.get("prompt_cache_hit_tokens"))
    write = write or _int(usage.get("cache_creation_input_tokens"))
    return read, write
