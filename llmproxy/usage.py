"""usage.py — token + cost accounting primitives for llmproxy.

This module holds the *pure* (web-framework-free) pieces of usage tracking so
they can be unit-tested in isolation and reused by the offline scraper probe
(scripts/sources/probe.py):

  * ``ModelUsage`` — a thread-safe per-model counter that tracks both requests
    (sliding 60s window + per-day) and tokens (same windows) plus lifetime
    totals and an accumulated dollar cost.
  * ``extract_usage`` / ``parse_stream_usage`` — pull the OpenAI ``usage`` block
    out of a non-streaming body or the final SSE chunk of a stream.
  * ``load_pricing_map`` / ``compute_cost`` — price a request in USD, preferring
    the provider-reported ``usage.cost`` (OpenRouter / Vercel) and otherwise
    computing it from the per-token ``pricing`` block bundled in providers.json.

Accounting is in-memory and per-process — consistent with the existing
request-count load balancer in server.py. It resets on restart.
"""

from __future__ import annotations

import collections
import datetime
import json
import threading
import time

from . import providers as _providers

# ---------------------------------------------------------------------------
# Per-model usage counter
# ---------------------------------------------------------------------------


def today_start_ts() -> float:
    """Unix timestamp for the start of today (local midnight)."""
    return time.mktime(datetime.date.today().timetuple())


class ModelUsage:
    """Thread-safe request + token + cost accounting for one upstream model.

    The request-count behavior is a superset of the original counter: calling
    ``record(requests=1)`` with no token arguments reproduces the old sliding
    60-second / per-day request windows exactly. Token windows mirror the
    request windows so the load balancer can honor tokens_per_minute /
    tokens_per_day the same way it honors requests_per_minute / _per_day.
    """

    __slots__ = (
        "_lock",
        "_minute_ts", "_day_count", "_day_start",          # request windows
        "_minute_tokens", "_day_tokens",                    # token windows
        "_lifetime_requests", "_lifetime_prompt",
        "_lifetime_completion", "_lifetime_total",
        "_lifetime_cost", "_cost_sources",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._minute_ts: collections.deque = collections.deque()
        self._day_count: int = 0
        self._day_start: float = today_start_ts()
        self._minute_tokens: collections.deque = collections.deque()  # (mono_ts, tokens)
        self._day_tokens: int = 0
        self._lifetime_requests: int = 0
        self._lifetime_prompt: int = 0
        self._lifetime_completion: int = 0
        self._lifetime_total: int = 0
        self._lifetime_cost: float = 0.0
        self._cost_sources: dict[str, int] = {}

    def record(
        self,
        *,
        requests: int = 0,
        prompt: int = 0,
        completion: int = 0,
        total: int = 0,
        cost: float = 0.0,
        cost_source: str | None = None,
    ) -> None:
        """Accumulate one observation.

        Requests and tokens are recorded independently so the streaming path can
        count the request up front (for load balancing) and add the token totals
        later, once the final SSE chunk has been parsed.
        """
        now_mono = time.monotonic()
        now_wall = time.time()
        with self._lock:
            if now_wall >= self._day_start + 86400:
                self._day_start = today_start_ts()
                self._day_count = 0
                self._day_tokens = 0
            if requests:
                self._day_count += requests
                self._lifetime_requests += requests
                for _ in range(requests):
                    self._minute_ts.append(now_mono)
            if total or prompt or completion:
                self._minute_tokens.append((now_mono, total))
                self._day_tokens += total
                self._lifetime_prompt += prompt
                self._lifetime_completion += completion
                self._lifetime_total += total
            if cost:
                self._lifetime_cost += cost
            if cost_source:
                self._cost_sources[cost_source] = self._cost_sources.get(cost_source, 0) + 1

    def snapshot(self) -> tuple[int, int]:
        """Return (requests_last_60s, requests_today), pruning stale entries."""
        now_mono = time.monotonic()
        now_wall = time.time()
        with self._lock:
            if now_wall >= self._day_start + 86400:
                return 0, 0
            cutoff = now_mono - 60.0
            while self._minute_ts and self._minute_ts[0] < cutoff:
                self._minute_ts.popleft()
            return len(self._minute_ts), self._day_count

    def token_snapshot(self) -> tuple[int, int]:
        """Return (tokens_last_60s, tokens_today), pruning stale entries."""
        now_mono = time.monotonic()
        now_wall = time.time()
        with self._lock:
            if now_wall >= self._day_start + 86400:
                return 0, 0
            cutoff = now_mono - 60.0
            while self._minute_tokens and self._minute_tokens[0][0] < cutoff:
                self._minute_tokens.popleft()
            tokens_min = sum(t for _, t in self._minute_tokens)
            return tokens_min, self._day_tokens

    def cost_snapshot(self) -> dict:
        """Return lifetime totals for the usage report."""
        with self._lock:
            return {
                "requests": self._lifetime_requests,
                "prompt_tokens": self._lifetime_prompt,
                "completion_tokens": self._lifetime_completion,
                "total_tokens": self._lifetime_total,
                "cost": round(self._lifetime_cost, 8),
                "cost_sources": dict(self._cost_sources),
            }


# ---------------------------------------------------------------------------
# usage-block extraction
# ---------------------------------------------------------------------------


def _coerce_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_cost(usage: dict) -> float | None:
    """Return a provider-reported USD cost when present, else None.

    OpenRouter returns ``usage.cost``; some gateways nest it under
    ``usage.cost_details.upstream_inference_cost`` or report ``total_cost``.
    """
    for key in ("cost", "total_cost"):
        val = usage.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    details = usage.get("cost_details")
    if isinstance(details, dict):
        val = details.get("upstream_inference_cost") or details.get("total_cost")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def extract_usage(body) -> dict | None:
    """Parse an OpenAI ``usage`` object from a response body.

    *body* may be raw bytes/str (a JSON response) or an already-decoded dict.
    Returns a normalized dict with ``prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens`` (ints) and, when the provider reported it, a float
    ``cost``. Returns ``None`` for any shape that does not carry usage — this is
    deliberately defensive so a malformed upstream body never raises.
    """
    if body is None:
        return None
    obj = body
    if isinstance(obj, (bytes, bytearray, str)):
        try:
            obj = json.loads(obj)
        except (ValueError, TypeError):
            return None
    if not isinstance(obj, dict):
        return None
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = _coerce_int(usage.get("prompt_tokens"))
    completion = _coerce_int(usage.get("completion_tokens"))
    total = _coerce_int(usage.get("total_tokens")) or (prompt + completion)
    if not (prompt or completion or total):
        # An empty/zero usage block carries no information.
        return None
    out = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    cost = _coerce_cost(usage)
    if cost is not None:
        out["cost"] = cost
    return out


def parse_stream_usage(tail: bytes | str) -> dict | None:
    """Scan the tail of an SSE byte stream for a chunk carrying ``usage``.

    OpenAI-style streams emit the usage block in the final ``data:`` event
    (before ``data: [DONE]``) when ``stream_options.include_usage`` is set.
    We scan ``data:`` lines from the end and return the first that yields a
    usage object.
    """
    if not tail:
        return None
    if isinstance(tail, (bytes, bytearray)):
        try:
            text = bytes(tail).decode("utf-8", errors="ignore")
        except Exception:
            return None
    else:
        text = tail
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        usage = extract_usage(data)
        if usage is not None:
            return usage
    return None


# ---------------------------------------------------------------------------
# pricing map + cost computation
# ---------------------------------------------------------------------------

_pricing_cache: dict[str, tuple[float, float]] | None = None
_pricing_cache_ts: float = 0.0
_pricing_lock = threading.Lock()
_PRICING_TTL = 300.0


def load_pricing_map(data: dict | None = None, *, force: bool = False) -> dict[str, tuple[float, float]]:
    """Return ``provider/model`` (lowercased) → (input_cost, output_cost) per token.

    Built from the top-level ``pricing`` block of providers.json (populated by
    the scraper's litellm cost-map pass). Cached in-process for a short TTL so
    the hot request path never re-reads the file. Pass *data* to bypass the file
    (used by tests); pass ``force=True`` to refresh the cache.
    """
    global _pricing_cache, _pricing_cache_ts
    if data is not None:
        return _providers.get_pricing_map(data)
    now = time.monotonic()
    with _pricing_lock:
        if not force and _pricing_cache is not None and (now - _pricing_cache_ts) < _PRICING_TTL:
            return _pricing_cache
        try:
            _pricing_cache = _providers.get_pricing_map(_providers.load_data())
        except Exception:
            _pricing_cache = {} if _pricing_cache is None else _pricing_cache
        _pricing_cache_ts = now
        return _pricing_cache


def compute_cost(
    provider: str,
    model: str,
    usage: dict | None,
    pricing: dict[str, tuple[float, float]] | None,
) -> tuple[float, str]:
    """Return (cost_usd, source).

    Preference order:
      1. ``provider`` — the upstream reported ``usage.cost`` directly.
      2. ``computed`` — priced from *pricing* (input*prompt + output*completion).
      3. ``unknown`` — no cost signal available; returns (0.0, "unknown").
    """
    if not usage:
        return 0.0, "unknown"
    reported = usage.get("cost")
    if isinstance(reported, (int, float)):
        return float(reported), "provider"
    if pricing:
        key = f"{provider}/{model}".lower()
        prices = pricing.get(key)
        if prices is None:
            # Fall back to a model-only key (some pricing maps are not provider-scoped).
            prices = pricing.get(model.lower())
        if prices:
            in_cost, out_cost = prices
            cost = usage.get("prompt_tokens", 0) * in_cost + usage.get("completion_tokens", 0) * out_cost
            return float(cost), "computed"
    return 0.0, "unknown"
