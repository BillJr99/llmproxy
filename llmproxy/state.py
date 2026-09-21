"""Mutable routing state, behind one interface.

Every piece of state the routing layer accumulates at runtime — quota counters,
health windows, saturation cooldowns, learned capability gaps, oversize
watermarks, conversation affinity pins, the failure ring — used to live as a
module-level dict in ``server.py`` with its own ``threading.Lock``. Ten globals,
eight locks, no single place to look.

That works while there is one process. It is also the reason ``server.workers``
defaults to 1: none of it is shared, so a second worker counts free-tier quotas a
second time and never sees the first worker's cooldowns.

This module puts all of it behind one interface so the storage can change without
the ~80 call sites changing. Today there is one implementation, ``InMemoryState``,
which holds exactly the same dicts under exactly the same locks and is therefore
bit-identical to what came before. A cross-process implementation slots in beside
it without touching ``server.py`` again.

**The accessors in server.py keep their names and signatures.** They keep their
policy too — key construction, config reads, the log lines — and delegate only
storage. That split is deliberate: it keeps this module free of config and
logging dependencies, which is what makes it testable in isolation and what will
let a second implementation be checked against this one for identical behaviour.

**Clocks do not cross the interface.** Callers pass a TTL (``cooldown_s``), never
an expiry, and read back a decided answer (``is_saturated``), never a timestamp.
``InMemoryState`` uses ``time.monotonic()`` internally, so it stays immune to
system clock jumps; an implementation shared between processes has to store wall
clock instead, because that is the only clock two processes agree on. Neither
fact is visible from outside.
"""

from __future__ import annotations

import datetime
import threading
import time
from collections import deque
from typing import Protocol, runtime_checkable

from .usage import ModelUsage

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

@runtime_checkable
class SharedState(Protocol):
    """Storage for everything the routing layer learns while running.

    Implementations own *storage* semantics — a watermark keeps the minimum, a
    capability gap is a set union, a cooldown expires — and nothing else. They do
    not read config, log, or construct keys.
    """

    kind: str

    # — usage, tokens and health (one metered credential per key) —
    def record_usage(self, key: str, *, requests: int = 0, prompt: int = 0,
                     completion: int = 0, total: int = 0, cost: float = 0.0,
                     cost_source: str | None = None) -> None: ...
    def record_outcome(self, key: str, ok: bool,
                       latency_ms: float | None = None) -> None: ...
    def usage_snapshot(self, key: str) -> tuple[int, int]: ...
    def token_snapshot(self, key: str) -> tuple[int, int]: ...
    def health_snapshot(self, key: str) -> tuple[float, float, int]: ...
    def usage_rows(self) -> list[tuple[str, dict]]: ...
    def reset_usage(self) -> str: ...
    @property
    def usage_since(self) -> str: ...

    # — believed-free models observed reporting a cost —
    def flag_paid_free(self, key: str, cost: float, source: str) -> bool: ...
    def paid_free_flags(self) -> dict[str, dict]: ...

    # — saturation cooldowns —
    def mark_saturated(self, key: str, cooldown_s: float) -> None: ...
    def is_saturated(self, key: str) -> bool: ...
    def reset_saturation(self) -> None: ...

    # — oversize watermarks —
    def record_oversize(self, key: str, size_bytes: int) -> None: ...
    def is_oversize(self, key: str, size_bytes: int) -> bool: ...
    def has_oversize(self) -> bool: ...
    def clear_oversize_at(self, key: str, size_bytes: int) -> bool: ...
    def reset_oversize(self) -> None: ...

    # — learned capability gaps —
    def record_capability_gap(self, key: str, cap: str) -> bool: ...
    def capability_gaps(self, key: str) -> set[str]: ...
    def reset_capability_gaps(self) -> None: ...

    # — conversation affinity pins —
    def record_affinity(self, akey: str, provider: str, model: str) -> None: ...
    def affinity_target(self, akey: str) -> tuple[str, str] | None: ...
    def affinity_count(self) -> int: ...
    def reset_affinity(self) -> None: ...

    # — failure ring —
    def record_failure(self, record: dict) -> None: ...
    def failure_records(self, since_ts: float | None = None) -> list[dict]: ...
    def reset_failures(self) -> None: ...


# ---------------------------------------------------------------------------
# Bounds, carried here with the storage they bound
# ---------------------------------------------------------------------------

AFFINITY_PIN_MAX: int = 2048
AFFINITY_PIN_TTL_S: float = 6 * 60 * 60
FAILURE_LOG_MAX: int = 250
FAILURE_LOG_TTL_S: float = 6 * 60 * 60


# ---------------------------------------------------------------------------
# In-process implementation
# ---------------------------------------------------------------------------

class InMemoryState:
    """Today's behaviour: the same dicts, under the same locks, in one object.

    Every method below is the body that used to sit in ``server.py``, moved
    verbatim. The per-registry locks are kept separate rather than merged into
    one: they guard unrelated maps touched at very different rates, and a single
    lock would serialise the health window against the failure ring for no
    reason.
    """

    kind = "memory"

    def __init__(self) -> None:
        self._usage: dict[str, ModelUsage] = {}
        self._usage_lock = threading.Lock()
        self._usage_since: str = datetime.datetime.now(datetime.UTC).isoformat()

        self._paid_free: dict[str, dict] = {}
        self._paid_free_lock = threading.Lock()

        # key -> monotonic expiry
        self._saturation: dict[str, float] = {}
        self._saturation_lock = threading.Lock()

        # provider/model -> smallest body observed being refused, in bytes
        self._oversize: dict[str, int] = {}
        self._oversize_lock = threading.Lock()

        self._capability_gaps: dict[str, set[str]] = {}
        self._capability_gap_lock = threading.Lock()

        # affinity key -> ((provider, model), monotonic last-seen)
        self._affinity: dict[str, tuple[tuple[str, str], float]] = {}
        self._affinity_lock = threading.Lock()

        self._failures: deque = deque(maxlen=FAILURE_LOG_MAX)
        self._failure_lock = threading.Lock()

    # — usage —————————————————————————————————————————————————————————————

    def _tracker(self, key: str) -> ModelUsage:
        with self._usage_lock:
            tracker = self._usage.get(key)
            if tracker is None:
                tracker = ModelUsage()
                self._usage[key] = tracker
        return tracker

    def record_usage(self, key: str, *, requests: int = 0, prompt: int = 0,
                     completion: int = 0, total: int = 0, cost: float = 0.0,
                     cost_source: str | None = None) -> None:
        self._tracker(key).record(
            requests=requests, prompt=prompt, completion=completion,
            total=total, cost=cost, cost_source=cost_source,
        )

    def record_outcome(self, key: str, ok: bool,
                       latency_ms: float | None = None) -> None:
        self._tracker(key).record_outcome(ok, latency_ms)

    def usage_snapshot(self, key: str) -> tuple[int, int]:
        with self._usage_lock:
            tracker = self._usage.get(key)
        return tracker.snapshot() if tracker else (0, 0)

    def token_snapshot(self, key: str) -> tuple[int, int]:
        with self._usage_lock:
            tracker = self._usage.get(key)
        return tracker.token_snapshot() if tracker else (0, 0)

    def health_snapshot(self, key: str) -> tuple[float, float, int]:
        """(success_rate, avg_latency_ms, samples); an unseen key reads healthy.

        1.0 over zero samples means untried, not proven good — callers gate on
        the sample count before trusting the rate.
        """
        with self._usage_lock:
            tracker = self._usage.get(key)
        return tracker.health_snapshot() if tracker else (1.0, 0.0, 0)

    def usage_rows(self) -> list[tuple[str, dict]]:
        """One flattened row per metered key, for the usage report.

        Flattened rather than handing back trackers, so the report never depends
        on how a given implementation stores this.
        """
        with self._usage_lock:
            items = list(self._usage.items())
        rows: list[tuple[str, dict]] = []
        for key, tracker in items:
            row = dict(tracker.cost_snapshot())
            tok_min, tok_day = tracker.token_snapshot()
            rate, latency, samples = tracker.health_snapshot()
            row.update({
                "tokens_last_60s": tok_min,
                "tokens_today": tok_day,
                "success_rate": rate,
                "avg_latency_ms": latency,
                "health_samples": samples,
            })
            rows.append((key, row))
        return rows

    def reset_usage(self) -> str:
        """Clear the counters and return the new ``since`` stamp."""
        with self._usage_lock:
            self._usage.clear()
        with self._paid_free_lock:
            self._paid_free.clear()
        with self._saturation_lock:
            self._saturation.clear()
        self._usage_since = datetime.datetime.now(datetime.UTC).isoformat()
        return self._usage_since

    @property
    def usage_since(self) -> str:
        return self._usage_since

    # — believed-free models that reported a cost ——————————————————————————

    def flag_paid_free(self, key: str, cost: float, source: str) -> bool:
        """True only on the FIRST observation, so the caller persists it once."""
        with self._paid_free_lock:
            entry = self._paid_free.get(key)
            if entry is None:
                self._paid_free[key] = {
                    "observed_cost": round(cost, 8),
                    "cost_source": source,
                    "samples": 1,
                }
                return True
            entry["samples"] += 1
            entry["observed_cost"] = round(max(entry["observed_cost"], cost), 8)
            return False

    def paid_free_flags(self) -> dict[str, dict]:
        with self._paid_free_lock:
            return {k: dict(v) for k, v in self._paid_free.items()}

    # — saturation ————————————————————————————————————————————————————————

    def mark_saturated(self, key: str, cooldown_s: float) -> None:
        if cooldown_s <= 0:
            return
        with self._saturation_lock:
            self._saturation[key] = time.monotonic() + cooldown_s

    def is_saturated(self, key: str) -> bool:
        """True while *key* is still cooling; lazily evicts expired entries."""
        now = time.monotonic()
        with self._saturation_lock:
            expiry = self._saturation.get(key)
            if expiry is None:
                return False
            if expiry <= now:
                del self._saturation[key]
                return False
            return True

    def reset_saturation(self) -> None:
        with self._saturation_lock:
            self._saturation.clear()

    # — oversize watermarks ————————————————————————————————————————————————

    def record_oversize(self, key: str, size_bytes: int) -> None:
        """Keep the MINIMUM, so a smaller rejection tightens the watermark and a
        larger one never loosens it: the limit can only be bounded from above by
        what has actually been observed being refused."""
        if size_bytes <= 0:
            return
        with self._oversize_lock:
            prev = self._oversize.get(key)
            if prev is None or size_bytes < prev:
                self._oversize[key] = size_bytes

    def is_oversize(self, key: str, size_bytes: int) -> bool:
        if size_bytes <= 0:
            return False
        with self._oversize_lock:
            limit = self._oversize.get(key)
        return limit is not None and size_bytes >= limit

    def has_oversize(self) -> bool:
        """Whether any watermark is held at all.

        Lets the demotion pass return before serializing a payload to measure
        it, which is the common case: nothing has 413'd yet.
        """
        with self._oversize_lock:
            return bool(self._oversize)

    def clear_oversize_at(self, key: str, size_bytes: int) -> bool:
        """Forget the watermark if a request at least that large just succeeded.

        Only a success AT or above the watermark is evidence; a smaller one says
        nothing about the limit. Returns whether anything was cleared, so the
        caller logs once.
        """
        if size_bytes <= 0:
            return False
        with self._oversize_lock:
            limit = self._oversize.get(key)
            if limit is not None and size_bytes >= limit:
                del self._oversize[key]
                return True
        return False

    def reset_oversize(self) -> None:
        with self._oversize_lock:
            self._oversize.clear()

    # — learned capability gaps ————————————————————————————————————————————

    def record_capability_gap(self, key: str, cap: str) -> bool:
        """Returns True only when newly learned, so the caller warns once."""
        if not cap:
            return False
        with self._capability_gap_lock:
            known = self._capability_gaps.setdefault(key, set())
            if cap in known:
                return False
            known.add(cap)
            return True

    def capability_gaps(self, key: str) -> set[str]:
        with self._capability_gap_lock:
            return set(self._capability_gaps.get(key, ()))

    def reset_capability_gaps(self) -> None:
        with self._capability_gap_lock:
            self._capability_gaps.clear()

    # — affinity pins —————————————————————————————————————————————————————

    def _prune_affinity_locked(self) -> None:
        """Expired first, then oldest, until back inside the cap.

        The cap matters more than the TTL: an unbounded map keyed by conversation
        is a slow leak on a long-lived process.
        """
        cutoff = time.monotonic() - AFFINITY_PIN_TTL_S
        for key in [k for k, (_t, seen) in self._affinity.items() if seen < cutoff]:
            self._affinity.pop(key, None)
        if len(self._affinity) <= AFFINITY_PIN_MAX:
            return
        overflow = len(self._affinity) - AFFINITY_PIN_MAX
        for key, _ in sorted(self._affinity.items(), key=lambda kv: kv[1][1])[:overflow]:
            self._affinity.pop(key, None)

    def record_affinity(self, akey: str, provider: str, model: str) -> None:
        with self._affinity_lock:
            self._affinity[akey] = ((provider, model), time.monotonic())
            self._prune_affinity_locked()

    def affinity_target(self, akey: str) -> tuple[str, str] | None:
        with self._affinity_lock:
            entry = self._affinity.get(akey)
            if not entry:
                return None
            target, seen = entry
            if seen < time.monotonic() - AFFINITY_PIN_TTL_S:
                self._affinity.pop(akey, None)
                return None
            return target

    def affinity_count(self) -> int:
        """How many pins are held. Exposed so the cap can be asserted on."""
        with self._affinity_lock:
            return len(self._affinity)

    def reset_affinity(self) -> None:
        with self._affinity_lock:
            self._affinity.clear()

    # — failure ring ——————————————————————————————————————————————————————

    def record_failure(self, record: dict) -> None:
        with self._failure_lock:
            self._failures.append(record)

    def failure_records(self, since_ts: float | None = None) -> list[dict]:
        """Recent failures, newest first, pruned of anything past the TTL."""
        cutoff = time.time() - FAILURE_LOG_TTL_S
        with self._failure_lock:
            rows = [r for r in self._failures if r.get("ts", 0) >= cutoff]
            if len(rows) != len(self._failures):
                self._failures.clear()
                self._failures.extend(rows)
        if since_ts is not None:
            rows = [r for r in rows if r.get("ts", 0) >= since_ts]
        return list(reversed(rows))

    def reset_failures(self) -> None:
        with self._failure_lock:
            self._failures.clear()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_BACKEND: SharedState | None = None
_BACKEND_LOCK = threading.Lock()


def get_backend() -> SharedState:
    """The backend this process routes through, built on first use."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = InMemoryState()
    return _BACKEND


def set_backend(backend: SharedState | None) -> None:
    """Install a backend, or None to rebuild on next use. For tests and startup."""
    global _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = backend


def reset_for_worker() -> None:
    """Drop any backend inherited across a fork.

    Called from gunicorn's ``post_worker_init``. Nothing here holds a file
    descriptor today, so this is currently a formality — but it is the hook that
    makes it structurally impossible for a future connection-backed
    implementation to be shared across a ``fork()``, which is the classic way to
    corrupt one.
    """
    set_backend(None)
