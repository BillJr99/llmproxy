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

    # — Responses conversation store (previous_response_id) —
    def store_response(self, response_id: str, messages: list[dict]) -> None: ...
    def load_response(self, response_id: str) -> list[dict] | None: ...
    def delete_response(self, response_id: str) -> bool: ...
    def clear_responses(self) -> None: ...

    # — derived-cache invalidation —
    def bump_cache_epoch(self) -> None: ...
    def cache_epoch(self) -> int: ...

    # — cross-process single-flight —
    def acquire_lease(self, job: str, ttl_s: float) -> str | None: ...
    def renew_lease(self, job: str, token: str, ttl_s: float) -> bool: ...
    def release_lease(self, job: str, token: str) -> None: ...
    def leases(self) -> list[dict]: ...

    # — failure ring —
    def record_failure(self, record: dict) -> None: ...
    def failure_records(self, since_ts: float | None = None) -> list[dict]: ...
    def reset_failures(self) -> None: ...


# ---------------------------------------------------------------------------
# Bounds, carried here with the storage they bound
# ---------------------------------------------------------------------------

# The longest a cooldown may last. Also the clamp a shared backend applies on
# READ: a backward system-clock jump would otherwise leave an absolute deadline
# far in the future, cooling a healthy candidate indefinitely. Clamping bounds
# that to one window, and a forward jump only ends a cooldown early — which is
# the safe direction, since the next 429 re-establishes it.
MAX_SATURATION_COOLDOWN_S: float = 3600.0

AFFINITY_PIN_MAX: int = 2048
AFFINITY_PIN_TTL_S: float = 6 * 60 * 60
# The Responses API keeps conversation state server-side, so a client may send
# only its newest turn and reference the rest by id. This is the one place
# llmproxy holds conversation data, and it stays modest about it.
MAX_STORED_RESPONSES: int = 256

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

        # job -> (holder token, monotonic expiry)
        self._leases: dict[str, tuple[str, float]] = {}
        self._lease_lock = threading.Lock()

        self._cache_epoch = 0
        self._cache_epoch_lock = threading.Lock()

        from collections import OrderedDict
        self._responses: OrderedDict[str, list[dict]] = OrderedDict()
        self._response_lock = threading.Lock()

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
        # Through the accessor, not the dict: a subclass that stores cooldowns
        # elsewhere must have them cleared too, and reaching past its override
        # is how a reset silently half-works.
        self.reset_saturation()
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

    # — Responses conversation store ——————————————————————————————————————

    def store_response(self, response_id: str, messages: list[dict]) -> None:
        if not response_id:
            return
        with self._response_lock:
            self._responses[response_id] = messages
            self._responses.move_to_end(response_id)
            while len(self._responses) > MAX_STORED_RESPONSES:
                self._responses.popitem(last=False)

    def load_response(self, response_id: str) -> list[dict] | None:
        with self._response_lock:
            found = self._responses.get(response_id)
            if found is not None:
                self._responses.move_to_end(response_id)
            return list(found) if found is not None else None

    def delete_response(self, response_id: str) -> bool:
        with self._response_lock:
            return self._responses.pop(response_id, None) is not None

    def clear_responses(self) -> None:
        with self._response_lock:
            self._responses.clear()

    # — derived-cache invalidation ————————————————————————————————————————

    def bump_cache_epoch(self) -> None:
        """Signal that caches derived from config or routing data are stale.

        Needed because background jobs run on ONE worker now. Before leases,
        every worker ran the refresh and invalidated its own copy; now the other
        workers never see the invalidation and would serve a stale model list
        until their TTL lapsed. A counter each worker reads costs a dict lookup
        and an int compare.
        """
        with self._cache_epoch_lock:
            self._cache_epoch += 1

    def cache_epoch(self) -> int:
        with self._cache_epoch_lock:
            return self._cache_epoch

    # — single-flight leases ——————————————————————————————————————————————

    def acquire_lease(self, job: str, ttl_s: float) -> str | None:
        """Claim *job*, or None if someone already holds it.

        Replaces a plain "is it running" boolean, and is strictly better than
        one even in a single process: a worker that dies without releasing
        leaves a flag set forever, whereas a lease lapses. The job can then run
        again instead of never running again.
        """
        import uuid

        now = time.monotonic()
        token = uuid.uuid4().hex
        with self._lease_lock:
            held = self._leases.get(job)
            if held is not None and held[1] > now:
                return None
            self._leases[job] = (token, now + ttl_s)
            return token

    def renew_lease(self, job: str, token: str, ttl_s: float) -> bool:
        """Extend a lease this holder still owns. For jobs that outrun their TTL."""
        with self._lease_lock:
            held = self._leases.get(job)
            if held is None or held[0] != token:
                return False
            self._leases[job] = (token, time.monotonic() + ttl_s)
            return True

    def release_lease(self, job: str, token: str) -> None:
        """Release, but only if still the holder.

        Holder-scoped so a job that overran its TTL and was taken over cannot
        release the new holder's lease on its way out.
        """
        with self._lease_lock:
            held = self._leases.get(job)
            if held is not None and held[0] == token:
                del self._leases[job]

    def leases(self) -> list[dict]:
        now = time.monotonic()
        with self._lease_lock:
            return [
                {"job": job, "holder": tok, "expires_in": round(exp - now, 1)}
                for job, (tok, exp) in sorted(self._leases.items())
                if exp > now
            ]

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

# Set once in the gunicorn master, before it forks, and inherited by every
# worker. A path here means "share state through this file"; None means each
# process keeps its own, which is the default and what a single worker wants.
_SHARED_DB_PATH: str | None = None


def configure(db_path: str | None) -> None:
    """Choose what workers build. Called in the master, before forking.

    Deliberately not read from config here: this module has no business
    importing the config layer, and the decision belongs to the one place that
    already knows the worker count.
    """
    global _SHARED_DB_PATH
    _SHARED_DB_PATH = str(db_path) if db_path else None


def shared_db_path() -> str | None:
    return _SHARED_DB_PATH


def get_backend() -> SharedState:
    """The backend this process routes through, built on first use."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = (SqliteState(_SHARED_DB_PATH) if _SHARED_DB_PATH
                        else InMemoryState())
    return _BACKEND


def probe_shared_store(db_path) -> str | None:
    """Open the shared store for real; return None on success, else the reason.

    A real open, not a permission check: an unwritable directory, a filesystem
    without the mmap WAL needs, and a full disk all fail here and all fail
    differently, and the caller needs to say which in a message an operator can
    act on.
    """
    try:
        backend = SqliteState(db_path)
    except Exception as exc:  # noqa: BLE001 — every failure is a fallback, not a crash
        return f"{type(exc).__name__}: {exc}"
    try:
        backend.truncate()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()
    return None


def set_backend(backend: SharedState | None) -> None:
    """Install a backend, or None to rebuild on next use. For tests and startup."""
    global _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = backend


def reset_for_worker() -> None:
    """Drop the backend instance inherited across a fork, keeping the choice.

    The configured path survives -- every worker should build the same kind of
    backend -- but the object does not, so a connection can never be shared
    across a fork(), which is the classic way to corrupt one.

    Called from gunicorn's ``post_worker_init``. Nothing here holds a file
    descriptor today, so this is currently a formality — but it is the hook that
    makes it structurally impossible for a future connection-backed
    implementation to be shared across a ``fork()``, which is the classic way to
    corrupt one.
    """
    set_backend(None)


# ---------------------------------------------------------------------------
# Cross-process implementation
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 3

# The tables this backend owns so far. Everything not listed here still falls
# through to the in-process storage inherited from InMemoryState, which is
# correct-but-per-worker; see the class docstring.
_DDL = """
CREATE TABLE IF NOT EXISTS meta (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS capability_gap (
  key TEXT NOT NULL,
  cap TEXT NOT NULL,
  PRIMARY KEY (key, cap)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS saturation (
  key        TEXT PRIMARY KEY,
  expires_at REAL NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS ix_saturation_exp ON saturation(expires_at);

CREATE TABLE IF NOT EXISTS oversize (
  key       TEXT PRIMARY KEY,
  min_bytes INTEGER NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS affinity_pin (
  akey     TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model    TEXT NOT NULL,
  seen_at  REAL NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS ix_affinity_seen ON affinity_pin(seen_at);

CREATE TABLE IF NOT EXISTS response (
  response_id TEXT PRIMARY KEY,
  messages    TEXT NOT NULL,
  last_used   REAL NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS ix_response_used ON response(last_used);

CREATE TABLE IF NOT EXISTS lease (
  job         TEXT PRIMARY KEY,
  holder      TEXT NOT NULL,
  acquired_at REAL NOT NULL,
  expires_at  REAL NOT NULL
) WITHOUT ROWID;
"""

_DDL_STATEMENTS = [st.strip() for st in _DDL.split(";") if st.strip()]

_PRUNE_INTERVAL_S = 60.0


class SqliteState(InMemoryState):
    """State shared between the worker processes of one host.

    Subclasses ``InMemoryState`` deliberately. Each table added here overrides
    one group of accessors; everything not yet overridden keeps the in-process
    behaviour it inherits. That makes every intermediate state coherent — some
    facts shared, the rest per-worker, none of it broken — rather than requiring
    the whole surface to land at once.

    **One host only.** WAL needs its ``-shm`` mmap region, which does not work
    across machines, and ``server.workers`` is a per-host setting anyway. A
    state directory on a network filesystem is caught at init and reported
    rather than corrupting quietly.

    **Wall clock, not monotonic.** ``time.monotonic()`` has no shared epoch
    across processes, so deadlines are stored as absolute unix time. That trades
    immunity to clock jumps for the ability to be read by another process; the
    read-side clamp on ``MAX_SATURATION_COOLDOWN_S`` bounds what a backward jump
    can cost. Callers see none of this: they still pass a TTL and read a bool.

    **Writers always ``BEGIN IMMEDIATE``.** A deferred transaction that reads
    and then writes can deadlock two writers into ``SQLITE_BUSY`` that
    ``busy_timeout`` will not resolve, because neither can proceed without the
    other yielding.
    """

    kind = "sqlite"

    def __init__(self, path, *, worker: str | None = None) -> None:
        super().__init__()
        import os
        import pathlib
        import socket

        self.path = pathlib.Path(path)
        self.worker = worker or f"{socket.gethostname()}:{os.getpid()}"
        self._local = threading.local()
        self._last_prune = 0.0
        self._init_schema()

    # — connection handling ————————————————————————————————————————————————

    def _conn(self):
        """This thread's connection. gthread runs several threads per worker."""
        import sqlite3

        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.path), timeout=5.0, isolation_level=None,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL, not FULL: a worker crash loses nothing (WAL is durable
            # against process death); only host power loss can lose the last
            # commits, and every byte here is regenerable operational telemetry.
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        """Create the schema, or raise so the caller can fall back.

        Racing workers are serialised by SQLite itself, so N of them running
        this concurrently is safe. A schema from an older version is dropped
        rather than migrated: this store is truncated on every boot anyway, so
        there is nothing in it worth migrating.
        """
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            have_meta = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
            ).fetchone()
            if have_meta:
                row = conn.execute(
                    "SELECT v FROM meta WHERE k='schema_version'").fetchone()
                if row is None or int(row["v"]) != _SCHEMA_VERSION:
                    for table in ("capability_gap", "saturation", "oversize",
                                  "affinity_pin", "lease", "response", "meta"):
                        conn.execute(f"DROP TABLE IF EXISTS {table}")
            # Statement by statement, not executescript(): that issues its own
            # COMMIT first, which would silently end the transaction this
            # function is relying on.
            for stmt in _DDL_STATEMENTS:
                conn.execute(stmt)
            conn.execute(
                "INSERT INTO meta(k, v) VALUES('schema_version', ?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (str(_SCHEMA_VERSION),),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def truncate(self) -> None:
        """Empty every table. Run once by the master before it forks.

        Restart semantics must match the in-process backend's: counters and
        cooldowns do not survive a restart, and a file-backed store would
        silently make them. It also clears state left by a worker that was
        killed rather than shut down.
        """
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table in ("capability_gap", "saturation", "oversize",
                          "affinity_pin", "lease", "response"):
                conn.execute(f"DELETE FROM {table}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # — opportunistic pruning ——————————————————————————————————————————————

    def _maybe_prune(self, conn) -> None:
        """Drop what has expired, at most once a minute, inside the caller's
        transaction. A background thread for this would be a thread to supervise
        and to shut down cleanly, for work that any write can carry."""
        now = time.time()
        if now - self._last_prune < _PRUNE_INTERVAL_S:
            return
        self._last_prune = now
        conn.execute("DELETE FROM saturation WHERE expires_at < ?", (now,))
        conn.execute("DELETE FROM affinity_pin WHERE seen_at < ?",
                     (now - AFFINITY_PIN_TTL_S,))
        conn.execute(
            "DELETE FROM affinity_pin WHERE akey IN ("
            "  SELECT akey FROM affinity_pin ORDER BY seen_at DESC LIMIT -1 OFFSET ?)",
            (AFFINITY_PIN_MAX,),
        )
        # Long after expiry, so a lapsed lease stays visible on the diagnostics
        # endpoint for a while: "who ran this last, and when" is the question
        # being asked when a scheduled job did not happen.
        conn.execute("DELETE FROM lease WHERE expires_at < ?", (now - 3600,))

    def _write(self, statements) -> int:
        """Run statements in one BEGIN IMMEDIATE; returns the last changes()."""
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            changed = 0
            for sql, params in statements:
                cur = conn.execute(sql, params)
                changed = cur.rowcount
            self._maybe_prune(conn)
            conn.execute("COMMIT")
            return changed
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # — capability gaps (monotone union, no clock) ——————————————————————————

    def record_capability_gap(self, key: str, cap: str) -> bool:
        if not cap:
            return False
        return self._write([(
            "INSERT INTO capability_gap(key, cap) VALUES(?, ?) "
            "ON CONFLICT(key, cap) DO NOTHING", (key, cap),
        )]) > 0

    def capability_gaps(self, key: str) -> set[str]:
        rows = self._conn().execute(
            "SELECT cap FROM capability_gap WHERE key = ?", (key,)).fetchall()
        return {r["cap"] for r in rows}

    def reset_capability_gaps(self) -> None:
        self._write([("DELETE FROM capability_gap", ())])

    # — saturation (absolute deadlines, clamped on read) ————————————————————

    def mark_saturated(self, key: str, cooldown_s: float) -> None:
        if cooldown_s <= 0:
            return
        # MAX on conflict, so a longer Retry-After wins rather than whichever
        # worker happened to write last.
        self._write([(
            "INSERT INTO saturation(key, expires_at) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET expires_at=MAX(expires_at, excluded.expires_at)",
            (key, time.time() + cooldown_s),
        )])

    def is_saturated(self, key: str) -> bool:
        now = time.time()
        row = self._conn().execute(
            "SELECT 1 FROM saturation WHERE key = ? AND expires_at > ? AND expires_at <= ?",
            (key, now, now + MAX_SATURATION_COOLDOWN_S),
        ).fetchone()
        return row is not None

    def reset_saturation(self) -> None:
        self._write([("DELETE FROM saturation", ())])

    # — oversize watermarks (no clock) ——————————————————————————————————————

    def record_oversize(self, key: str, size_bytes: int) -> None:
        if size_bytes <= 0:
            return
        self._write([(
            "INSERT INTO oversize(key, min_bytes) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET min_bytes=MIN(min_bytes, excluded.min_bytes)",
            (key, size_bytes),
        )])

    def is_oversize(self, key: str, size_bytes: int) -> bool:
        if size_bytes <= 0:
            return False
        row = self._conn().execute(
            "SELECT 1 FROM oversize WHERE key = ? AND min_bytes <= ?",
            (key, size_bytes),
        ).fetchone()
        return row is not None

    def has_oversize(self) -> bool:
        return self._conn().execute(
            "SELECT 1 FROM oversize LIMIT 1").fetchone() is not None

    def clear_oversize_at(self, key: str, size_bytes: int) -> bool:
        if size_bytes <= 0:
            return False
        return self._write([(
            "DELETE FROM oversize WHERE key = ? AND min_bytes <= ?",
            (key, size_bytes),
        )]) > 0

    def reset_oversize(self) -> None:
        self._write([("DELETE FROM oversize", ())])

    # — affinity pins ——————————————————————————————————————————————————————

    def record_affinity(self, akey: str, provider: str, model: str) -> None:
        # The cap is enforced in the same transaction rather than left to the
        # pruner: it is what bounds growth, and a bound that can be exceeded for
        # a minute at a time is not one. The TTL stays opportunistic, since
        # affinity_target already filters expired pins on read.
        self._write([
            (
                "INSERT INTO affinity_pin(akey, provider, model, seen_at) "
                "VALUES(?, ?, ?, ?) "
                "ON CONFLICT(akey) DO UPDATE SET provider=excluded.provider, "
                "model=excluded.model, seen_at=excluded.seen_at",
                (akey, provider, model, time.time()),
            ),
            (
                "DELETE FROM affinity_pin WHERE akey IN ("
                "  SELECT akey FROM affinity_pin ORDER BY seen_at DESC "
                "  LIMIT -1 OFFSET ?)",
                (AFFINITY_PIN_MAX,),
            ),
        ])

    def affinity_target(self, akey: str) -> tuple[str, str] | None:
        row = self._conn().execute(
            "SELECT provider, model FROM affinity_pin WHERE akey = ? AND seen_at >= ?",
            (akey, time.time() - AFFINITY_PIN_TTL_S),
        ).fetchone()
        return (row["provider"], row["model"]) if row else None

    def affinity_count(self) -> int:
        return self._conn().execute(
            "SELECT COUNT(*) AS n FROM affinity_pin").fetchone()["n"]

    def reset_affinity(self) -> None:
        self._write([("DELETE FROM affinity_pin", ())])

    # — single-flight leases ——————————————————————————————————————————————

    def acquire_lease(self, job: str, ttl_s: float) -> str | None:
        """Claim *job* across every worker on this host, atomically.

        SQLite rather than an flock, for three reasons. An flock is released
        when a process dies but NOT when it hangs, so a worker stuck in a
        twenty-minute scrape would hold it forever and the job could never run
        again; a lease expires. The bug being fixed is also not really a mutex
        bug -- the due-check reads a timestamp written at the END of a job that
        takes minutes, so the "due" window is minutes wide and every worker
        passes its own in-process flag -- and making the check-and-claim atomic
        is a compare-and-set, which this is. And a lease is visible to an
        operator asking why a job did not run, where an flock is not.
        """
        import uuid

        now = time.time()
        token = uuid.uuid4().hex
        claimed = self._write([(
            "INSERT INTO lease(job, holder, acquired_at, expires_at) "
            "VALUES(?, ?, ?, ?) "
            "ON CONFLICT(job) DO UPDATE SET holder=excluded.holder, "
            "acquired_at=excluded.acquired_at, expires_at=excluded.expires_at "
            "WHERE lease.expires_at < excluded.acquired_at",
            (job, token, now, now + ttl_s),
        )])
        return token if claimed else None

    def renew_lease(self, job: str, token: str, ttl_s: float) -> bool:
        return self._write([(
            "UPDATE lease SET expires_at = ? WHERE job = ? AND holder = ?",
            (time.time() + ttl_s, job, token),
        )]) > 0

    def release_lease(self, job: str, token: str) -> None:
        self._write([(
            "DELETE FROM lease WHERE job = ? AND holder = ?", (job, token),
        )])

    def leases(self) -> list[dict]:
        now = time.time()
        rows = self._conn().execute(
            "SELECT job, holder, acquired_at, expires_at FROM lease "
            "WHERE expires_at > ? ORDER BY job", (now,),
        ).fetchall()
        return [
            {
                "job": r["job"],
                "holder": r["holder"],
                "expires_in": round(r["expires_at"] - now, 1),
                "held_for": round(now - r["acquired_at"], 1),
            }
            for r in rows
        ]

    # — derived-cache invalidation ————————————————————————————————————————

    def bump_cache_epoch(self) -> None:
        self._write([(
            "INSERT INTO meta(k, v) VALUES('cache_epoch', '1') "
            "ON CONFLICT(k) DO UPDATE SET v = CAST(CAST(meta.v AS INTEGER) + 1 AS TEXT)",
            (),
        )])

    def cache_epoch(self) -> int:
        row = self._conn().execute(
            "SELECT v FROM meta WHERE k = 'cache_epoch'").fetchone()
        try:
            return int(row["v"]) if row else 0
        except (TypeError, ValueError):
            return 0

    # — Responses conversation store ——————————————————————————————————————

    def store_response(self, response_id: str, messages: list[dict]) -> None:
        """Save a transcript where every worker can find it.

        This is the one piece of state whose absence is a hard error rather than
        a degradation: a client that sends previous_response_id and lands on a
        worker without it gets a 400, roughly (N-1)/N of the time. Everything
        else here merely routes worse when it is not shared.
        """
        if not response_id:
            return
        import json

        self._write([
            (
                "INSERT INTO response(response_id, messages, last_used) "
                "VALUES(?, ?, ?) "
                "ON CONFLICT(response_id) DO UPDATE SET messages=excluded.messages, "
                "last_used=excluded.last_used",
                (response_id, json.dumps(messages), time.time()),
            ),
            (
                "DELETE FROM response WHERE response_id IN ("
                "  SELECT response_id FROM response ORDER BY last_used DESC "
                "  LIMIT -1 OFFSET ?)",
                (MAX_STORED_RESPONSES,),
            ),
        ])

    def load_response(self, response_id: str) -> list[dict] | None:
        import json

        row = self._conn().execute(
            "SELECT messages, last_used FROM response WHERE response_id = ?",
            (response_id,),
        ).fetchone()
        if row is None:
            return None
        # An LRU touch on every read would make every read a write transaction.
        # Refreshing only once a minute keeps the eviction order accurate to far
        # finer than a 256-entry cap needs.
        if time.time() - row["last_used"] > 60:
            try:
                self._write([(
                    "UPDATE response SET last_used = ? WHERE response_id = ?",
                    (time.time(), response_id),
                )])
            except Exception:  # noqa: BLE001 — a stale LRU stamp is not worth failing a read
                pass
        try:
            return json.loads(row["messages"])
        except (TypeError, ValueError):
            return None

    def delete_response(self, response_id: str) -> bool:
        return self._write([(
            "DELETE FROM response WHERE response_id = ?", (response_id,),
        )]) > 0

    def clear_responses(self) -> None:
        self._write([("DELETE FROM response", ())])
