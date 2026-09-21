#!/usr/bin/env python3
"""Time the read path that decides whether a shared backend is usable.

The three ordering passes read the state accessors once per candidate. On a
large free pool that is roughly a thousand reads per request, against about four
writes -- so the read path, not the write path, is what a shared backend has to
survive. This measures it both ways: per-key reads (what a naive port would do)
and one bulk snapshot reused across the pass (what the proxy actually does).

Not a pass/fail test. It exists to justify or refute the decision not to batch
writes, and to catch a future change that reintroduces a per-key query into that
loop.

    python scripts/bench_state.py [--candidates 50,200,1000]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llmproxy.state import InMemoryState, SqliteState  # noqa: E402


def _seed(backend, n: int) -> list[str]:
    keys = [f"provider{i % 7}/model-{i}" for i in range(n)]
    for k in keys:
        backend.record_usage(k, requests=1, total=25)
        backend.record_outcome(k, True, latency_ms=12.0)
    return keys


def _read_per_key(backend, keys) -> None:
    """What a naive port would do: one query per accessor per candidate."""
    for k in keys:
        backend.is_saturated(k)
        backend.is_saturated(k + "/__provider__")
        backend.usage_snapshot(k)
        backend.token_snapshot(k)
        backend.health_snapshot(k)


def _read_via_snapshot(backend, keys) -> None:
    """Mirror the real loop: three separate accessor calls per candidate.

    _is_candidate_saturated asks twice (model and provider circuit), and the
    capacity score then asks for requests, tokens and health independently.
    Measuring one row() per key would flatter the snapshot by collapsing work
    the real path does not collapse.
    """
    snap = backend.snapshot()
    for k in keys:
        snap.is_saturated(k)
        snap.is_saturated(k + "/__provider__")
        req = snap.row(k)
        _ = req.req_min, req.req_day
        tok = snap.row(k)
        _ = tok.tok_min, tok.tok_day
        health = snap.row(k)
        _ = health.success_rate, health.health_samples


def _time(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - start) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="50,200,1000")
    args = ap.parse_args()
    sizes = [int(x) for x in args.candidates.split(",") if x.strip()]

    print(f"{'backend':<10} {'pool':>6} {'per-key ms':>12} {'snapshot ms':>13} "
          f"{'speedup':>9} {'write ms':>10}")
    print("-" * 66)

    with tempfile.TemporaryDirectory() as tmp:
        for n in sizes:
            for name in ("memory", "sqlite"):
                backend = (InMemoryState() if name == "memory"
                           else SqliteState(pathlib.Path(tmp) / f"b{n}.db"))
                keys = _seed(backend, n)

                _read_per_key(backend, keys[:5])          # warm
                per_key = _time(_read_per_key, backend, keys)
                snap = _time(_read_via_snapshot, backend, keys)

                w_start = time.perf_counter()
                for _ in range(20):
                    backend.record_usage(keys[0], requests=1, total=10)
                write = (time.perf_counter() - w_start) * 1000.0 / 20

                ratio = (per_key / snap) if snap else float("inf")
                print(f"{name:<10} {n:>6} {per_key:>12.1f} {snap:>13.1f} "
                      f"{ratio:>8.1f}x {write:>10.3f}")
                close = getattr(backend, "close", None)
                if close:
                    close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
