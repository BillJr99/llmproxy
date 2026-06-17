"""Tests for deterministic sidecar serialization (canonicalize_sidecar / dump_sidecar).

The optional providers PR is reopened whenever the written providers.json differs
from the base branch. Insertion-ordered per-provider blocks made re-scrapes churn
the byte output even with no logical change; canonicalize_sidecar pins the order.
"""

from __future__ import annotations

import json

from scripts.update_free_models import canonicalize_sidecar, dump_sidecar


def _sidecar() -> dict:
    return {
        "providers": {
            "groq": {
                "base_url": "http://groq/v1",
                "believed_free": ["groq/zeta", "groq/Alpha", "groq/mu"],
                "free_limits": {"groq/zeta": {"rpm": 1}, "groq/Alpha": {"rpm": 2}},
                "model_reasoning": {"groq/zeta": "deep", "groq/Alpha": "standard"},
                "model_capabilities": {"groq/zeta": ["tools"], "groq/Alpha": ["json"]},
            }
        },
        "pricing": {"b/m": {}, "a/m": {}},
    }


def test_canonicalize_sorts_blocks_case_insensitively():
    s = _sidecar()
    canonicalize_sidecar(s)
    prov = s["providers"]["groq"]
    assert prov["believed_free"] == ["groq/Alpha", "groq/mu", "groq/zeta"]
    assert list(prov["free_limits"]) == ["groq/Alpha", "groq/zeta"]
    assert list(prov["model_reasoning"]) == ["groq/Alpha", "groq/zeta"]
    assert list(prov["model_capabilities"]) == ["groq/Alpha", "groq/zeta"]


def test_canonicalize_is_pure_reorder():
    s = _sidecar()
    before = json.loads(json.dumps(s))  # deep copy
    canonicalize_sidecar(s)

    def deep_sort(o):
        if isinstance(o, dict):
            return {k: deep_sort(v) for k, v in sorted(o.items())}
        if isinstance(o, list):
            # Order-insensitive: believed_free is intentionally reordered.
            return sorted((deep_sort(x) for x in o), key=lambda v: json.dumps(v, sort_keys=True))
        return o

    # Same keys and values everywhere — only ordering changed.
    assert deep_sort(before) == deep_sort(s)


def test_dump_sidecar_is_idempotent_and_stable():
    s = _sidecar()
    first = dump_sidecar(s)
    # Re-dumping the already-canonical structure yields identical bytes, and
    # dumping a fresh copy of the same input lands on the same output.
    assert dump_sidecar(s) == first
    assert dump_sidecar(_sidecar()) == first


def test_canonicalize_tolerates_missing_or_malformed_blocks():
    s = {"providers": {"p": {"base_url": "x"}, "bad": "not-a-dict"}}
    canonicalize_sidecar(s)  # must not raise
    assert s["providers"]["p"] == {"base_url": "x"}
