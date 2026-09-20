"""The `:free` variant of a model must not inherit its paid sibling's specs.

`normalize_model_id` deliberately strips billing suffixes so one benchmark score
covers every spelling of the same weights. That join is right for scores and for
a provider the catalog does not list at all; it is wrong for a variant the
catalog lists separately, because the variant is a different set of endpoints.

The shipped defect this file exists to prevent: OpenRouter lists
`z-ai/glm-5.2` with `tools` and a 1M window, and `z-ai/glm-5.2:free` with
neither. Both normalize to `glm52`, so the free variant was admitted to the
flagship tier on the paid one's specs and then failed every tool-calling request
with a non-transient upstream 404 that no amount of failover could fix.

Model names here are fictional-future so they cannot collide with a real
catalog, except where a test is deliberately pinned to the shipped case.
"""

from __future__ import annotations

import pytest

from llmproxy.flagship import (
    Candidate,
    fetch_openrouter_profiles,
    fetch_profiles,
    has_variant_suffix,
    normalize_model_id,
    passes_spec_gate,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _catalog(monkeypatch, payload: dict) -> dict[str, dict]:
    """Call the REAL fetcher with only requests.get stubbed.

    A hand-written stub would let a dead branch look alive, which is exactly how
    the bug above shipped, so the real function's output shape is what is tested.
    """
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(payload))
    return fetch_openrouter_profiles()


PAID_AND_FREE = {"data": [
    {"id": "z-ai/glm-9.9",
     "supported_parameters": ["tools", "reasoning"],
     "context_length": 1048576,
     "benchmarks": {"artificial_analysis": {"agentic_index": 61.0}}},
    {"id": "z-ai/glm-9.9:free",
     "supported_parameters": ["reasoning"],
     "context_length": 32768},
]}


# ── the variant detector ────────────────────────────────────────────────────

@pytest.mark.parametrize("model_id, expected", [
    ("z-ai/glm-9.9:free", True),
    ("z-ai/glm-9.9:batch", True),
    ("vendor/model:nitro", True),
    ("teamorouter/glm-9.9-flash-free", True),
    ("z-ai/glm-9.9", False),
    ("qwen/qwen9.9-27b", False),
    ("", False),
])
def test_variant_suffixes_are_recognised(model_id, expected):
    """A join must know which ids name a variant rather than a distinct model."""
    assert has_variant_suffix(model_id) is expected


# ── the catalog fetch ───────────────────────────────────────────────────────

def test_the_merged_profile_still_joins_across_the_billing_suffix(monkeypatch):
    """The cross-provider carry-across must survive: it is what ranks the tier."""
    profiles = _catalog(monkeypatch, PAID_AND_FREE)
    key = normalize_model_id("z-ai/glm-9.9")
    assert normalize_model_id("z-ai/glm-9.9:free") == key
    assert profiles[key]["supports_tools"] is True
    assert profiles[key]["context_length"] == 1048576
    assert profiles[key]["scores"]["openrouter_aa"] == 61.0


def test_the_catalog_records_each_exact_id_unmerged(monkeypatch):
    """Beside the joined view, what the catalog said about THIS id alone."""
    profiles = _catalog(monkeypatch, PAID_AND_FREE)
    by_id = profiles[normalize_model_id("z-ai/glm-9.9")]["by_id"]
    assert by_id["z-ai/glm-9.9"] == {"supports_tools": True, "context_length": 1048576}
    assert by_id["z-ai/glm-9.9:free"] == {"supports_tools": False, "context_length": 32768}


def test_exact_id_specs_survive_the_multi_source_merge(monkeypatch):
    """fetch_profiles merges sources; it must not flatten by_id while doing so."""
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(PAID_AND_FREE))
    merged = fetch_profiles(["openrouter_aa"])
    by_id = merged[normalize_model_id("z-ai/glm-9.9")]["by_id"]
    assert by_id["z-ai/glm-9.9:free"]["supports_tools"] is False
    assert by_id["z-ai/glm-9.9"]["supports_tools"] is True


def test_an_id_absent_from_the_catalog_has_no_exact_entry(monkeypatch):
    """Absence is what makes the merged profile the correct fallback."""
    profiles = _catalog(monkeypatch, PAID_AND_FREE)
    by_id = profiles[normalize_model_id("z-ai/glm-9.9")]["by_id"]
    assert "someprovider/glm-9.9" not in by_id


# ── the spec gate, on the specs the variant actually has ────────────────────

def _cand(model, *, tools, ctx):
    return Candidate(provider="openrouter", upstream_id=model, is_free=True,
                     context_length=ctx, supports_tools=tools,
                     scores={"openrouter_aa": 61.0})


def test_the_paid_model_passes_and_its_free_variant_does_not(monkeypatch):
    """The regression, end to end over the real catalog shape.

    Same score, same normalized key, opposite verdicts — because the gate now
    sees each id's own specs rather than the union.
    """
    profiles = _catalog(monkeypatch, PAID_AND_FREE)
    by_id = profiles[normalize_model_id("z-ai/glm-9.9")]["by_id"]

    paid = _cand("z-ai/glm-9.9", **{
        "tools": by_id["z-ai/glm-9.9"]["supports_tools"],
        "ctx": by_id["z-ai/glm-9.9"]["context_length"],
    })
    free = _cand("z-ai/glm-9.9:free", **{
        "tools": by_id["z-ai/glm-9.9:free"]["supports_tools"],
        "ctx": by_id["z-ai/glm-9.9:free"]["context_length"],
    })

    assert passes_spec_gate(paid, min_context=200000, require_tools=True) is True
    assert passes_spec_gate(free, min_context=200000, require_tools=True) is False


def test_the_variant_is_vetoed_on_context_alone_as_well(monkeypatch):
    """Two independent vetoes, so the fix does not rest on a single signal."""
    profiles = _catalog(monkeypatch, PAID_AND_FREE)
    by_id = profiles[normalize_model_id("z-ai/glm-9.9")]["by_id"]
    free = _cand("z-ai/glm-9.9:free", tools=True,  # pretend tools were fine
                 ctx=by_id["z-ai/glm-9.9:free"]["context_length"])
    assert passes_spec_gate(free, min_context=200000, require_tools=True) is False


def test_a_capable_free_variant_is_still_admitted(monkeypatch):
    """The fix must veto the incapable variant, not every `:free` id."""
    payload = {"data": [
        {"id": "qwen/qwen9.9-27b",
         "supported_parameters": ["tools"], "context_length": 1000000,
         "benchmarks": {"artificial_analysis": {"agentic_index": 55.0}}},
        {"id": "qwen/qwen9.9-27b:free",
         "supported_parameters": ["tools"], "context_length": 262144},
    ]}
    profiles = _catalog(monkeypatch, payload)
    spec = profiles[normalize_model_id("qwen/qwen9.9-27b")]["by_id"]["qwen/qwen9.9-27b:free"]
    free = _cand("qwen/qwen9.9-27b:free", tools=spec["supports_tools"], ctx=spec["context_length"])
    assert passes_spec_gate(free, min_context=200000, require_tools=True) is True
