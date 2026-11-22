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

import json

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


# ── a catalog that stopped publishing agentic_index ─────────────────────────

NO_AGENTIC = {"data": [
    {"id": "acme/big", "context_length": 262144,
     "benchmarks": {"artificial_analysis": {
         "agentic_index": None, "coding_index": None, "intelligence_index": 48.2}}},
    {"id": "acme/small:free", "context_length": 262144,
     "benchmarks": {"artificial_analysis": {
         "agentic_index": None, "coding_index": None, "intelligence_index": 12.0}}},
    {"id": "acme/unscored", "benchmarks": {"artificial_analysis": {"agentic_index": None}}},
]}


def test_intelligence_index_scores_when_agentic_is_unpublished(monkeypatch):
    """The regression: OpenRouter nulled agentic_index catalog-wide, nothing was
    scored, and flagship__free emptied. Populated indices must still rank."""
    from llmproxy.flagship import Candidate, combine_scores
    profiles = _catalog(monkeypatch, NO_AGENTIC)
    big, small = normalize_model_id("acme/big"), normalize_model_id("acme/small")
    assert profiles[big]["scores"] == {"openrouter_aa_intelligence": 48.2}
    assert profiles[small]["scores"] == {"openrouter_aa_intelligence": 12.0}
    assert profiles[normalize_model_id("acme/unscored")]["scores"] == {}
    combined = combine_scores([
        Candidate("p", "acme/big", scores=profiles[big]["scores"]),
        Candidate("p", "acme/small:free", is_free=True, scores=profiles[small]["scores"]),
    ])
    assert combined[big] > combined[small]


def test_every_populated_index_is_its_own_source(monkeypatch):
    payload = {"data": [{"id": "acme/all", "benchmarks": {"artificial_analysis": {
        "agentic_index": 50.0, "coding_index": 40.0, "intelligence_index": 30.0}}}]}
    profiles = _catalog(monkeypatch, payload)
    assert profiles[normalize_model_id("acme/all")]["scores"] == {
        "openrouter_aa": 50.0, "openrouter_aa_coding": 40.0,
        "openrouter_aa_intelligence": 30.0,
    }


# ── Epoch / LMArena sources, joins and the bundled snapshot ─────────────────

def test_display_names_join_api_ids():
    from llmproxy.flagship import display_name_key
    assert display_name_key("Gemma 4 31B IT") == normalize_model_id("google/gemma-4-31b-it:free")
    assert display_name_key("Claude Opus 5.5 (Max)") == normalize_model_id("anthropic/claude-opus-5.5")
    assert display_name_key("DeepSeek V4.1 Flash") == normalize_model_id("deepseek/deepseek-v4.1-flash")


def test_profile_for_merges_the_bare_moe_family():
    from llmproxy.flagship import profile_for
    profiles = {
        "nemotron3ultra": {"scores": {"epoch_eci": 147.0}},
        "nemotron3ultra550ba55b": {"scores": {"openrouter_aa_intelligence": 11.8},
                                   "context_length": 1000000},
    }
    got = profile_for(profiles, "nvidia/nemotron-3-ultra-550b-a55b:free")
    assert got["scores"] == {"epoch_eci": 147.0, "openrouter_aa_intelligence": 11.8}
    assert got["context_length"] == 1000000
    # A plain size is NOT stripped: Gemma 4 26B and 31B stay distinct.
    assert profile_for({"gemma4": {"scores": {"x": 1.0}}}, "google/gemma-4-31b-it") == {}


def test_epoch_eci_parses_the_export(monkeypatch):
    import io
    import zipfile

    import requests

    from llmproxy import flagship
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("epoch_capabilities_index/eci_scores.csv",
                    "Model,Display name,eci\nGLM-5.3,GLM-5.3,157.1\nBad,Bad,\n")

    class R:
        content = buf.getvalue()
        def raise_for_status(self): pass

    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    assert flagship.fetch_epoch_eci_scores() == {"glm53": 157.1}


def test_lmarena_pages_and_keeps_the_best_effort_level(monkeypatch):
    import requests

    from llmproxy import flagship
    pages = [
        {"num_rows_total": 3, "rows": [
            {"row": {"model_name": "Qwen4 Max (High)", "rating": 1400.0}},
            {"row": {"model_name": "Qwen4 Max (Low)", "rating": 1350.0}}]},
        {"num_rows_total": 3, "rows": [
            {"row": {"model_name": "GLM-5.3", "rating": 1420.0}}]},
    ]
    seen = []

    class R:
        def __init__(self, body): self.body = body
        def raise_for_status(self): pass
        def json(self): return self.body

    def fake_get(url, params=None, **k):
        seen.append(params["offset"])
        return R(pages[len(seen) - 1])

    monkeypatch.setattr(requests, "get", fake_get)
    assert flagship.fetch_lmarena_scores("text", "rating") == {"qwen4max": 1400.0, "glm53": 1420.0}
    assert seen == [0, 2]


def test_a_failed_source_falls_back_to_the_bundled_snapshot(monkeypatch, tmp_path):
    from llmproxy import flagship
    snap = tmp_path / "benchmark_scores.json"
    snap.write_text(json.dumps({"sources": {"epoch_eci": {"scores": {"glm53": 157.1}}}}))
    monkeypatch.setattr(flagship, "BENCHMARK_SNAPSHOT_PATH", snap)

    def boom():
        raise OSError("offline")

    monkeypatch.setitem(flagship._SOURCE_FETCHERS, "epoch_eci", boom)
    merged = flagship.fetch_profiles(["epoch"])  # the documented alias
    assert merged["glm53"]["scores"] == {"epoch_eci": 157.1}


def test_the_shipped_snapshot_is_loadable_and_redistributable_only():
    from llmproxy import flagship
    snap = flagship.load_benchmark_snapshot()
    assert set(snap) <= set(flagship.SNAPSHOT_SOURCES)
    assert snap.get("epoch_eci"), "the bundled snapshot must carry Epoch scores"
    raw = json.loads(flagship.BENCHMARK_SNAPSHOT_PATH.read_text())
    assert "openrouter_aa" not in raw["sources"]
    for block in raw["sources"].values():
        assert block["license"] == "CC-BY-4.0" and block["attribution"]


def test_aliases_expand():
    from llmproxy.flagship import expand_sources
    assert expand_sources(["openrouter_aa", "epoch"]) == ["openrouter_aa", "epoch_eci"]
    assert expand_sources(["lmarena", "lmarena_text"]) == ["lmarena_agent", "lmarena_text"]
