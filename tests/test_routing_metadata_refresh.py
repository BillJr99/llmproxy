"""The WRITE path of routing metadata: what a refresh learns, keeps and refuses.

tests/test_routing_metadata.py covers reading the merged layers. It never calls
``_recompute_routing_metadata`` and never stubs ``fetch_openrouter_profiles``,
and that gap let three defects ship:

  * the refresh replaced ``by_model`` wholesale, so the first run wiped every
    ``reasoning`` tier it did not itself write (620 of them, on a live box);
  * the catalog layer read ``profile["capabilities"]`` from a function that
    never set that key, so an entire source was dead while looking alive;
  * capability lookup stopped at the first matching id form, so a thin
    per-provider listing shadowed the richer joined entry and the model was
    ranked as KNOWN_INCAPABLE for something it demonstrably does.

Everything here runs against a tmp_path config dir and stubs the network.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
from pathlib import Path

import pytest

CATALOG_PROFILE_KEYS = {"scores", "context_length", "supports_tools",
                        "capabilities", "model_id", "by_id"}


# ── fixtures ────────────────────────────────────────────────────────────────
#
# Same idiom as tests/test_routing_metadata.py: reload config + server under a
# monkeypatched LLMPROXY_CONFIG so nothing touches a real config dir.

def _make_server(monkeypatch, tmp_path: Path, config: dict | None = None,
                 sidecar: dict | None = None):
    cfg = {
        "providers": {
            "alpha": {"base_url": "http://alpha.example/v1", "api_key": "k"},
            "beta": {"base_url": "http://beta.example/v1", "api_key": "k"},
        },
        "server": {"log_level": "ERROR"},
        **(config or {}),
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if sidecar is not None:
        (tmp_path / "routing_metadata.json").write_text(
            json.dumps(sidecar), encoding="utf-8")
    monkeypatch.setenv("LLMPROXY_CONFIG", str(tmp_path / "config.json"))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    server_mod._reset_routing_sidecar_cache()
    return server_mod


@pytest.fixture
def server(monkeypatch, tmp_path: Path):
    return _make_server(monkeypatch, tmp_path)


def _refresh(server_mod, monkeypatch, tmp_path: Path, *, routes=(), caps=None,
             profiles=None):
    """Run one ``_recompute_routing_metadata`` pass with every input stubbed.

    ``fetch_openrouter_profiles`` is imported inside the function under test, so
    it is patched on the flagship MODULE rather than on server.
    """
    import llmproxy.flagship as flagship_mod

    monkeypatch.setattr(flagship_mod, "fetch_openrouter_profiles",
                        lambda *a, **k: dict(profiles or {}))
    monkeypatch.setattr(server_mod, "_get_model_capability_snapshot",
                        lambda: {k: set(v) for k, v in (caps or {}).items()})
    monkeypatch.setattr(server_mod, "_get_distinct_routes", lambda: list(routes))
    return server_mod._recompute_routing_metadata(
        server_mod.load_config(), str(tmp_path / "config.json"))


def _sidecar_on_disk(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "routing_metadata.json").read_text(encoding="utf-8"))


# ── 1-2. the refresh must never lose what it did not relearn ────────────────

def test_a_curated_reasoning_tier_survives_a_refresh(monkeypatch, tmp_path):
    """If this fails, the next refresh cadence silently wipes every hand-set and
    migrated reasoning tier — the exact production incident that cost 620 tiers,
    because the pass writes capabilities and nothing carried the tiers forward.
    """
    s = _make_server(monkeypatch, tmp_path, sidecar={
        "by_model": {"glm53flash": {"reasoning": "deep",
                                    "reasoning_source": "curated"}},
    })
    state = _refresh(s, monkeypatch, tmp_path,
                     routes=[("alpha", "z-ai/glm-5.3-flash")],
                     caps={"alpha/z-ai/glm-5.3-flash": {"tools"}})

    assert state is not None
    entry = state["by_model"]["glm53flash"]
    assert entry["reasoning"] == "deep"
    assert entry["reasoning_source"] == "curated"
    # and it is what actually landed on disk, not just what was returned
    assert _sidecar_on_disk(tmp_path)["by_model"]["glm53flash"]["reasoning"] == "deep"
    # the pass still did its own job
    assert entry["capabilities"] == ["tools"]


def test_a_model_absent_from_this_pass_keeps_its_capabilities(monkeypatch, tmp_path):
    """If this fails, one provider being down at refresh time thins the routing
    data for every model only that provider served — capability-scoped pools and
    ordering quietly lose models that nothing is wrong with."""
    s = _make_server(monkeypatch, tmp_path, sidecar={
        "by_model": {
            "gemma34b": {"capabilities": ["vision", "tools"],
                         "capabilities_source": "observed",
                         "reasoning": "standard", "reasoning_source": "observed"},
        },
    })
    state = _refresh(s, monkeypatch, tmp_path,
                     routes=[("alpha", "z-ai/glm-5.3-flash")],
                     caps={"alpha/z-ai/glm-5.3-flash": {"tools"}})

    assert state is not None
    kept = state["by_model"]["gemma34b"]
    assert set(kept["capabilities"]) == {"vision", "tools"}
    assert kept["reasoning"] == "standard"
    assert "glm53flash" in state["by_model"]      # this pass still recorded its own


# ── 3-4. capabilities are evidence, so they union ───────────────────────────

def test_capabilities_union_across_providers_serving_the_same_weights(
        monkeypatch, tmp_path):
    """If this fails, the terser of two gateways serving identical weights wins
    and a real capability is erased — the model then loses ordering for requests
    it is the best available answer to."""
    s = _make_server(monkeypatch, tmp_path)
    state = _refresh(
        s, monkeypatch, tmp_path,
        routes=[("alpha", "z-ai/glm-5.3-flash"), ("beta", "glm-5.3-flash")],
        caps={
            "alpha/z-ai/glm-5.3-flash": {"tools", "json"},   # the rich listing
            "beta/glm-5.3-flash": {"tools"},                 # the thin one
        },
    )

    assert state is not None
    assert set(state["by_model"]["glm53flash"]["capabilities"]) == {"tools", "json"}
    assert state["by_model"]["glm53flash"]["capabilities_source"] == "observed"

    # and the union reaches the router for the provider whose listing was thin
    cap_map = s._model_capabilities(s.load_config())
    assert s._capability_state("beta", "glm-5.3-flash", "json", cap_map) == \
        s._CAP_KNOWN_CAPABLE


def test_a_thin_provider_listing_does_not_shadow_the_joined_entry(server):
    """If this fails, a gateway publishing a partial supported_parameters ranks
    its model BELOW an untagged one for a capability the weights demonstrably
    have — a regression to first-match-wins lookup."""
    s = server
    cap_map = {
        "prov/glm-5.3-flash": {"reasoning", "tools"},                 # thin, per provider
        "glm53flash": {"json", "reasoning", "tools", "vision"},       # joined, per weights
    }
    assert s._capability_state("prov", "glm-5.3-flash", "json", cap_map) == \
        s._CAP_KNOWN_CAPABLE
    assert s._lookup_capabilities(cap_map, "prov", "glm-5.3-flash") == \
        {"json", "reasoning", "tools", "vision"}


def test_the_three_valued_capability_logic_survives_the_union(server):
    """The union must not collapse KNOWN_INCAPABLE and UNKNOWN into one another:
    ordering puts an untagged model BETWEEN a confirmed match and a confirmed
    mismatch, and losing that distinction buries strong untagged models."""
    s = server
    cap_map = {"prov/tiny-model": {"tools"}}     # genuinely tagged, no vision

    assert s._capability_state("prov", "tiny-model", "tools", cap_map) == \
        s._CAP_KNOWN_CAPABLE
    assert s._capability_state("prov", "tiny-model", "vision", cap_map) == \
        s._CAP_KNOWN_INCAPABLE
    assert s._capability_state("prov", "never-heard-of-it", "vision", cap_map) == \
        s._CAP_UNKNOWN
    assert s._CAP_KNOWN_CAPABLE > s._CAP_UNKNOWN > s._CAP_KNOWN_INCAPABLE


# ── 5. the catalog layer is alive, measured against the REAL profile shape ──

class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _real_catalog_profiles(monkeypatch, payload: dict) -> dict[str, dict]:
    """Call the REAL fetch_openrouter_profiles with only requests.get stubbed.

    The point is that the refresh is exercised against the shape the real
    function returns. A hand-written stub that invents a ``capabilities`` key
    would make a dead branch look alive — which is precisely how the dead
    catalog layer shipped.
    """
    import requests

    from llmproxy.flagship import fetch_openrouter_profiles

    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(payload))
    return fetch_openrouter_profiles()


def test_the_catalog_layer_contributes_capabilities_in_its_real_shape(
        monkeypatch, tmp_path):
    """If this fails, the OpenRouter catalog contributes nothing and any model
    whose gateway publishes a bare OpenAI object can never earn a tag — the
    original defect, where the refresh read a key the fetch never wrote."""
    payload = {"data": [{
        "id": "z-ai/glm-5.3-flash",
        "supported_parameters": ["tools", "response_format", "reasoning"],
        "architecture": {"input_modalities": ["text", "image"]},
        "context_length": 131072,
        "benchmarks": {"artificial_analysis": {"agentic_index": 42.0}},
    }]}
    profiles = _real_catalog_profiles(monkeypatch, payload)

    # The contract the refresh depends on, asserted against the real function.
    assert set(profiles) == {"glm53flash"}
    profile = profiles["glm53flash"]
    assert set(profile) == CATALOG_PROFILE_KEYS
    assert profile["capabilities"] == {"tools", "json", "reasoning", "vision"}
    assert profile["model_id"] == "z-ai/glm-5.3-flash"

    # Now drive the refresh from exactly that. The deployment HAS a route to
    # the model but its gateway publishes no capability fields at all — the bare
    # OpenAI object this layer exists for — so whatever lands can only have come
    # from the catalog.
    s = _make_server(monkeypatch, tmp_path)
    state = _refresh(s, monkeypatch, tmp_path,
                     routes=[("beta", "glm-5.3-flash")], caps={},
                     profiles=profiles)

    assert state is not None
    entry = state["by_model"]["glm53flash"]
    assert set(entry["capabilities"]) == {"tools", "json", "reasoning", "vision"}
    assert entry["capabilities_source"] == "observed"
    # The tier is inferred from the catalog's RAW model_id, not the join key.
    assert entry["reasoning_source"] == "inferred"


def test_a_catalog_without_capabilities_teaches_nothing_about_capabilities(
        monkeypatch, tmp_path):
    """The other half of the same contract: a catalog entry that publishes no
    capability fields must leave the model UNKNOWN rather than manufacture a
    tag from its name."""
    payload = {"data": [{"id": "z-ai/glm-5.3-flash", "context_length": 131072}]}
    profiles = _real_catalog_profiles(monkeypatch, payload)
    assert profiles["glm53flash"]["capabilities"] == set()

    s = _make_server(monkeypatch, tmp_path)
    state = _refresh(s, monkeypatch, tmp_path,
                     routes=[("beta", "glm-5.3-flash")], caps={},
                     profiles=profiles)

    assert state is not None
    assert state["by_model"]["glm53flash"].get("capabilities") in (None, [])


def test_the_catalog_does_not_populate_models_this_deployment_cannot_route_to(
        monkeypatch, tmp_path):
    """The catalog describes thousands of models most deployments never serve.

    Its capabilities are wanted — as a base for the models we DO serve, and as
    family evidence — but writing a fact for every catalog entry would bloat the
    sidecar with models that have no route, and make the "N models known" log
    line mean something other than what it reads as.
    """
    payload = {"data": [
        {"id": "z-ai/glm-5.3-flash", "supported_parameters": ["tools"]},
        {"id": "some-vendor/never-served", "supported_parameters": ["tools"]},
    ]}
    profiles = _real_catalog_profiles(monkeypatch, payload)
    assert set(profiles) == {"glm53flash", "neverserved"}

    s = _make_server(monkeypatch, tmp_path)
    state = _refresh(s, monkeypatch, tmp_path,
                     routes=[("beta", "glm-5.3-flash")], caps={},
                     profiles=profiles)

    assert state is not None
    assert "glm53flash" in state["by_model"], "a served model must be recorded"
    assert "neverserved" not in state["by_model"], (
        "the catalog must not write facts about models with no route"
    )


# ── 6. every derivation reads the RAW id ────────────────────────────────────

# Measured values. The right column is what inferring from normalize_model_id's
# output would give: the separators are gone, so the regex reads digits that
# were never a parameter count.
RAW_VS_NORMALIZED_TIERS = [
    ("meta-llama/llama-3.1-8b-instruct", "exploratory", "deep"),
    ("llama-3.3-70b-versatile", "standard", "deep"),
    ("qwen3-30b-a3b", "standard", "exploratory"),
    ("gemma-3-4b-it", "exploratory", "standard"),
]


@pytest.mark.parametrize("raw,from_raw,from_normalized", RAW_VS_NORMALIZED_TIERS)
def test_reasoning_is_inferred_from_the_raw_id_not_the_join_key(
        raw, from_raw, from_normalized):
    """If this fails, an 8B model is tagged 'deep' and routed to llmproxy/deep,
    where it is the worst thing in the pool — the join key is lossy by design
    and every inference has to run before it."""
    from llmproxy.flagship import normalize_model_id
    from llmproxy.providers import infer_reasoning_level

    assert infer_reasoning_level(raw) == from_raw
    assert infer_reasoning_level(normalize_model_id(raw)) == from_normalized
    assert from_raw != from_normalized      # the table would be pointless otherwise


def test_the_refresh_infers_tiers_from_the_raw_upstream_id(monkeypatch, tmp_path):
    """The same property, end to end through the refresh rather than the helper."""
    s = _make_server(monkeypatch, tmp_path)
    state = _refresh(s, monkeypatch, tmp_path,
                     routes=[("alpha", "meta-llama/llama-3.1-8b-instruct")])

    assert state is not None
    from llmproxy.flagship import normalize_model_id
    key = normalize_model_id("meta-llama/llama-3.1-8b-instruct")
    assert state["by_model"][key]["reasoning"] == "exploratory"
    assert state["by_model"][key]["reasoning_source"] == "inferred"


# ── 7. family derivation ────────────────────────────────────────────────────

FAMILY_KEYS = [
    ("qwen/qwen3-235b-a22b", "qwen3", "qwen"),
    ("qwen/qwen3.5-flash", "qwen35", "qwen"),
    ("meta-llama/llama-2-7b-chat", "llama2", "llama"),
    ("meta-llama/llama-4-scout-17b-16e", "llama4", "llama"),
    ("z-ai/glm-5.3-flash", "glm53", "glm"),
    ("@cf/zai-org/glm-5.2", "glm52", "glm"),
    ("minimax/minimax-m2.7:free", "minimaxm27", "minimax"),
    ("openai/gpt-oss-120b", "gptoss", "gptoss"),
    ("cohere/command-a", "commanda", "commanda"),
    ("cohere/command-r", "commandr", "commandr"),
    ("cohere/command-a-reasoning", "commanda", "commanda"),
    ("deepseek-r1-distill-qwen-32b", "deepseekr1", "deepseek"),
    ("gemma-3-4b-it", "gemma3", "gemma"),
    ("mistral-7b-instruct-v0.2", "mistral", "mistral"),
]


@pytest.mark.parametrize("model_id,generation,bare", FAMILY_KEYS)
def test_family_key_groups_models_the_way_the_refresh_assumes(
        model_id, generation, bare):
    """If this fails, the family layer lends capabilities across the wrong
    boundary — either too narrowly (every model its own family, so nothing is
    learned) or too broadly (unrelated weights vouching for each other)."""
    from llmproxy.providers import family_key

    assert family_key(model_id) == generation
    assert family_key(model_id, generation=False) == bare


def test_llama2_and_llama4_are_different_generation_families():
    """This is the property that stops llama-4's tool calling from reaching
    llama-2, which would have the router hand a tool request to a model that
    cannot call tools and fail it non-transiently."""
    from llmproxy.providers import family_key

    two = family_key("meta-llama/llama-2-7b-chat")
    four = family_key("meta-llama/llama-4-scout-17b-16e")
    assert two != four
    # but the bare family still joins them, which is why it is only the fallback
    assert family_key("meta-llama/llama-2-7b-chat", generation=False) == \
        family_key("meta-llama/llama-4-scout-17b-16e", generation=False)


# ── 8. family inference needs unanimity and a quorum ────────────────────────

def test_a_family_lends_only_what_all_of_its_members_agree_on(server):
    """If this fails, one member's capability is lent to siblings that do not
    have it, and the router confidently routes a vision request to a text-only
    model."""
    observed = {
        "llama4scout": {"tools", "json"},
        "llama4maverick": {"tools", "json"},
        "llama4behemoth": {"tools"},          # no json — so json is not unanimous
    }
    raw_for_key = {
        "llama4scout": "meta-llama/llama-4-scout-17b-16e",
        "llama4maverick": "meta-llama/llama-4-maverick-17b-128e",
        "llama4behemoth": "meta-llama/llama-4-behemoth",
    }
    by_gen, by_bare = server._family_capability_profiles(observed, raw_for_key, 3)

    assert by_gen["llama4"] == {"tools"}
    assert "json" not in by_gen["llama4"]
    assert by_bare["llama"] == {"tools"}


def test_a_family_below_the_member_minimum_lends_nothing(server):
    """Two models agreeing is not evidence about a third. If this fails, a pair
    of coincidentally-similar models manufactures capability data for every
    sibling that publishes none."""
    observed = {"llama4scout": {"tools"}, "llama4maverick": {"tools"}}
    raw_for_key = {
        "llama4scout": "meta-llama/llama-4-scout-17b-16e",
        "llama4maverick": "meta-llama/llama-4-maverick-17b-128e",
    }
    by_gen, by_bare = server._family_capability_profiles(observed, raw_for_key, 3)
    assert by_gen == {} and by_bare == {}

    # ...and with the bar lowered to the member count, the same family speaks.
    by_gen, _ = server._family_capability_profiles(observed, raw_for_key, 2)
    assert by_gen["llama4"] == {"tools"}


def test_the_family_layer_lends_to_a_silent_sibling_but_never_overwrites(
        monkeypatch, tmp_path):
    """The family layer's whole purpose, end to end: a model whose gateway
    publishes nothing inherits its family's unanimous set, graded 'family' so a
    later reading outranks it."""
    s = _make_server(monkeypatch, tmp_path)
    routes = [
        ("alpha", "meta-llama/llama-4-scout-17b-16e"),
        ("alpha", "meta-llama/llama-4-maverick-17b-128e"),
        ("alpha", "meta-llama/llama-4-behemoth"),
        ("alpha", "meta-llama/llama-4-silent"),          # no listing at all
    ]
    state = _refresh(s, monkeypatch, tmp_path, routes=routes, caps={
        "alpha/meta-llama/llama-4-scout-17b-16e": {"tools", "json"},
        "alpha/meta-llama/llama-4-maverick-17b-128e": {"tools", "json"},
        "alpha/meta-llama/llama-4-behemoth": {"tools"},
    })

    assert state is not None
    from llmproxy.flagship import normalize_model_id
    silent = state["by_model"][normalize_model_id("meta-llama/llama-4-silent")]
    assert silent["capabilities"] == ["tools"]
    assert silent["capabilities_source"] == "family"
    # A member with its own reading keeps it, ungraded down to "family".
    behemoth = state["by_model"][normalize_model_id("meta-llama/llama-4-behemoth")]
    assert behemoth["capabilities_source"] == "observed"


# ── 9. provenance ordering ──────────────────────────────────────────────────

def test_a_stronger_source_is_never_overwritten_by_a_weaker_one(server):
    """If this fails, every refresh undoes the corrections made in the admin UI
    to fix what inference got wrong — the user edits the same fact forever."""
    previous = {
        "a": {"reasoning": "deep", "reasoning_source": "curated"},
        "b": {"reasoning": "deep", "reasoning_source": "observed"},
        "c": {"capabilities": ["tools"], "capabilities_source": "family"},
    }
    learned = {
        "a": {"reasoning": "exploratory", "reasoning_source": "observed"},
        "b": {"reasoning": "standard", "reasoning_source": "inferred"},
        "c": {"capabilities": ["vision"], "capabilities_source": "inferred"},
    }
    merged, counts = server._merge_model_facts(previous, learned)

    assert merged["a"]["reasoning"] == "deep"          # curated > observed
    assert merged["b"]["reasoning"] == "deep"          # observed > inferred
    assert merged["c"]["capabilities"] == ["tools"]    # family > inferred
    assert counts["refused"] == 3
    assert counts["written"] == 0


def test_an_equal_or_stronger_source_does_replace(server):
    """The other half: a fresh reading must be able to update a stale one, and a
    hand correction must be able to override anything. A refresh that can only
    refuse would freeze the metadata at whatever the first pass guessed."""
    previous = {
        "a": {"reasoning": "deep", "reasoning_source": "observed"},
        "b": {"reasoning": "deep", "reasoning_source": "inferred"},
        "c": {"capabilities": ["tools"], "capabilities_source": "observed"},
    }
    learned = {
        "a": {"reasoning": "standard", "reasoning_source": "observed"},   # equal
        "b": {"reasoning": "standard", "reasoning_source": "family"},     # stronger
        "c": {"capabilities": ["tools", "vision"],
              "capabilities_source": "curated"},                          # strongest
    }
    merged, counts = server._merge_model_facts(previous, learned)

    assert merged["a"]["reasoning"] == "standard"
    assert merged["b"]["reasoning"] == "standard"
    assert merged["b"]["reasoning_source"] == "family"
    assert merged["c"]["capabilities"] == ["tools", "vision"]
    assert counts["refused"] == 0
    assert counts["written"] == 3


def test_a_fact_with_no_recorded_source_reads_as_observed(server):
    """A sidecar written before provenance existed carries no grade. If those
    read as the WEAKEST grade, the first refresh after upgrading overwrites the
    hand-migrated data with inferences — the loss provenance exists to prevent.
    """
    from llmproxy.providers import DEFAULT_FACT_SOURCE, FACT_SOURCES, fact_rank

    assert DEFAULT_FACT_SOURCE == "observed"
    assert fact_rank(None) == fact_rank("observed") == FACT_SOURCES.index("observed")
    assert fact_rank("nonsense") == fact_rank(DEFAULT_FACT_SOURCE)

    pre_provenance = {"a": {"reasoning": "deep"}}      # no reasoning_source at all
    merged, _ = server._merge_model_facts(
        pre_provenance, {"a": {"reasoning": "standard", "reasoning_source": "inferred"}})
    assert merged["a"]["reasoning"] == "deep"

    # ...but an equally-graded reading still updates it.
    merged, _ = server._merge_model_facts(
        pre_provenance, {"a": {"reasoning": "standard", "reasoning_source": "observed"}})
    assert merged["a"]["reasoning"] == "standard"


def test_merging_keeps_models_the_learned_pass_never_mentioned(server):
    """The carry-forward rule stated directly: absent from `learned` means
    untouched, not deleted."""
    previous = {"gone": {"capabilities": ["vision"], "capabilities_source": "observed"}}
    merged, _ = server._merge_model_facts(previous, {"new": {"capabilities": ["tools"]}})
    assert merged["gone"]["capabilities"] == ["vision"]
    assert merged["new"]["capabilities"] == ["tools"]


# ── 10. the config.json migration ───────────────────────────────────────────

def test_the_migration_moves_the_five_keys_and_backs_config_up_first(
        monkeypatch, tmp_path):
    """If this fails, a user's hand-set routing facts are either lost on upgrade
    (nothing reads config.json as a layer any more) or destroyed without a
    backup to recover them from."""
    s = _make_server(monkeypatch, tmp_path, config={
        "believed_free": ["alpha/m1"],
        "cost_observed_free_tier": ["alpha/m2"],
        "model_reasoning": {"alpha/m1": "deep"},
        "model_capabilities": {"alpha/m1": ["tools"]},
        "free_limits": {"alpha/m1": {"requests_per_day": 50}},
    })
    cfg_path = str(tmp_path / "config.json")

    report = s._migrate_config_routing_keys(cfg_path)
    assert report is not None
    assert set(report) == {"believed_free", "cost_observed_free_tier",
                           "model_reasoning", "model_capabilities", "free_limits"}

    backups = list(tmp_path.glob("config.json.backup-*"))
    assert len(backups) == 1
    restored = json.loads(backups[0].read_text(encoding="utf-8"))
    assert restored["model_reasoning"] == {"alpha/m1": "deep"}   # the pre-move copy

    on_disk = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    for key in ("believed_free", "cost_observed_free_tier", "model_reasoning",
                "model_capabilities", "free_limits"):
        assert not on_disk.get(key), f"{key} was not stripped from config.json"

    curated = _sidecar_on_disk(tmp_path)["curated"]
    assert curated["believed_free"] == ["alpha/m1"]
    assert curated["cost_observed_free_tier"] == ["alpha/m2"]
    assert curated["model_reasoning"] == {"alpha/m1": "deep"}
    assert curated["model_capabilities"] == {"alpha/m1": ["tools"]}
    assert curated["free_limits"] == {"alpha/m1": {"requests_per_day": 50}}

    # The migrated facts still resolve exactly as they did from config.json.
    s._reset_routing_sidecar_cache()
    assert s._get_model_reasoning(s.load_config(cfg_path, force_reload=True))[
        "alpha/m1"] == "deep"


def test_the_migration_is_a_no_op_the_second_time(monkeypatch, tmp_path):
    """It runs on every worker start. If a second run were not a no-op it would
    take a fresh backup each boot and churn the sidecar forever."""
    s = _make_server(monkeypatch, tmp_path,
                     config={"model_reasoning": {"alpha/m1": "deep"}})
    cfg_path = str(tmp_path / "config.json")

    assert s._migrate_config_routing_keys(cfg_path) is not None
    assert s._migrate_config_routing_keys(cfg_path) is None
    assert len(list(tmp_path.glob("config.json.backup-*"))) == 1


def test_the_migration_refuses_to_drain_config_when_the_sidecar_cannot_be_saved(
        monkeypatch, tmp_path):
    """If this fails, an unwritable state directory DESTROYS the operator's
    hand-set routing facts.

    The migration's whole job is to move five keys out of config.json and into
    the sidecar. Deleting them from config.json before knowing the sidecar write
    reached disk leaves no copy of them anywhere, and the loss survives a
    restart because config.json on disk no longer carries them. That is exactly
    the state a container hits when /config or the state directory is owned by
    the wrong uid.
    """
    s = _make_server(monkeypatch, tmp_path, config={
        "model_reasoning": {"alpha/m1": "deep"},
        "believed_free": ["alpha/m1"],
    })
    cfg_path = str(tmp_path / "config.json")

    from llmproxy import config as config_mod
    monkeypatch.setattr(config_mod, "_save_state_file",
                        lambda *a, **k: False)   # every sidecar write fails

    assert s._migrate_config_routing_keys(cfg_path) is None

    on_disk = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    assert on_disk["model_reasoning"] == {"alpha/m1": "deep"}
    assert on_disk["believed_free"] == ["alpha/m1"]


def test_a_failed_config_rewrite_does_not_strip_the_shared_config_cache(
        monkeypatch, tmp_path):
    """load_config hands back its cache object itself, not a copy.

    Popping the five keys off that object would strip the snapshot every other
    reader in the process sees, and config.json is still read as the curated
    layer. With the rewrite failing, the keys have to survive in memory as well
    as on disk.
    """
    s = _make_server(monkeypatch, tmp_path, config={
        "model_reasoning": {"alpha/m1": "deep"},
    })
    cfg_path = str(tmp_path / "config.json")

    from llmproxy import config as config_mod
    monkeypatch.setattr(config_mod, "save_config", lambda *a, **k: False)
    monkeypatch.setattr(s, "save_config", lambda *a, **k: False)

    s._migrate_config_routing_keys(cfg_path)

    # Deliberately NOT force_reload: the bug was that the cached dict itself had
    # been stripped while its fingerprint still matched the untouched file, so
    # every cache-satisfied read returned the stripped config for the life of
    # the process. A forced re-read would have hidden exactly that.
    cached = s.load_config(cfg_path)
    assert cached["model_reasoning"] == {"alpha/m1": "deep"}
    on_disk = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    assert on_disk["model_reasoning"] == {"alpha/m1": "deep"}


def test_an_existing_curated_fact_beats_the_incoming_config_value(
        monkeypatch, tmp_path):
    """If this fails, a correction made in the admin UI is resurrected back to
    the older config.json value by the next restart's migration."""
    s = _make_server(
        monkeypatch, tmp_path,
        config={"model_reasoning": {"alpha/m1": "exploratory"},
                "believed_free": ["alpha/m1"]},
        sidecar={"curated": {"model_reasoning": {"alpha/m1": "deep"},
                             "believed_free": ["alpha/m9"]}},
    )
    report = s._migrate_config_routing_keys(str(tmp_path / "config.json"))

    curated = _sidecar_on_disk(tmp_path)["curated"]
    assert curated["model_reasoning"]["alpha/m1"] == "deep"      # the UI edit wins
    assert report["model_reasoning"] == 0                        # nothing new moved
    # Lists are additive rather than contested, so both survive.
    assert set(curated["believed_free"]) == {"alpha/m1", "alpha/m9"}


# ── 11. concurrent writers ──────────────────────────────────────────────────

def test_a_cost_observation_and_a_refresh_do_not_lose_each_others_writes(
        monkeypatch, tmp_path):
    """Two writers, one file. Before the shared transaction the refresh took no
    lock at all and the cost-observed path took a different one, so a model that
    started charging money could be re-marked free by a refresh that had read
    the sidecar before the observation landed — and stay in the free pool."""
    s = _make_server(monkeypatch, tmp_path, sidecar={"by_model": {}, "by_provider": {}})
    cfg_path = str(tmp_path / "config.json")

    n = 8
    start = threading.Barrier(n)
    errors: list[BaseException] = []

    def observe_cost(i: int) -> None:
        try:
            start.wait(timeout=10)
            with s._routing_sidecar_txn(cfg_path) as state:
                entry = state.setdefault("by_provider", {}).setdefault("alpha", {})
                observed = list(entry.get("cost_observed_free_tier") or [])
                observed.append(f"paid-{i}")
                entry["cost_observed_free_tier"] = observed
        except BaseException as exc:      # noqa: BLE001 — reported, not swallowed
            errors.append(exc)

    def learn_model(i: int) -> None:
        try:
            start.wait(timeout=10)
            with s._routing_sidecar_txn(cfg_path) as state:
                by_model = state.setdefault("by_model", {})
                by_model[f"learned{i}"] = {"capabilities": ["tools"],
                                           "capabilities_source": "observed"}
        except BaseException as exc:      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=observe_cost, args=(i,)) for i in range(n // 2)]
    threads += [threading.Thread(target=learn_model, args=(i,)) for i in range(n // 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a sidecar transaction deadlocked"
    assert not errors, errors

    final = _sidecar_on_disk(tmp_path)
    assert set(final["by_provider"]["alpha"]["cost_observed_free_tier"]) == \
        {f"paid-{i}" for i in range(n // 2)}
    assert set(final["by_model"]) == {f"learned{i}" for i in range(n // 2)}


# ── 12. the read cache must be keyed by path ────────────────────────────────

def test_the_sidecar_cache_is_keyed_by_path_not_mtime_alone(monkeypatch, tmp_path):
    """Two deployments' sidecars written in the same second have equal mtimes.
    If the memo keys on mtime alone, whichever was read first answers for both —
    one deployment routes on another's metadata, and in tests a tmp_path sidecar
    leaks into the next case."""
    s = _make_server(monkeypatch, tmp_path)

    alpha_dir, beta_dir = tmp_path / "a", tmp_path / "b"
    for d, caps in ((alpha_dir, ["tools"]), (beta_dir, ["vision"])):
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"providers": {}}), encoding="utf-8")
        (d / "routing_metadata.json").write_text(
            json.dumps({"by_model": {"m": {"capabilities": caps}}}), encoding="utf-8")
        os.utime(d / "routing_metadata.json", (1_700_000_000, 1_700_000_000))

    s._reset_routing_sidecar_cache()
    first = s._load_routing_sidecar(str(alpha_dir / "config.json"))
    second = s._load_routing_sidecar(str(beta_dir / "config.json"))

    assert first["by_model"]["m"]["capabilities"] == ["tools"]
    assert second["by_model"]["m"]["capabilities"] == ["vision"], \
        "the second path was served the first path's cached state"


# ── promotion into providers.json ───────────────────────────────────────────

def _providers_text() -> str:
    from pathlib import Path

    import llmproxy.providers as providers_mod
    return Path(providers_mod.DATA_PATH).read_text(encoding="utf-8")


def _promotion_server(monkeypatch, tmp_path, sidecar, routes):
    s = _make_server(monkeypatch, tmp_path)
    (tmp_path / "routing_metadata.json").write_text(json.dumps(sidecar), encoding="utf-8")
    s._reset_routing_sidecar_cache()
    monkeypatch.setattr(s, "_get_distinct_routes", lambda: routes)
    return s


def test_promotion_carries_each_fact_with_its_provenance(monkeypatch, tmp_path):
    """Inferred facts are promoted alongside observed ones, so the PR body has
    to say which is which. A wrong capability tag in providers.json routes every
    deployment's request to a model that cannot serve it."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "by_model": {
            "llama3370b": {"capabilities": ["tools", "json"],
                           "capabilities_source": "observed",
                           "reasoning": "standard", "reasoning_source": "inferred"},
            "qwen332b": {"capabilities": ["tools"], "capabilities_source": "family"},
        },
        "curated": {"model_reasoning": {"groq/qwen3-32b": "deep"}},
    }, [("groq", "llama-3.3-70b"), ("groq", "qwen3-32b")])

    text, report = s._promote_sidecar_to_providers(_providers_text())
    grades = report["providers"]["groq"]
    assert grades["observed"] >= 1 and grades["inferred"] >= 1
    assert grades["family"] >= 1 and grades["curated"] >= 1

    groq = json.loads(text)["providers"]["groq"]
    assert groq["model_capabilities"]["groq/llama-3.3-70b"] == ["json", "tools"]
    # A hand correction outranks the learned tier even here.
    assert groq["model_reasoning"]["groq/qwen3-32b"] == "deep"

    body = s._promotion_body(report)
    for grade in ("curated", "observed", "family", "inferred"):
        assert grade in body
    assert "guesses" in body, "the body must flag which entries are not provider-stated"


def test_promotion_never_invents_a_provider_this_repo_does_not_ship(
        monkeypatch, tmp_path):
    """A provider someone added locally is theirs, not a default for everyone."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "by_model": {"mything": {"capabilities": ["tools"],
                                 "capabilities_source": "observed"}},
    }, [("bills-homebrew-provider", "my-thing")])

    before = _providers_text()
    text, report = s._promote_sidecar_to_providers(before)
    assert text == before
    assert report["total"] == 0
    assert report["skipped_providers"] == ["bills-homebrew-provider"]
    assert s._promotion_body(report) == ""


def test_promotion_survives_a_malformed_providers_file(monkeypatch, tmp_path):
    """Promotion is best effort. A PR without it beats no PR at all."""
    s = _promotion_server(monkeypatch, tmp_path, {"by_model": {}}, [])
    text, report = s._promote_sidecar_to_providers("{not json")
    assert text == "{not json"
    assert report["total"] == 0


# ── substring family matching ───────────────────────────────────────────────

BY_GEN = {"glm53": {"json", "reasoning", "tools"},
          "llama32": {"tools", "vision"},
          "claudehaiku45": {"json", "reasoning", "tools", "vision"}}
BY_BARE = {"glm": {"json", "reasoning", "tools"},
           "claudehaiku": {"json", "reasoning", "tools", "vision"},
           "gpt": {"json", "tools"}, "llama": {"tools"}}


@pytest.mark.parametrize("key,raw,want_family", [
    # An exact derived family still wins, and is trusted at any length.
    ("glm53flash", "z-ai/glm-5.3-flash", "glm53"),
    # The vendor folded into the NAME rather than behind a path separator.
    # These derive zaiglm5 / zhipuglm5, families of one with nothing to lend.
    ("zaiglm5", "zai-glm-5", "glm"),
    ("zhipuglm5", "zhipu-glm-5", "glm"),
    ("cerebraszaiglm47", "cerebras/zai-glm-4.7", "glm"),
    # Regional deployments of one model: eleven of these carried no
    # capabilities on a live box while claudehaiku sat there with data.
    ("claudehaiku45useast1", "claude-haiku-4.5-us-east-1", "claudehaiku45"),
    ("claudehaiku4520251001", "claude-haiku-4.5-20251001", "claudehaiku45"),
    ("cerebrasgptoss120b", "cerebras-gpt-oss-120b", "gpt"),
])
def test_a_family_reaches_a_model_whose_id_hides_it(key, raw, want_family):
    """Deriving a family from the id only works when the vendor sits behind a
    path separator. Matching the normalized key as a substring finds the family
    that is plainly there when it does not."""
    from llmproxy import server as s
    caps, fam = s._family_capabilities_for(key, raw, BY_GEN, BY_BARE)
    assert fam == want_family, f"{key} was lent by {fam}, expected {want_family}"
    assert caps


def test_the_longest_matching_family_wins():
    """A more specific family must supersede a more general one.

    llama-3.2's vision variants do not share llama-3's capability set, so
    shortest-wins would actively mislead rather than merely under-inform.
    """
    from llmproxy import server as s
    caps, fam = s._family_capabilities_for("llama32vision11b", "llama-3.2-vision-11b",
                                           BY_GEN, BY_BARE)
    assert fam == "llama32"
    assert caps == {"tools", "vision"}, "took the broader llama family instead"

    # Same rule between a generation family and its bare form.
    _caps, fam = s._family_capabilities_for("claudehaiku45useast1",
                                            "claude-haiku-4.5-us-east-1",
                                            BY_GEN, BY_BARE)
    assert fam == "claudehaiku45"


@pytest.mark.parametrize("key,raw", [
    ("bgem3", "bge-m3"),
    ("bgererankerbase", "bge-reranker-base"),
    ("allminilml6v2", "all-minilm-l6-v2"),
    ("allmpnetbasev2", "all-mpnet-base-v2"),
    ("aura2en", "aura-2-en"),
    ("bartlargecnn", "bart-large-cnn"),
    ("aisyntheticvideodetector", "ai-synthetic-video-detector"),
])
def test_embeddings_and_speech_models_are_lent_nothing(key, raw):
    """A model with no chat capabilities must stay empty.

    This holds structurally rather than by luck: chat families are named after
    chat models, so an embedding, reranker or TTS name shares no stem with one.
    Tagging these would put a model that cannot answer into llmproxy/tools.
    """
    from llmproxy import server as s
    caps, fam = s._family_capabilities_for(key, raw, BY_GEN, BY_BARE)
    assert caps is None and fam is None, f"{key} was wrongly lent {caps} by {fam}"


def test_a_family_below_the_substring_floor_never_matches():
    """A two-character family would collide with almost anything."""
    from llmproxy import server as s
    from llmproxy.providers import FAMILY_MIN_SUBSTRING_LENGTH
    assert FAMILY_MIN_SUBSTRING_LENGTH == 3
    caps, fam = s._family_capabilities_for(
        "someunrelatedmodel", "some-unrelated-model", {}, {"ed": {"tools"}})
    assert caps is None and fam is None
    # ...but an EXACT match is trusted at any length, floor or no floor.
    caps, fam = s._family_capabilities_for("ed", "ed", {}, {"ed": {"tools"}})
    assert fam == "ed" and caps == {"tools"}


def test_substring_lending_still_records_the_family_grade(monkeypatch, tmp_path):
    """Whatever route it arrives by, a lend is a guess and must say so.

    The grade is what lets a later reading from the provider, or one click in
    the admin UI, replace it — and what stops it outranking either.
    """
    s = _make_server(monkeypatch, tmp_path)
    caps = {f"alpha/glm-5.{i}": {"reasoning", "tools"} for i in range(1, 4)}
    caps["beta/zhipu-glm-5-turbo"] = set()          # the one with no data
    state = _refresh(s, monkeypatch, tmp_path,
                     routes=[("alpha", f"glm-5.{i}") for i in range(1, 4)]
                            + [("beta", "zhipu-glm-5-turbo")],
                     caps=caps, profiles={})
    assert state is not None
    entry = state["by_model"]["zhipuglm5turbo"]
    assert set(entry["capabilities"]) == {"reasoning", "tools"}
    assert entry["capabilities_source"] == "family"


# ── cost observations cross into the catalog too ────────────────────────────
#
# `believed_free` and `cost_observed_free_tier` are a PAIR: one adds a model to
# the free pool, the other takes it back out. Only the first was promoted, so a
# deployment that discovered a provider BILLING for a supposedly-free model kept
# that correction to itself while the claim that created the problem shipped to
# everyone. That is the GMI case: real money, and every other deployment
# repeating it.

def test_a_cost_observation_is_promoted_like_a_free_one(monkeypatch, tmp_path):
    """The regression. The sidecar holds these bare; providers.json holds them
    qualified, so it must arrive prefixed exactly once."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "by_provider": {"groq": {"cost_observed_free_tier": ["llama-3.3-70b"]}},
    }, [("groq", "llama-3.3-70b")])

    text, report = s._promote_sidecar_to_providers(_providers_text())
    groq = json.loads(text)["providers"]["groq"]
    assert "groq/llama-3.3-70b" in groq["cost_observed_free_tier"]
    assert report["providers"]["groq"]["observed"] >= 1


def test_a_promoted_id_is_qualified_exactly_once(monkeypatch, tmp_path):
    """GUARD on the documented 603-entry bug: providers.json stores these
    qualified and the sidecar stores them bare, so prefixing on both sides
    produced 'google/google/...' for every entry in the file.

    Asserted on the id this test promotes rather than on every entry in the
    shipped file, because a doubled-looking prefix is not always wrong: groq
    really does serve an upstream model called `groq/compound`, whose correct
    qualified id is `groq/groq/compound`.
    """
    s = _promotion_server(monkeypatch, tmp_path, {
        "by_provider": {"groq": {"cost_observed_free_tier": ["llama-3.3-70b"],
                                 "believed_free": ["llama-3.3-70b"]}},
    }, [("groq", "llama-3.3-70b")])

    groq = json.loads(s._promote_sidecar_to_providers(_providers_text())[0])["providers"]["groq"]
    for key in ("believed_free", "cost_observed_free_tier"):
        assert "groq/llama-3.3-70b" in groq[key]
        assert "groq/groq/llama-3.3-70b" not in groq[key]
        assert "llama-3.3-70b" not in groq[key], "bare id left unqualified"


def test_the_promoted_list_is_sorted_and_accumulates(monkeypatch, tmp_path):
    """GUARD: an unstable order would churn the PR diff on every run, and
    replacing rather than accumulating would drop what the catalog already
    held."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "by_provider": {"groq": {"cost_observed_free_tier": ["zzz-model", "aaa-model"]}},
    }, [("groq", "zzz-model"), ("groq", "aaa-model")])

    groq = json.loads(s._promote_sidecar_to_providers(_providers_text())[0])["providers"]["groq"]
    assert groq["cost_observed_free_tier"] == sorted(groq["cost_observed_free_tier"])


def test_the_pr_body_names_a_cost_observation_as_one_deployments(monkeypatch, tmp_path):
    """Billing is the most deployment-specific of the five facts: trial credits
    and promotional tiers differ per account, and merging one REMOVES a model
    from everyone's free pool. The reviewer has to know which kind of claim it
    is."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "by_provider": {"groq": {"cost_observed_free_tier": ["llama-3.3-70b"]}},
    }, [("groq", "llama-3.3-70b")])

    body = s._promotion_body(s._promote_sidecar_to_providers(_providers_text())[1])
    assert "cost_observed_free_tier" in body
    assert "billing observation" in body


def test_a_cost_observation_read_back_is_provider_scoped(monkeypatch, tmp_path):
    """GUARD, and the one that would silently undo an earlier fix. A provider's
    declared id must not be matchable BARE by a different provider serving an
    upstream id that spells the same — that is how one vendor's free tier leaked
    onto a paid gateway. The new key has to join that set, not bypass it."""
    s = _make_server(monkeypatch, tmp_path)
    monkeypatch.setattr(s, "get_provider_free_info", lambda *a, **k: {
        "someprov": {"believed_free": [],
                     "cost_observed_free_tier": ["someprov/shared-model"],
                     "model_reasoning": {}, "model_capabilities": {},
                     "free_limits": {}},
    })
    s._bump_routing_generation()
    layer = s._defaults_layer()
    assert "someprov/shared-model" in layer["cost_observed_free_tier"]
    assert "someprov/shared-model" in layer[s._ROUTING_PROVIDER_SCOPED]


def test_the_four_existing_keys_are_unaffected(monkeypatch, tmp_path):
    """GUARD: this change adds a fifth key and must not perturb the others."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "by_model": {"llama3370b": {"capabilities": ["tools", "json"],
                                    "capabilities_source": "observed"}},
        "by_provider": {"groq": {"believed_free": ["llama-3.3-70b"],
                                 "free_limits": {"llama-3.3-70b":
                                                 {"requests_per_minute": 30}}}},
    }, [("groq", "llama-3.3-70b")])

    groq = json.loads(s._promote_sidecar_to_providers(_providers_text())[0])["providers"]["groq"]
    assert groq["model_capabilities"]["groq/llama-3.3-70b"] == ["json", "tools"]
    assert "groq/llama-3.3-70b" in groq["believed_free"]
    assert groq["free_limits"]["groq/llama-3.3-70b"] == {"requests_per_minute": 30}


# ── a hand-typed cost observation promotes too ──────────────────────────────
#
# The dict keys (capabilities, reasoning) are matched by LOOKUP, so a curated
# entry finds its route whichever way it was written. The list keys are the
# other way round — the entries are the input — and the per-provider loop read
# only `by_provider`. So a cost observation someone TYPED after being billed sat
# in `curated` and never crossed, while one the runtime flagger caught did.
# Typing it in is the common case: you notice the charge, you write it down.

def test_a_hand_typed_cost_observation_is_promoted(monkeypatch, tmp_path):
    """The regression. No by_provider entry for this provider at all — which is
    exactly the shape when nothing was observed at runtime."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "curated": {"cost_observed_free_tier": ["groq/llama-3.3-70b"]},
    }, [])

    text, report = s._promote_sidecar_to_providers(_providers_text())
    groq = json.loads(text)["providers"]["groq"]
    assert "groq/llama-3.3-70b" in groq["cost_observed_free_tier"]
    assert report["providers"]["groq"]["curated"] >= 1


def test_a_hand_typed_entry_is_graded_curated_not_observed(monkeypatch, tmp_path):
    """A person typing it and a runtime 402 are different kinds of claim, and
    the PR body's whole job is letting a reviewer tell them apart."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "by_provider": {"groq": {"cost_observed_free_tier": ["compound-mini"]}},
        "curated": {"cost_observed_free_tier": ["groq/llama-3.3-70b"]},
    }, [])

    grades = s._promote_sidecar_to_providers(_providers_text())[1]["providers"]["groq"]
    assert grades.get("curated", 0) >= 1
    assert grades.get("observed", 0) >= 1


def test_a_bare_curated_entry_is_not_filed_under_one_provider(monkeypatch, tmp_path):
    """GUARD. A bare curated id deliberately means 'this model wherever I have
    it' — a statement about one deployment, not about one provider. Filing it
    under a provider block would be a bigger claim than the data supports, and
    prefixing it would risk the double-qualification bug."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "curated": {"cost_observed_free_tier": ["llama-3.3-70b"]},
    }, [])

    before = _providers_text()
    text, report = s._promote_sidecar_to_providers(before)
    assert text == before
    assert report["total"] == 0


def test_a_curated_entry_for_an_unshipped_provider_is_reported(monkeypatch, tmp_path):
    """GUARD: the only-providers-this-repo-ships bound still holds when the
    provider is reached through curated rather than by_provider, and it is
    named rather than dropped silently."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "curated": {"cost_observed_free_tier": ["bills-homebrew-provider/thing"]},
    }, [])

    before = _providers_text()
    text, report = s._promote_sidecar_to_providers(before)
    assert text == before
    assert "bills-homebrew-provider" in report["skipped_providers"]


def test_a_curated_entry_is_qualified_exactly_once(monkeypatch, tmp_path):
    """GUARD: curated entries arrive ALREADY qualified, so they must be used
    verbatim. Prefixing them is the 603-entry bug."""
    s = _promotion_server(monkeypatch, tmp_path, {
        "curated": {"cost_observed_free_tier": ["groq/llama-3.3-70b"]},
    }, [])

    groq = json.loads(s._promote_sidecar_to_providers(_providers_text())[0])["providers"]["groq"]
    assert "groq/llama-3.3-70b" in groq["cost_observed_free_tier"]
    assert "groq/groq/llama-3.3-70b" not in groq["cost_observed_free_tier"]
