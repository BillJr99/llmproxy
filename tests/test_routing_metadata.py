"""Routing metadata resolves across layers instead of living in config.json.

Five keys — believed_free, cost_observed_free_tier, model_reasoning,
model_capabilities, free_limits — now come from four sources, highest first:

    config.json            your overrides, and the admin UI's
    provider listings      what each gateway says about its own models
    routing_metadata.json  what the refresh learned
    providers.json         shipped defaults

The bug that prompted this: a capability ask outranks benchmark score (which is
correct and unchanged), but the tags were missing, so the top-ranked model
scored 0 and lost to a weaker tagged one. Widening the data is the fix, and the
tests that matter most here are the ones proving the widening actually reaches
the surfaces that consume it.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


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


# ── the safety property ─────────────────────────────────────────────────────

def test_without_a_sidecar_nothing_changes(server):
    """An un-migrated deployment must behave exactly as it did before, which is
    what makes this safe to ship ahead of anyone running the migration."""
    cfg = {"believed_free": ["alpha/m"], "model_reasoning": {"alpha/m": "deep"}}
    merged = server._merged_routing_config(cfg)
    assert "alpha/m" in merged["believed_free"]
    assert merged["model_reasoning"]["alpha/m"] == "deep"


def test_a_malformed_sidecar_is_ignored_not_fatal(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"providers": {}}), encoding="utf-8")
    (tmp_path / "routing_metadata.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("LLMPROXY_CONFIG", str(tmp_path / "config.json"))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as s
    importlib.reload(s)
    s._reset_routing_sidecar_cache()
    assert isinstance(s._merged_routing_config({}), dict)


# ── precedence ──────────────────────────────────────────────────────────────

def test_config_beats_the_sidecar(monkeypatch, tmp_path):
    """A hand correction must survive every refresh — otherwise the admin
    editors would be writing something the next cadence silently undoes."""
    s = _make_server(
        monkeypatch, tmp_path,
        config={"model_reasoning": {"alpha/m": "exploratory"}},
        sidecar={"by_model": {"m": {"reasoning": "deep"}}},
    )
    assert s._get_model_reasoning(s.load_config())["alpha/m"] == "exploratory"


def test_the_sidecar_supplies_what_config_does_not(monkeypatch, tmp_path):
    s = _make_server(monkeypatch, tmp_path,
                     sidecar={"by_model": {"m": {"capabilities": ["tools"]}}})
    caps = s._model_capabilities(s.load_config())
    assert caps["m"] == {"tools"}


def test_list_keys_union_rather_than_replace(monkeypatch, tmp_path):
    """believed_free adds a model to the free pool and cost_observed removes it
    again, so additive layers still give complete control. Replacing would let
    one hand-added entry discard everything the refresh had learned."""
    s = _make_server(
        monkeypatch, tmp_path,
        config={"believed_free": ["alpha/mine"]},
        sidecar={"by_provider": {"beta": {"believed_free": ["learned"]}}},
    )
    free = s._normalized_believed_free(s.load_config())
    assert "alpha/mine" in free and "beta/learned" in free


# ── the normalized join, which is why this scales ───────────────────────────

@pytest.mark.parametrize("provider,upstream", [
    ("alpha", "z-ai/glm-5.3-flash"),
    ("beta", "zai/glm-5.3-flash"),
    ("alpha", "zai-org/glm-5.3-flash"),
    ("beta", "glm-5.3-flash"),
])
def test_one_learned_entry_covers_every_provider_spelling(
        monkeypatch, tmp_path, provider, upstream):
    """The scaling failure this exists to fix: the same weights appear under
    four spellings across a couple of dozen providers, and literal keys reach
    only the ones that happen to match."""
    s = _make_server(monkeypatch, tmp_path,
                     sidecar={"by_model": {"glm53flash": {"capabilities": ["tools"]}}})
    caps = s._model_capabilities(s.load_config())
    assert s._model_has_capability(provider, upstream, "tools", caps)


def test_an_exact_entry_still_beats_the_inherited_one(monkeypatch, tmp_path):
    """Normalized is tried LAST, so a fact about this model on this provider
    outranks one inherited from the same weights elsewhere."""
    s = _make_server(
        monkeypatch, tmp_path,
        config={"model_reasoning": {"alpha/glm-5.3-flash": "exploratory"}},
        sidecar={"by_model": {"glm53flash": {"reasoning": "deep"}}},
    )
    reasoning = s._get_model_reasoning(s.load_config())
    assert s._lookup_model_fact(reasoning, "alpha", "glm-5.3-flash") == "exploratory"
    assert s._lookup_model_fact(reasoning, "beta", "glm-5.3-flash") == "deep"


# ── the widening reaches every surface that consumes it ─────────────────────
#
# This is the actual fix for the reported bug. Capability asks outrank score
# everywhere, by design; the top-ranked model was losing because it was UNTAGGED
# and scored 0. Once the data covers it, it stops losing.

def test_a_learned_capability_reaches_the_ordering(monkeypatch, tmp_path):
    s = _make_server(monkeypatch, tmp_path,
                     sidecar={"by_model": {"glm53flash": {"capabilities": ["tools"]}}})
    caps = s._model_capabilities(s.load_config())
    # Untagged scored 0 and lost to a tagged rival; now it is known-capable.
    assert s._capability_state("alpha", "z-ai/glm-5.3-flash", "tools", caps) == \
        s._CAP_KNOWN_CAPABLE


def test_a_learned_capability_reaches_v1_models(monkeypatch, tmp_path):
    """/v1/models advertises supported_parameters for models that previously
    reported none, because it reads the same merged map."""
    s = _make_server(monkeypatch, tmp_path,
                     sidecar={"by_model": {"glm53flash": {"capabilities": ["tools"]}}})
    params = s._supported_parameters("alpha", "z-ai/glm-5.3-flash", s.load_config())
    assert "tools" in params and "tool_choice" in params


def test_a_learned_capability_reaches_the_capability_pools(monkeypatch, tmp_path):
    """llmproxy/tools and llmproxy/vision admit more models for the same reason."""
    s = _make_server(monkeypatch, tmp_path,
                     sidecar={"by_model": {"glm53flash": {"capabilities": ["tools"]}}})
    with s._model_route_cache_lock:
        s._model_route_cache.clear()
        s._model_route_cache["alpha__x"] = ("alpha", "z-ai/glm-5.3-flash")
    got = [(pn, um) for pn, _pc, um in s._get_capability_model_candidates("tools")]
    assert got == [("alpha", "z-ai/glm-5.3-flash")]


def test_a_learned_reasoning_tier_reaches_the_reasoning_pools(monkeypatch, tmp_path):
    s = _make_server(monkeypatch, tmp_path,
                     sidecar={"by_model": {"glm53flash": {"reasoning": "deep"}}})
    with s._model_route_cache_lock:
        s._model_route_cache.clear()
        s._model_route_cache["alpha__x"] = ("alpha", "z-ai/glm-5.3-flash")
    got = [(pn, um) for pn, _pc, um in s._get_reasoning_model_candidates("deep")]
    assert got == [("alpha", "z-ai/glm-5.3-flash")]


# ── the refresh cadence ─────────────────────────────────────────────────────

def test_the_cadence_gates_the_refresh(server, tmp_path):
    import datetime as _dt
    now = _dt.datetime.now(_dt.UTC).isoformat()
    stale = (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=99)).isoformat()
    cfgp = str(tmp_path / "config.json")
    meta = {"enabled": True, "refresh_frequency_days": 7}

    assert server._routing_metadata_due(meta, cfgp) is True          # no state yet
    (tmp_path / "routing_metadata.json").write_text(
        json.dumps({"last_refresh_at": now}), encoding="utf-8")
    assert server._routing_metadata_due(meta, cfgp) is False         # throttled
    (tmp_path / "routing_metadata.json").write_text(
        json.dumps({"last_refresh_at": stale}), encoding="utf-8")
    assert server._routing_metadata_due(meta, cfgp) is True          # stale
    assert server._routing_metadata_due({"enabled": False}, cfgp) is False


def test_the_sidecar_is_reread_when_it_changes(monkeypatch, tmp_path):
    """providers.json is cached for the process lifetime, which is right for a
    shipped default and wrong here — the refresh rewrites this file while the
    server runs."""
    s = _make_server(monkeypatch, tmp_path,
                     sidecar={"by_model": {"m": {"capabilities": ["tools"]}}})
    assert s._model_capabilities(s.load_config())["m"] == {"tools"}

    import os
    import time
    path = tmp_path / "routing_metadata.json"
    path.write_text(json.dumps({"by_model": {"m": {"capabilities": ["vision"]}}}),
                    encoding="utf-8")
    future = time.time() + 10
    os.utime(path, (future, future))
    assert s._model_capabilities(s.load_config())["m"] == {"vision"}
