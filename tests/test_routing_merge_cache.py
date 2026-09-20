"""The merged routing config is memoized, and the memo must never go stale.

`_merged_routing_config` rebuilds four layers and copies the whole per-provider
capability snapshot. The per-route helpers each call it, twice per model, so
assembling one candidate list on a deployment serving a few thousand models cost
thousands of identical merges — tens of seconds of CPU holding the GIL, with no
log line to show for it. Memoizing collapses that to one merge per generation.

The risk a cache introduces is staleness, and a stale routing decision is the
kind that goes unnoticed: a model keeps being treated as free after a cost was
observed, or a capability learned this minute is not believed until restart. So
every invalidation source gets a test here, and they are the point of the file.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from llmproxy.config import get_routing_metadata_path


def _load_server(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def S(tmp_path: Path, monkeypatch):
    cfg = {
        "providers": {"p1": {"base_url": "http://p1.example/v1", "api_key": "k"}},
        "sync_believed_free_on_startup": False,
        "server": {"log_level": "ERROR"},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return _load_server(monkeypatch, tmp_path / "config.json")


def _caps(S, provider_model: str, caps: set[str]):
    with S._model_route_cache_lock:
        S._model_capability_cache.clear()
        S._model_capability_cache[provider_model] = set(caps)
    S._bump_routing_generation()


# ── it actually caches ──────────────────────────────────────────────────────

def test_the_merge_is_reused_within_a_generation(S):
    config = S.load_config()
    first = S._merged_routing_config(config)
    assert S._merged_routing_config(config) is first


def test_the_curated_and_uncurated_views_do_not_share_an_entry(S):
    """They differ by a whole layer; one must never be served for the other."""
    config = S.load_config()
    with_curated = S._merged_routing_config(config, include_curated=True)
    without = S._merged_routing_config(config, include_curated=False)
    assert with_curated is not without


# ── every invalidation source ───────────────────────────────────────────────

def test_a_capability_snapshot_change_is_picked_up(S):
    """The listing layer is built from it, so a rebuild must invalidate."""
    config = S.load_config()
    _caps(S, "p1/m1", {"tools"})
    assert S._merged_routing_config(config)["model_capabilities"]["p1/m1"] == ["tools"]

    _caps(S, "p1/m1", {"tools", "vision"})
    after = S._merged_routing_config(config)["model_capabilities"]["p1/m1"]
    assert sorted(after) == ["tools", "vision"]


def test_a_route_cache_rebuild_invalidates_the_memo(S, monkeypatch):
    """The real trigger in production: _rebuild_route_cache swaps the snapshot."""
    config = S.load_config()
    _caps(S, "p1/m1", {"tools"})
    before = S._merged_routing_config(config)

    monkeypatch.setattr(S, "_fetch_provider_models", lambda name, cfg, timeout: [
        {"id": "p1__m1", "supported_parameters": ["tools", "response_format"],
         "_route": ("p1", "m1")},
    ])
    S._rebuild_route_cache(config["providers"], 5)
    after = S._merged_routing_config(config)
    assert after is not before
    assert "json" in after["model_capabilities"]["p1/m1"]


def test_resetting_the_sidecar_cache_invalidates_the_memo(S):
    """The two caches are derived; dropping one without the other is a bug."""
    config = S.load_config()
    first = S._merged_routing_config(config)
    S._reset_routing_sidecar_cache()
    assert S._merged_routing_config(config) is not first


def test_a_sidecar_rewritten_on_disk_is_picked_up(S, tmp_path):
    """An edit from outside this process, caught by the path+mtime in the key."""
    config = S.load_config()
    path = get_routing_metadata_path(str(tmp_path / "config.json"))
    path.write_text(json.dumps(
        {"curated": {"model_capabilities": {"p1/m1": ["tools"]}}}), encoding="utf-8")
    os.utime(path, (1_000_000, 1_000_000))
    S._reset_routing_sidecar_cache()
    assert S._merged_routing_config(config)["model_capabilities"]["p1/m1"] == ["tools"]

    path.write_text(json.dumps(
        {"curated": {"model_capabilities": {"p1/m1": ["tools", "vision"]}}}), encoding="utf-8")
    os.utime(path, (2_000_000, 2_000_000))
    after = S._merged_routing_config(config)["model_capabilities"]["p1/m1"]
    assert sorted(after) == ["tools", "vision"]


def test_routing_keys_carried_by_config_itself_are_part_of_the_key(S):
    """An un-migrated config.json still carries these; two must not collide."""
    a = dict(S.load_config())
    a["believed_free"] = ["p1/free-a"]
    b = dict(S.load_config())
    b["believed_free"] = ["p1/free-b"]
    assert "p1/free-a" in S._merged_routing_config(a)["believed_free"]
    assert "p1/free-b" in S._merged_routing_config(b)["believed_free"]
    assert "p1/free-a" not in S._merged_routing_config(b)["believed_free"]


# ── the hoisted free check agrees with the unhoisted one ────────────────────

@pytest.mark.parametrize("provider, model, expected", [
    ("p1", "some-free-model", True),    # 'free' in the id
    ("p1", "listed-model", True),       # in believed_free
    ("p1", "paid-model", False),
    ("p1", "charged-model", False),     # cost observed beats everything
])
def test_the_hoisted_free_check_matches_the_original(S, provider, model, expected):
    """_is_model_free delegates to it, so a divergence would be silent."""
    believed = {"p1/listed-model", "p1/charged-model"}
    observed = {"p1/charged-model"}
    assert S._is_model_free_with(provider, model, believed, observed) is expected


# ── provenance survives the memo ────────────────────────────────────────────
#
# The bare-match rejection is decided on `_ROUTING_PROVIDER_SCOPED`, which is
# merged and cached alongside the five routing keys. A memo that dropped it, or
# held it stale, would silently restore the leak it exists to close.

def test_the_provenance_set_rides_along_in_the_merged_config(S):
    config = S.load_config()
    merged = S._merged_routing_config(config)
    assert isinstance(merged.get(S._ROUTING_PROVIDER_SCOPED), set)


def test_the_provenance_set_is_stable_within_a_generation(S):
    config = S.load_config()
    first = S._provider_scoped_ids(config)
    assert S._provider_scoped_ids(config) == first


def test_the_provenance_set_is_invalidated_with_everything_else(S, monkeypatch):
    """Same invalidation as the rest of the merge. A set that outlived its
    generation would keep rejecting a bare id the operator has since removed
    from the template."""
    config = S.load_config()
    assert "vendor/some-model" not in S._provider_scoped_ids(config)
    monkeypatch.setattr(S, "get_provider_free_info", lambda *a, **k: {
        "vendor": {"believed_free": ["vendor/some-model"], "model_reasoning": {},
                   "model_capabilities": {}, "free_limits": {}},
    })
    S._bump_routing_generation()
    assert "vendor/some-model" in S._provider_scoped_ids(config)


def test_the_provenance_key_is_not_treated_as_a_routing_key(S):
    """GUARD: it must stay out of the list/dict merge loops, which would union
    it as if it were `believed_free` and hand every consumer a stray key."""
    merged = S._merged_routing_config(S.load_config())
    for key in S._ROUTING_LIST_KEYS:
        assert isinstance(merged.get(key, []), list)
    for key in S._ROUTING_DICT_KEYS:
        assert isinstance(merged.get(key, {}), dict)
    assert S._ROUTING_PROVIDER_SCOPED not in S._ROUTING_LIST_KEYS
    assert S._ROUTING_PROVIDER_SCOPED not in S._ROUTING_DICT_KEYS


def test_a_hand_written_entry_never_enters_the_provenance_set(S):
    """The whole distinction: config.json is a person speaking, so it keeps
    matching every provider."""
    config = dict(S.load_config())
    config["believed_free"] = ["a-bare-model"]
    assert "a-bare-model" not in S._provider_scoped_ids(config)
