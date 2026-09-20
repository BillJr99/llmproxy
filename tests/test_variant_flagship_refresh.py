"""Flagship membership is gated on each routing target's OWN specs.

`_recompute_flagship_members` used to read only the merged benchmark profile,
which is keyed by normalized model. A `:free` variant therefore inherited the
tool support and context window of the paid sibling it normalizes onto, was
admitted to the tier, and then failed every tool-calling request with an
upstream 404 that failover could not resolve.

Three sources now decide, most specific first (see `_flagship_specs_for`): the
provider's own listing for this exact qualified id, the catalog's entry for this
exact id, then the merged profile. The last of those is the documented
cross-provider carry-across and must keep working, so it is guarded here too.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


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
def cfg(tmp_path: Path) -> Path:
    """One gateway serving a paid model and its `:free` variant, plus a third provider."""
    data = {
        "providers": {
            "openrouter": {"base_url": "http://openrouter.example/v1", "api_key": "k"},
            "otherprov": {"base_url": "http://otherprov.example/v1", "api_key": "k"},
        },
        "believed_free": ["openrouter/z-ai/glm-9.9:free", "otherprov/glm-9.9"],
        # The tools veto is opt-in and off by default; the tests that exercise
        # it turn it on the way a config would.
        "flagship_tier": {"require_tools": True},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def server(monkeypatch, cfg):
    s = _load_server(monkeypatch, cfg)
    with s._model_route_cache_lock:
        s._model_route_cache.clear()
        s._model_route_cache.update({
            "openrouter__z-ai/glm-9.9": ("openrouter", "z-ai/glm-9.9"),
            "openrouter__z-ai/glm-9.9:free": ("openrouter", "z-ai/glm-9.9:free"),
            "otherprov__glm-9.9": ("otherprov", "glm-9.9"),
        })
    return s


def _stub_profiles(monkeypatch, *, by_id=None):
    """Merged profile says tools + 1M, exactly as the join produces."""
    from llmproxy import flagship as _flagship

    key = _flagship.normalize_model_id("z-ai/glm-9.9")
    monkeypatch.setattr(_flagship, "fetch_profiles", lambda sources=None: {
        key: {
            "scores": {"openrouter_aa": 61.0},
            "context_length": 1048576,
            "supports_tools": True,
            "by_id": by_id or {},
        },
    })


def _members(server, cfg, monkeypatch, *, caps=None, ctx=None):
    monkeypatch.setattr(server, "_get_model_capability_snapshot",
                        lambda: {k: set(v) for k, v in (caps or {}).items()})
    monkeypatch.setattr(server, "_get_model_context_snapshot", lambda: dict(ctx or {}))
    state = server._recompute_flagship_members(server.load_config(), str(cfg))
    return set(state["members"])


# ── signal 2: the catalog's entry for the exact id ──────────────────────────

def test_the_free_variant_is_vetoed_by_the_catalogs_own_entry_for_it(server, cfg, monkeypatch):
    """The shipped bug: glm-5.2:free admitted on glm-5.2's tool support."""
    _stub_profiles(monkeypatch, by_id={
        "z-ai/glm-9.9": {"supports_tools": True, "context_length": 1048576},
        "z-ai/glm-9.9:free": {"supports_tools": False, "context_length": 32768},
    })
    members = _members(server, cfg, monkeypatch)
    assert "openrouter/z-ai/glm-9.9" in members
    assert "openrouter/z-ai/glm-9.9:free" not in members


# ── signal 1: the provider's own listing for the exact qualified id ─────────

def test_the_free_variant_is_vetoed_by_the_providers_own_listing(server, cfg, monkeypatch):
    """The live config's shape: openrouter/z-ai/glm-5.2:free tagged ["reasoning"].

    No catalog by_id at all here, so the listing is the only thing standing
    between the variant and its sibling's specs. Requires `require_tools`,
    which this fixture's config opts into: the veto is off by default.
    """
    _stub_profiles(monkeypatch, by_id={})
    members = _members(
        server, cfg, monkeypatch,
        caps={"openrouter/z-ai/glm-9.9:free": {"reasoning"},
              "openrouter/z-ai/glm-9.9": {"tools"}},
    )
    assert "openrouter/z-ai/glm-9.9" in members
    assert "openrouter/z-ai/glm-9.9:free" not in members


def test_the_providers_listing_outranks_the_catalog(server, cfg, monkeypatch):
    """A gateway is authoritative about its own routing target."""
    _stub_profiles(monkeypatch, by_id={
        "z-ai/glm-9.9:free": {"supports_tools": True, "context_length": 1048576},
    })
    members = _members(
        server, cfg, monkeypatch,
        caps={"openrouter/z-ai/glm-9.9:free": {"reasoning"}},
    )
    assert "openrouter/z-ai/glm-9.9:free" not in members


def test_a_real_context_window_from_the_listing_vetoes_on_size(server, cfg, monkeypatch):
    """The variant's window is 32k however large the joined profile claims."""
    _stub_profiles(monkeypatch, by_id={})
    members = _members(
        server, cfg, monkeypatch,
        caps={"openrouter/z-ai/glm-9.9:free": {"tools"}},
        ctx={"openrouter/z-ai/glm-9.9:free": 32768},
    )
    assert "openrouter/z-ai/glm-9.9:free" not in members


# ── the documented carry-across must not regress ────────────────────────────

def test_a_provider_the_catalog_never_lists_still_inherits_the_joined_specs(
    server, cfg, monkeypatch
):
    """The whole point of the normalized join: otherprov publishes nothing."""
    _stub_profiles(monkeypatch, by_id={
        "z-ai/glm-9.9": {"supports_tools": True, "context_length": 1048576},
        "z-ai/glm-9.9:free": {"supports_tools": False, "context_length": 32768},
    })
    members = _members(server, cfg, monkeypatch)
    assert "otherprov/glm-9.9" in members


def test_a_capable_free_variant_is_still_admitted(server, cfg, monkeypatch):
    """Vetoing every `:free` id would be a different bug, not a fix."""
    _stub_profiles(monkeypatch, by_id={
        "z-ai/glm-9.9:free": {"supports_tools": True, "context_length": 262144},
    })
    members = _members(server, cfg, monkeypatch)
    assert "openrouter/z-ai/glm-9.9:free" in members


# ── the tools veto is opt-in, and the bug stays fixed without it ────────────
#
# `require_tools` used to default on. That was the wrong layer: a request
# needing tools already cannot select a model that lacks them, per request, so
# vetoing at membership time as well only made the floating bar hunt further
# down the ranking for free models that happened to carry the tag — filling the
# tier with weaker models on the strength of a capability rather than a score.

@pytest.fixture
def cfg_shipped_defaults(tmp_path: Path) -> Path:
    """The same deployment with no flagship_tier block, i.e. shipped defaults."""
    data = {
        "providers": {
            "openrouter": {"base_url": "http://openrouter.example/v1", "api_key": "k"},
            "otherprov": {"base_url": "http://otherprov.example/v1", "api_key": "k"},
        },
        "believed_free": ["openrouter/z-ai/glm-9.9:free", "otherprov/glm-9.9"],
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def server_shipped_defaults(monkeypatch, cfg_shipped_defaults):
    s = _load_server(monkeypatch, cfg_shipped_defaults)
    with s._model_route_cache_lock:
        s._model_route_cache.clear()
        s._model_route_cache.update({
            "openrouter__z-ai/glm-9.9": ("openrouter", "z-ai/glm-9.9"),
            "openrouter__z-ai/glm-9.9:free": ("openrouter", "z-ai/glm-9.9:free"),
            "otherprov__glm-9.9": ("otherprov", "glm-9.9"),
        })
    return s


def test_lacking_tools_no_longer_vetoes_membership_by_default(
    server_shipped_defaults, cfg_shipped_defaults, monkeypatch
):
    """Membership is decided on merit; capability is decided per request."""
    _stub_profiles(monkeypatch, by_id={})
    members = _members(
        server_shipped_defaults, cfg_shipped_defaults, monkeypatch,
        caps={"openrouter/z-ai/glm-9.9:free": {"reasoning"}},
    )
    assert "openrouter/z-ai/glm-9.9:free" in members


def test_the_real_context_window_still_vetoes_without_the_tools_veto(
    server_shipped_defaults, cfg_shipped_defaults, monkeypatch
):
    """The reported 404's model is still kept out, on its 32k window.

    Two defences survive the veto becoming opt-in — the context floor at
    membership time and the capability gate at request time — and neither
    depends on `require_tools`.
    """
    _stub_profiles(monkeypatch, by_id={
        "z-ai/glm-9.9:free": {"supports_tools": False, "context_length": 32768},
    })
    members = _members(server_shipped_defaults, cfg_shipped_defaults, monkeypatch)
    assert "openrouter/z-ai/glm-9.9:free" not in members


def test_a_tools_request_still_cannot_select_a_tool_less_member(server_shipped_defaults):
    """The second defence: admitted to the tier, never picked for a tool call."""
    S = server_shipped_defaults
    caps = {"openrouter/z-ai/glm-9.9:free": {"reasoning"},
            "otherprov/glm-9.9": {"tools", "reasoning"}}
    pool = [("openrouter", {}, "z-ai/glm-9.9:free"), ("otherprov", {}, "glm-9.9")]
    kept, dropped = S._drop_known_incapable(pool, {"tools"}, caps)
    assert dropped == 1
    assert [um for _pn, _c, um in kept] == ["glm-9.9"]
