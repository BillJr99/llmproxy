"""One provider's free-tier claim must not leak onto another provider.

The bug: GMI bills for ``google/gemini-3.8-flash``, and llmproxy routed it
through ``flagship__free`` anyway. Nothing the user wrote said it was free. The
claim came from the ``google`` provider template in providers.json, which
declares ``google/gemini-3.8-flash`` in its ``believed_free`` — correct for
Google's own API, which does have a free tier for it.

``_defaults_layer`` flattens every template's list into one global list, and
the free check matched a bare upstream id against it. GMI's *upstream id* is
character-identical to Google's *qualified id*, so the bare arm matched and
GMI's copy was treated as free. That contradicts the README's own principle:
free status is a property of the routing target, not of the model.

Bare matching cannot simply be removed — a hand-written entry like
``glm-5.3-flash-free`` is documented to cover every provider. The distinction
is provenance, not spelling, and these tests pin both sides of it.
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


# The shape of the real leak, reduced.
#
# Google's own provider block serves the model as the bare `gemini-3.8-flash`,
# so the template's entry for it is the QUALIFIED `google/gemini-3.8-flash`.
# GMI namespaces its catalog by vendor, so the very same string is GMI's
# UPSTREAM id — and GMI bills for it. Two different key spaces, one identical
# sequence of characters, which is the whole collision.
GOOGLE_UPSTREAM = "gemini-3.8-flash"
TEMPLATE_ENTRY = f"google/{GOOGLE_UPSTREAM}"
GMI_UPSTREAM = TEMPLATE_ENTRY

TEMPLATES = {
    "google": {
        "believed_free": [TEMPLATE_ENTRY],
        "model_reasoning": {},
        "model_capabilities": {},
        "free_limits": {},
    },
    "gmi": {
        "believed_free": [],
        "model_reasoning": {},
        "model_capabilities": {},
        "free_limits": {},
    },
}


def _server(tmp_path: Path, monkeypatch, **cfg_over):
    cfg = {
        "providers": {
            "google": {"base_url": "http://google.example/v1", "api_key": "k"},
            "gmi": {"base_url": "http://gmi.example/v1", "api_key": "k"},
        },
        "sync_believed_free_on_startup": False,
        "server": {"log_level": "ERROR"},
        **cfg_over,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    S = _load_server(monkeypatch, tmp_path / "config.json")
    monkeypatch.setattr(S, "get_provider_free_info", lambda *a, **k: TEMPLATES)
    S._bump_routing_generation()
    return S


@pytest.fixture
def S(tmp_path: Path, monkeypatch):
    return _server(tmp_path, monkeypatch)


# ── the reported bug ────────────────────────────────────────────────────────

def test_a_gateway_does_not_inherit_another_vendors_free_tier(S):
    """The regression, and it cost real money: gmi's upstream id spells the
    same as google's qualified id, and must not be free because of it."""
    assert S._is_model_free("gmi", GMI_UPSTREAM, S.load_config()) is False


def test_the_declaring_provider_keeps_its_own_free_tier(S):
    """The other half. Scoping the claim must not throw it away — google's own
    copy is exactly what the template is vouching for."""
    assert S._is_model_free("google", GOOGLE_UPSTREAM, S.load_config()) is True


def test_the_scoped_set_carries_the_templates_entries(S):
    """The provenance set is what the rejection is decided on, so an empty one
    would make this test file pass for the wrong reason."""
    assert TEMPLATE_ENTRY in S._provider_scoped_ids(S.load_config())


# ── the behaviour that must not regress ─────────────────────────────────────

def test_a_hand_written_bare_entry_still_matches_every_provider(tmp_path, monkeypatch):
    """Documented behaviour: a person writing a bare id in config.json means
    'this model, wherever I have it'. No template declared it, so nothing
    scopes it."""
    S = _server(tmp_path, monkeypatch, believed_free=["some-shared-model"])
    cfg = S.load_config()
    assert S._is_model_free("gmi", "some-shared-model", cfg) is True
    assert S._is_model_free("google", "some-shared-model", cfg) is True


def test_a_hand_written_qualified_entry_matches_its_own_provider(tmp_path, monkeypatch):
    S = _server(tmp_path, monkeypatch, believed_free=["gmi/some-paid-model"])
    cfg = S.load_config()
    assert S._is_model_free("gmi", "some-paid-model", cfg) is True
    assert S._is_model_free("google", "some-paid-model", cfg) is False


def test_a_user_can_reclaim_a_scoped_id_by_qualifying_it(tmp_path, monkeypatch):
    """The escape hatch. If gmi really does serve it free, saying so
    explicitly is honoured — a qualified match is unambiguous and always
    wins."""
    S = _server(tmp_path, monkeypatch, believed_free=[f"gmi/{GMI_UPSTREAM}"])
    assert S._is_model_free("gmi", GMI_UPSTREAM, S.load_config()) is True


def test_an_id_containing_free_is_free_whatever_its_provenance(S):
    """The suffix rule predates all of this and is independent of it."""
    assert S._is_model_free("gmi", "qwen/qwen3.8-27b:free", S.load_config()) is True


# ── cost_observed still beats everything ────────────────────────────────────

def test_cost_observed_beats_a_providers_own_declaration(tmp_path, monkeypatch):
    """The stopgap must keep working: a real cost seen at runtime outranks any
    belief, including the declaring provider's own."""
    S = _server(tmp_path, monkeypatch,
                cost_observed_free_tier=[f"google/{GOOGLE_UPSTREAM}"])
    assert S._is_model_free("google", GOOGLE_UPSTREAM, S.load_config()) is False


def test_cost_observed_is_scoped_to_the_provider_that_billed(tmp_path, monkeypatch):
    """It is matched qualified-only, so flagging gmi's copy as paid says
    nothing about google's. Same principle, arrived at from the other side —
    and it is why this key needed no provenance work of its own."""
    S = _server(tmp_path, monkeypatch,
                cost_observed_free_tier=[f"gmi/{GMI_UPSTREAM}"])
    cfg = S.load_config()
    assert S._is_model_free("gmi", GMI_UPSTREAM, cfg) is False
    assert S._is_model_free("google", GOOGLE_UPSTREAM, cfg) is True


# ── the two entry points cannot diverge ─────────────────────────────────────

@pytest.mark.parametrize("provider,upstream", [
    ("gmi", GMI_UPSTREAM),
    ("google", GOOGLE_UPSTREAM),
    ("gmi", "qwen/qwen3.8-27b:free"),
    ("gmi", "something-nobody-mentioned"),
])
def test_the_hoisted_hot_loop_path_agrees_with_the_general_one(S, provider, upstream):
    """``_is_model_free_with`` is what the candidate walks actually call. If it
    disagreed with ``_is_model_free``, every test above would be testing a path
    production does not take."""
    cfg = S.load_config()
    assert S._is_model_free(provider, upstream, cfg) == S._is_model_free_with(
        provider, upstream,
        S._normalized_believed_free(cfg), S._normalized_cost_observed(cfg),
        S._provider_scoped_ids(cfg),
    )


def test_omitting_the_scoped_set_keeps_the_old_permissive_behaviour(S):
    """The parameter defaults to empty for callers that predate it, and an
    empty set must mean 'nothing is scoped' rather than 'nothing is free'."""
    cfg = S.load_config()
    assert S._is_model_free_with(
        "gmi", GMI_UPSTREAM,
        S._normalized_believed_free(cfg), S._normalized_cost_observed(cfg),
    ) is True


# ── the free pool itself ────────────────────────────────────────────────────

def test_the_free_candidate_pool_excludes_the_leaked_target(S, monkeypatch):
    """End to end through the selector the router actually uses, since that is
    where the charge was incurred."""
    monkeypatch.setattr(S, "_get_distinct_routes", lambda *a, **k: [
        ("gmi", GMI_UPSTREAM),
        ("google", GOOGLE_UPSTREAM),
    ])
    picked = {(pn, um) for pn, _cfg, um in S._get_free_model_candidates()}
    assert ("gmi", GMI_UPSTREAM) not in picked
    assert ("google", GOOGLE_UPSTREAM) in picked
