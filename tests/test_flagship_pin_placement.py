"""A pin can say WHERE it belongs, not just that it belongs.

`flagship_tier.pin` admitted a model to the tier but could not place it. Nothing
scored it — that is usually why you pinned it — so it sorted below every scored
candidate (`_FLAGSHIP_UNSCORED = -1.0`) and ended up last in the failover queue.
You asked for the model and got it only after everything else had been tried.

An entry may now carry a percentile:

    "pin": [{"name": "atria-asi/Atria-Dawn-Preview", "percentile": 100}]

Bare strings still work and still mean "admit it, place it nowhere".
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from llmproxy.flagship import (
    Candidate,
    coerce_pin_percentile,
    parse_pins,
    select_flagship,
)

BASE = {
    "min_flagship_free_models": 0,
    "start_percentile": 0.9,
    "min_context": 200000,
    "require_tools": False,
    "max_models": None,
    "pin": [],
    "exclude": [],
}


def _c(provider, model, score=None, free=False, ctx=262144, tools=True):
    return Candidate(
        provider=provider, upstream_id=model, is_free=free,
        context_length=ctx, supports_tools=tools,
        scores={} if score is None else {"openrouter_aa": score},
    )


def _cfg(**over):
    return {**BASE, **over}


# ── the scale ───────────────────────────────────────────────────────────────
#
# Percentiles are [0, 1] internally, matching `start_percentile`, but "98" is
# the natural thing to write. One rule settles it: <= 1 is a fraction, > 1 is a
# percentage. No value is ambiguous between the two readings.

@pytest.mark.parametrize("raw,expected", [
    (0.98, 0.98),
    (98, 0.98),
    (100, 1.0),      # what you write to mean "first"
    (1, 1.0),        # the boundary: reads as the top of the fraction scale
    (0, 0.0),
    (0.5, 0.5),
    (50, 0.5),
])
def test_a_percentile_is_accepted_in_either_scale(raw, expected):
    assert coerce_pin_percentile(raw) == pytest.approx(expected)


def test_an_out_of_range_percentile_is_clamped_rather_than_wrapped(_=None):
    """`percentile: 980` is a typo. Clamping keeps it meaning 'first' instead
    of sorting it somewhere arbitrary."""
    assert coerce_pin_percentile(980) == 1.0


@pytest.mark.parametrize("raw", [True, False, None, "98", "", [], {}, -5])
def test_an_unusable_percentile_degrades_to_a_plain_pin(raw):
    """GUARD: a typo costs placement, not the tier. `True` matters on its own —
    bool is an int subclass, so it would otherwise read as the 100th
    percentile."""
    assert coerce_pin_percentile(raw) is None


# ── parsing the two entry shapes ────────────────────────────────────────────

def test_strings_and_objects_mix_freely_in_one_list():
    parsed = parse_pins([
        "plain/string",
        {"name": "placed/model", "percentile": 98},
        {"name": "object/no-percentile"},
    ])
    assert parsed == {
        "plain/string": None,
        "placed/model": 0.98,
        "object/no-percentile": None,
    }


@pytest.mark.parametrize("junk", [None, 42, [], {"percentile": 98}, {"name": ""}])
def test_a_malformed_entry_is_skipped_rather_than_raising(junk):
    """GUARD: a hand-edited config must not take the tier down."""
    assert parse_pins(["good/one", junk])["good/one"] is None


def test_an_empty_or_missing_pin_list_is_fine():
    assert parse_pins(None) == {} and parse_pins([]) == {}


# ── placement through selection ─────────────────────────────────────────────

def _pool():
    return [
        _c("strong", "top-model", 100),
        _c("mid", "middling", 50),
        _c("atria-asi", "Atria-Dawn-Preview"),   # unscored, the real case
    ]


def test_a_pin_without_a_percentile_still_sorts_last(_=None):
    """GUARD: the existing behaviour is the default and must not change for
    anyone who does not opt in."""
    sel = select_flagship(_pool(), _cfg(pin=["atria-asi/Atria-Dawn-Preview"]))
    assert "atria-asi/atria-dawn-preview" in sel.members
    assert "atria-asi/atria-dawn-preview" not in sel.scores


def test_a_percentile_places_an_unscored_pin(_=None):
    """The point of the feature."""
    sel = select_flagship(_pool(), _cfg(
        pin=[{"name": "atria-asi/Atria-Dawn-Preview", "percentile": 100}]))
    entry = sel.scores["atria-asi/atria-dawn-preview"]
    assert entry["combined"] == 1.0
    assert entry["pinned"] is True


def test_a_percentile_overrides_a_measured_score(_=None):
    """A pin is an explicit instruction, which is what lets it demote a model
    you distrust as well as promote one nothing has scored."""
    sel = select_flagship(_pool(), _cfg(
        pin=[{"name": "strong/top-model", "percentile": 10}]))
    assert sel.scores["strong/top-model"]["combined"] == pytest.approx(0.10)
    assert sel.pinned_placements["strong/top-model"] == pytest.approx(0.10)


def test_an_override_is_reported_so_the_caller_can_say_so(_=None):
    """Displacing real evidence is silent by design; the refresh logs it, and
    it can only do that if the selection hands the fact back."""
    sel = select_flagship(_pool(), _cfg(
        pin=[{"name": "atria-asi/Atria-Dawn-Preview", "percentile": 100}]))
    assert sel.pinned_placements == {"atria-asi/atria-dawn-preview": 1.0}


def test_a_bare_pin_places_every_provider_serving_that_model(_=None):
    """One pin, one placement, every routing target for those weights."""
    pool = [_c("strong", "top-model", 100),
            _c("a", "shared-model"), _c("b", "shared-model")]
    sel = select_flagship(pool, _cfg(
        pin=[{"name": "shared-model", "percentile": 95}]))
    assert sel.scores["a/shared-model"]["combined"] == pytest.approx(0.95)
    assert sel.scores["b/shared-model"]["combined"] == pytest.approx(0.95)


def test_an_excluded_pin_gets_no_placement(_=None):
    """GUARD: exclude still beats pin, placement or not."""
    sel = select_flagship(_pool(), _cfg(
        pin=[{"name": "atria-asi/Atria-Dawn-Preview", "percentile": 100}],
        exclude=["atria-asi/atria-dawn-preview"]))
    assert "atria-asi/atria-dawn-preview" not in sel.members
    assert "atria-asi/atria-dawn-preview" not in sel.scores


# ── placement at read time ──────────────────────────────────────────────────

def _load_server(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


ROUTES = [
    ("strong", "top-model"),
    ("atria-asi", "Atria-Dawn-Preview"),
    ("otherprov", "Atria-Dawn-Preview"),
]


def _server(tmp_path: Path, monkeypatch, pin):
    cfg = {
        "providers": {p: {"base_url": f"http://{p}.example/v1", "api_key": "k"}
                      for p, _m in ROUTES},
        "flagship_tier": {"pin": pin},
        "sync_believed_free_on_startup": False,
        "server": {"log_level": "ERROR"},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "flagship_models.json").write_text(json.dumps({
        "last_refresh_at": "2026-09-20T00:00:00+00:00",
        "members": [f"{p}/{m}".lower() for p, m in ROUTES],
        "model_scores": {},
        "scores": {"strong/top-model": {"combined": 1.0, "model_key": "topmodel"}},
    }), encoding="utf-8")
    S = _load_server(monkeypatch, tmp_path / "config.json")
    monkeypatch.setattr(S, "_get_distinct_routes", lambda *a, **k: list(ROUTES))
    return S


def test_editing_a_percentile_takes_effect_without_a_refresh(tmp_path, monkeypatch):
    """Applied at read time, matching how pins and excludes already behave —
    otherwise a placement waits out refresh_frequency_days, seven by default."""
    S = _server(tmp_path, monkeypatch,
                [{"name": "atria-asi/Atria-Dawn-Preview", "percentile": 100}])
    assert S._get_flagship_scores()["atria-asi/atria-dawn-preview"] == 1.0


def test_a_qualified_pin_does_not_move_the_same_model_on_other_providers(
    tmp_path, monkeypatch,
):
    """GUARD, and a bug this nearly shipped with.

    `normalize_model_id` strips the provider, so
    'atria-asi/Atria-Dawn-Preview' and 'Atria-Dawn-Preview' normalise to the
    SAME key. Placing a qualified pin through that key would silently move
    every provider serving those weights — exactly what naming the provider is
    meant to prevent. Pins resolve against the live route cache instead.
    """
    S = _server(tmp_path, monkeypatch,
                [{"name": "atria-asi/Atria-Dawn-Preview", "percentile": 100}])
    scores = S._get_flagship_scores()
    assert scores["atria-asi/atria-dawn-preview"] == 1.0
    assert scores.get("otherprov/atria-dawn-preview") is None


def test_a_bare_pin_does_move_every_provider(tmp_path, monkeypatch):
    """The other half: naming no provider means all of them."""
    S = _server(tmp_path, monkeypatch,
                [{"name": "Atria-Dawn-Preview", "percentile": 100}])
    scores = S._get_flagship_scores()
    assert scores["atria-asi/atria-dawn-preview"] == 1.0
    assert scores["otherprov/atria-dawn-preview"] == 1.0


def test_a_placed_pin_is_ordered_first(tmp_path, monkeypatch):
    """End to end through the ordering the router actually uses: percentile 100
    beats a model the benchmarks scored at the top."""
    S = _server(tmp_path, monkeypatch,
                [{"name": "atria-asi/Atria-Dawn-Preview", "percentile": 100}])
    pool = [(p, {}, m) for p, m in ROUTES]
    ordered = S._flagship_ordered_candidates(pool, S._get_flagship_scores(), {})
    assert (ordered[0][0], ordered[0][2]) == ("atria-asi", "Atria-Dawn-Preview")


def test_a_placed_pin_still_yields_to_its_own_cooldown(tmp_path, monkeypatch):
    """GUARD: placement is not immunity. A candidate cooling after a 402/429
    goes to the back whatever its rank, so a pin cannot spend every turn
    retrying the one model known to be rate limited."""
    S = _server(tmp_path, monkeypatch,
                [{"name": "atria-asi/Atria-Dawn-Preview", "percentile": 100}])
    monkeypatch.setattr(
        S, "_is_candidate_saturated",
        lambda pn, um, account_id=None: pn == "atria-asi")
    pool = [(p, {}, m) for p, m in ROUTES]
    ordered = S._flagship_ordered_candidates(pool, S._get_flagship_scores(), {})
    assert ordered[-1][0] == "atria-asi"


def test_a_pin_without_a_percentile_changes_nothing_at_read_time(tmp_path, monkeypatch):
    """GUARD: opting out is the default."""
    S = _server(tmp_path, monkeypatch, ["atria-asi/Atria-Dawn-Preview"])
    assert "atria-asi/atria-dawn-preview" not in S._get_flagship_scores()


def test_an_override_of_a_measured_score_is_logged_once(tmp_path, monkeypatch, caplog):
    """The override is invisible otherwise. Latched, because this runs on every
    request that builds a flagship pool."""
    S = _server(tmp_path, monkeypatch,
                [{"name": "strong/top-model", "percentile": 10}])
    S._pin_override_warned.clear()
    with caplog.at_level("INFO"):
        S._get_flagship_scores()
    assert "overriding its measured percentile" in caplog.text
    caplog.clear()
    with caplog.at_level("INFO"):
        S._get_flagship_scores()
    assert "overriding its measured percentile" not in caplog.text
