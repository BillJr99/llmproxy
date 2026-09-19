"""The reasoning-tier tuple has one definition; every consumer must agree.

The tuple was previously restated in four places (server, providers, the setup
wizard and a hardcoded fallback in the admin UI). Because routing rank is the
tuple *index*, and an unrecognised tier used to fall back to rank 0, drift
between those copies degraded routing silently and in the worst direction.
These tests pin the consumers to the single definition in llmproxy.providers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llmproxy import providers as _providers
from llmproxy import server as _server
from llmproxy import setup_wizard as _wizard

_ADMIN_HTML = Path(_providers.__file__).parent / "static" / "admin" / "index.html"


def test_server_uses_the_canonical_tuple():
    assert _server._REASONING_LEVELS is _providers.REASONING_LEVELS


def test_order_is_weakest_to_strongest():
    """Order is load bearing: rank is the index, so a new tier goes last."""
    levels = _providers.REASONING_LEVELS
    assert levels[:3] == ("exploratory", "standard", "deep")
    assert levels.index("flagship") > levels.index("deep")


def test_overlay_levels_are_not_hand_assignable():
    """Computed tiers must not be settable via the admin API or the wizard."""
    assert _providers.OVERLAY_REASONING_LEVELS
    assert not (_providers.VALID_REASONING_LEVELS
                & _providers.OVERLAY_REASONING_LEVELS)
    for lvl in _providers.OVERLAY_REASONING_LEVELS:
        assert lvl not in _wizard._REASONING_LEVELS


def test_wizard_picker_is_the_hand_assignable_levels_in_canonical_order():
    expected = tuple(lvl for lvl in _providers.REASONING_LEVELS
                     if lvl in _providers.VALID_REASONING_LEVELS)
    assert _wizard._REASONING_LEVELS == expected


def test_max_inferred_level_is_the_strongest_non_overlay_tier():
    """A prompt-size heuristic must never reach an overlay tier on its own."""
    idx = _server._MAX_INFERRED_LEVEL_INDEX
    assert _providers.REASONING_LEVELS[idx] == "deep"
    assert idx == len(_providers.REASONING_LEVELS) - 2


def test_admin_html_has_no_hardcoded_level_list():
    """The UI must take the tier list from the API, not a stale literal."""
    html = _ADMIN_HTML.read_text(encoding="utf-8")
    m = re.search(r"valid_reasoning_levels:\s*\[([^\]]*)\]", html)
    assert m, "admin UI no longer declares a valid_reasoning_levels fallback"
    assert m.group(1).strip() == "", (
        "admin UI hardcodes a tier list that will drift from providers.py"
    )


def test_providers_json_uses_only_hand_assignable_levels():
    """Overlay membership lives in its own block, never in model_reasoning."""
    data = json.loads(Path(_providers.DATA_PATH).read_text(encoding="utf-8"))
    for key, prov in data["providers"].items():
        for model, lvl in (prov.get("model_reasoning") or {}).items():
            assert lvl in _providers.VALID_REASONING_LEVELS, (
                f"{key}:{model} carries non-assignable tier {lvl!r}"
            )
