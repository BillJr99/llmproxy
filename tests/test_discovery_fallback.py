"""A provider with no catalog endpoint is still usable via `model_filter`.

Some upstreams publish no `GET /v1/models` at all. Unbiased AI is the case that
prompted this file: an authenticated request returns `404 unknown_url`, while an
*unauthenticated* one returns `401` on every path, because the gateway checks
the key before routing — so probing without a key cannot tell you whether the
endpoint exists.

When discovery fails outright, llmproxy synthesizes the model list from
`model_filter` rather than leaving the provider contributing nothing. The
distinction that matters, and that these tests pin, is *failed* versus *empty*:
a catalog that loads and then filters down to zero is a filter result and must
be left alone, or every over-narrow filter would silently resurrect its own
models.

`fallback_models` is deliberately NOT a second mechanism for this. It was dead
schema — emitted into every template by the scaffolding script and read by
nothing — and is asserted gone here so it cannot quietly come back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

import llmproxy.server as S


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.headers = {"Content-Type": "application/json"}
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._payload


UNBIASED = {"base_url": "https://api.unbiased.ai/v1", "api_key": "k"}


def test_a_404_catalog_synthesizes_from_model_filter(monkeypatch):
    """The reported case: no catalog, so name the models yourself."""
    monkeypatch.setattr(S.requests, "get", lambda *a, **k: _Resp(
        {"error": {"message": "Not found", "code": "unknown_url"}}, status=404))
    cfg = {**UNBIASED, "model_filter": ["pareto"]}
    models = S._fetch_provider_models("unbiased-ai", cfg, 5)
    assert [m["id"] for m in models] == ["unbiased-ai__pareto"]
    assert models[0]["_route"] == ("unbiased-ai", "pareto")


def test_a_404_catalog_without_a_filter_contributes_nothing(monkeypatch):
    """Without a filter there is nothing to synthesize from, and that is correct.

    Inventing ids would put unroutable models into every pool.
    """
    monkeypatch.setattr(S.requests, "get", lambda *a, **k: _Resp(
        {"error": {"message": "Not found"}}, status=404))
    assert S._fetch_provider_models("unbiased-ai", dict(UNBIASED), 5) == []


def test_a_connection_failure_also_synthesizes(monkeypatch):
    """A catalog that is merely unreachable is a discovery failure too."""
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("no route to host")
    monkeypatch.setattr(S.requests, "get", boom)
    cfg = {**UNBIASED, "model_filter": ["pareto"]}
    assert [m["id"] for m in S._fetch_provider_models("unbiased-ai", cfg, 5)] == \
        ["unbiased-ai__pareto"]


def test_a_working_catalog_that_filters_to_zero_is_left_alone(monkeypatch):
    """GUARD: filtering to nothing is a filter result, not a discovery failure.

    Treating it as one would make an over-narrow filter resurrect the very
    models it was written to exclude.
    """
    monkeypatch.setattr(S.requests, "get", lambda *a, **k: _Resp(
        {"data": [{"id": "something-else"}, {"id": "another"}]}))
    cfg = {**UNBIASED, "model_filter": ["pareto"]}
    assert S._fetch_provider_models("unbiased-ai", cfg, 5) == []


def test_a_working_catalog_is_used_in_preference_to_the_filter(monkeypatch):
    """The filter restricts a live catalog; it does not replace it."""
    monkeypatch.setattr(S.requests, "get", lambda *a, **k: _Resp(
        {"data": [{"id": "pareto"}, {"id": "other"}]}))
    cfg = {**UNBIASED, "model_filter": ["pareto"]}
    models = S._fetch_provider_models("unbiased-ai", cfg, 5)
    assert [m["id"] for m in models] == ["unbiased-ai__pareto"]


# ── the dead key stays dead ─────────────────────────────────────────────────

def test_no_template_ships_the_dead_fallback_models_key():
    """It was in all 45 templates and read by nothing. It must not return."""
    from llmproxy.providers import load_data

    data = load_data()
    providers = data.get("providers", data)
    offenders = [k for k, v in providers.items()
                 if isinstance(v, dict) and "fallback_models" in v]
    assert offenders == []


def test_the_unbiased_template_documents_the_real_behaviour():
    """Its note claimed a working /v1/models, which sent users hunting."""
    from llmproxy.providers import load_data

    providers = load_data().get("providers", {})
    note = providers["unbiased-ai"]["_note"]
    assert "model_filter" in note
    assert "not available" in note.lower() or "unknown_url" in note


def test_the_scaffolding_script_no_longer_emits_the_dead_key():
    """add_provider.sh was the source of the key in every template."""
    text = (Path(__file__).resolve().parent.parent / "scripts" / "add_provider.sh").read_text()
    assert "fallback_models" not in text


@pytest.mark.parametrize("field", ["models_url", "models_id_field", "models_keep_task"])
def test_the_real_discovery_overrides_are_still_template_fields(field):
    """GUARD: these are the supported overrides and must keep flowing through."""
    from llmproxy.providers import _TEMPLATE_FIELDS

    assert field in _TEMPLATE_FIELDS
