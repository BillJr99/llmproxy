"""Ollama-protocol endpoints: /api/show, /api/tags, /api/ps and friends.

These routes exist because a client speaking Ollama (Hermes, Open WebUI) asks
/api/show for a model's context window. Before them the path fell through to
Flask's HTML 404 and the client silently substituted a hardcoded default, which
is worse than no answer: it is a wrong answer the client has no way to notice.

The tests below pin the two properties that make the endpoint trustworthy — a
pool advertises the window that is true of *every* member, and a fact llmproxy
does not have is omitted rather than invented.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _load_server_with_config(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def server(monkeypatch, minimal_config):
    return _load_server_with_config(monkeypatch, minimal_config)


@pytest.fixture
def client(server):
    server.app.config["TESTING"] = True
    return server.app.test_client()


def _route_real_model(server, provider="fakeprov", upstream="big-model"):
    """Put one real model in the route cache, as discovery would."""
    with server._model_route_cache_lock:
        server._model_route_cache[f"{provider}__{upstream}"] = (provider, upstream)
    return f"{provider}__{upstream}"


def _pool(server, monkeypatch, members):
    """Make every virtual model resolve to *members* [(provider, upstream), ...]."""
    cfg = {"base_url": "http://upstream.example/v1"}
    monkeypatch.setattr(
        server, "_get_virtual_candidates",
        lambda _mid: [(pn, cfg, um) for pn, um in members],
    )


# ---------------------------------------------------------------------------
# Routing: both spellings must land on the handler
# ---------------------------------------------------------------------------

def test_show_is_reachable_bare_and_under_api(client, server):
    """Hermes probes /show and /api/show; _StripApiPrefix maps the latter."""
    mid = _route_real_model(server)
    bare = client.post("/show", json={"model": mid})
    via_api = client.post("/api/show", json={"model": mid})
    assert bare.status_code == 200
    assert via_api.status_code == 200
    assert bare.get_json() == via_api.get_json()


def test_show_accepts_the_legacy_name_key(client, server):
    """Older Ollama clients send {"name": ...} rather than {"model": ...}."""
    mid = _route_real_model(server)
    assert client.post("/api/show", json={"name": mid}).status_code == 200


def test_show_without_a_model_is_a_json_400(client):
    resp = client.post("/api/show", json={})
    assert resp.status_code == 400
    assert isinstance(resp.get_json()["error"], str)


def test_unknown_model_is_a_json_404_not_html(client):
    """The whole point: a JSON client must get JSON, even when it is an error."""
    resp = client.post("/api/show", json={"model": "nope/not-a-model"})
    assert resp.status_code == 404
    assert resp.is_json
    assert "not served by this proxy" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# Context length
# ---------------------------------------------------------------------------

def test_real_model_reports_its_discovered_context(client, server, monkeypatch):
    mid = _route_real_model(server)
    monkeypatch.setattr(
        server, "_get_model_context_snapshot",
        lambda: {"fakeprov/big-model": 131072},
    )
    info = client.post("/api/show", json={"model": mid}).get_json()["model_info"]
    arch = info["general.architecture"]
    assert info[f"{arch}.context_length"] == 131072


def test_pool_reports_the_minimum_context_not_the_maximum(client, server, monkeypatch):
    """The router may dispatch to any member, and an oversized prompt is only
    demoted, never blocked — so the smallest window is the only honest answer."""
    _pool(server, monkeypatch, [("fakeprov", "small"), ("fakeprov", "big")])
    monkeypatch.setattr(
        server, "_get_model_context_snapshot",
        lambda: {"fakeprov/small": 8192, "fakeprov/big": 1000000},
    )
    info = client.post("/api/show", json={"model": "llmproxy/free"}).get_json()["model_info"]
    arch = info["general.architecture"]
    assert info[f"{arch}.context_length"] == 8192


def test_context_is_omitted_when_no_candidate_knows_it(client, server, monkeypatch):
    """Absent beats invented: the client then applies its own default knowingly."""
    _pool(server, monkeypatch, [("fakeprov", "mystery")])
    monkeypatch.setattr(server, "_get_model_context_snapshot", lambda: {})
    info = client.post("/api/show", json={"model": "llmproxy/free"}).get_json()["model_info"]
    assert not any(k.endswith(".context_length") for k in info)


def test_config_override_beats_discovery(client, server, monkeypatch):
    """model_context exists to correct an upstream that misreports its window."""
    mid = _route_real_model(server)
    monkeypatch.setattr(
        server, "_get_model_context_snapshot", lambda: {"fakeprov/big-model": 4096})
    monkeypatch.setattr(
        server, "_get_model_context", lambda _cfg: {"fakeprov/big-model": 200000})
    info = client.post("/api/show", json={"model": mid}).get_json()["model_info"]
    assert info[f"{info['general.architecture']}.context_length"] == 200000


# ---------------------------------------------------------------------------
# Capabilities and details
# ---------------------------------------------------------------------------

def test_reasoning_is_reported_as_ollamas_thinking(client, server, monkeypatch):
    mid = _route_real_model(server)
    monkeypatch.setattr(
        server, "_model_capabilities",
        lambda _cfg: {"fakeprov/big-model": {"reasoning", "tools"}},
    )
    caps = client.post("/api/show", json={"model": mid}).get_json()["capabilities"]
    assert "thinking" in caps        # not "reasoning"
    assert "reasoning" not in caps
    assert "tools" in caps
    assert "completion" in caps      # every routing target does chat


def test_json_capability_has_no_ollama_name_and_is_dropped(client, server, monkeypatch):
    mid = _route_real_model(server)
    monkeypatch.setattr(
        server, "_model_capabilities", lambda _cfg: {"fakeprov/big-model": {"json"}})
    caps = client.post("/api/show", json={"model": mid}).get_json()["capabilities"]
    assert caps == ["completion"]


def test_pool_capabilities_are_the_union(client, server, monkeypatch):
    """Capability gating steers toward a capable candidate, so a capability any
    member has is one the router can honour. (Contrast with context, which has
    no such steering and therefore takes the minimum.)"""
    _pool(server, monkeypatch, [("fakeprov", "a"), ("fakeprov", "b")])
    monkeypatch.setattr(
        server, "_model_capabilities",
        lambda _cfg: {"fakeprov/a": {"tools"}, "fakeprov/b": {"vision"}},
    )
    caps = client.post("/api/show", json={"model": "llmproxy/free"}).get_json()["capabilities"]
    assert "tools" in caps and "vision" in caps


def test_quantization_level_is_never_emitted(client, server):
    """llmproxy holds no quantization data for any model. A placeholder here
    would be a fabricated hardware claim, so the key is absent entirely."""
    mid = _route_real_model(server)
    details = client.post("/api/show", json={"model": mid}).get_json()["details"]
    assert "quantization_level" not in details


def test_parameter_size_only_when_the_id_says_so(client, server):
    sized = _route_real_model(server, upstream="llama-3.3-70b")
    unsized = _route_real_model(server, upstream="mystery-model")
    assert client.post("/api/show", json={"model": sized}) \
        .get_json()["details"]["parameter_size"] == "70B"
    assert "parameter_size" not in client.post(
        "/api/show", json={"model": unsized}).get_json()["details"]


def test_empty_pool_503s_and_names_the_cause(client, server, monkeypatch):
    """Same condition that 503s a chat request, and it must say the same thing."""
    _pool(server, monkeypatch, [])
    resp = client.post("/api/show", json={"model": "llmproxy/free"})
    assert resp.status_code == 503
    assert resp.is_json


# ---------------------------------------------------------------------------
# /api/tags, /api/ps, and the endpoints we deliberately do not implement
# ---------------------------------------------------------------------------

def test_tags_returns_the_ollama_envelope(client):
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    body = resp.get_json()
    assert isinstance(body["models"], list)
    for entry in body["models"]:
        assert entry["name"] == entry["model"]
        assert "digest" in entry and "size" in entry


def test_tags_is_reachable_bare(client):
    assert client.get("/tags").status_code == 200


def test_ps_reports_nothing_loaded(client):
    """Truthful rather than a stub: llmproxy forwards, it loads nothing."""
    assert client.get("/api/ps").get_json() == {"models": []}


@pytest.mark.parametrize("path", ["/api/pull", "/api/push", "/api/create",
                                  "/api/copy", "/api/delete"])
def test_model_management_fails_as_json_not_html(client, path):
    """Without these, Flask's HTML 404 reaches a JSON client as a parse error
    that says nothing about what went wrong."""
    resp = client.post(path, json={})
    assert resp.status_code == 404
    assert resp.is_json
    assert "routing proxy" in resp.get_json()["error"]


def test_blobs_endpoint_also_answers_json(client):
    resp = client.head("/api/blobs/sha256:abc123")
    assert resp.status_code == 404


def test_admin_is_still_not_exposed_under_api(client):
    """The new blueprint must not widen the /api surface onto /admin."""
    assert client.get("/api/admin").status_code == 404


def test_family_is_omitted_when_a_pool_spans_several(client, server, monkeypatch):
    """`family` is singular. A pool of mixed families has no single answer, so
    the list is reported and the singular key left out rather than guessed."""
    _pool(server, monkeypatch, [("fakeprov", "glm-5-flash"), ("fakeprov", "llama-3-8b")])
    details = client.post("/api/show", json={"model": "llmproxy/free"}).get_json()["details"]
    assert len(details["families"]) > 1
    assert "family" not in details


def test_family_is_reported_for_a_single_model(client, server):
    mid = _route_real_model(server, upstream="llama-3.3-70b")
    details = client.post("/api/show", json={"model": mid}).get_json()["details"]
    assert details["family"] == details["families"][0]
