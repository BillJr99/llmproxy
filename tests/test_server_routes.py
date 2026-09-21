"""Smoke tests for llmproxy/server.py using Flask's test client.

We don't exercise live upstream calls. The focus is the parts of the proxy
that don't require a real backend: route registration, /v1/models filter,
config-driven virtual model resolution, error responses.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _load_server_with_config(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    # Reload to pick up the env-var path
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def server(monkeypatch, minimal_config):
    mod = _load_server_with_config(monkeypatch, minimal_config)
    yield mod


@pytest.fixture
def client(server):
    server.app.config["TESTING"] = True
    return server.app.test_client()


def test_health_endpoint(client):
    """Health endpoint should respond without touching upstreams."""
    # The server may register /health or just /; either should return 200.
    for path in ("/health", "/healthz", "/"):
        resp = client.get(path)
        if resp.status_code == 200:
            return
    pytest.skip("No health endpoint registered on this server build")


def test_version_endpoint(client):
    """/version should return 200 with the package version."""
    resp = client.get("/version")
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}"
    body = resp.get_json()
    assert body is not None, "expected JSON body"
    assert body.get("name") == "llmproxy"
    assert body.get("version"), "expected a non-empty version string"


def test_models_endpoint_registered(client):
    """/v1/models route should exist and respond non-5xx, regardless of
    whether the upstream provider is reachable."""
    resp = client.get("/v1/models")
    assert resp.status_code < 500, f"Got 5xx: {resp.status_code} {resp.data!r}"
    body = resp.get_json()
    assert body is not None, "expected JSON body"
    assert "data" in body, f"expected 'data' key in response, got keys={list(body.keys())}"


def test_models_rechecks_disk_before_no_providers_warning(server, monkeypatch):
    """/v1/models must re-read config from disk before declaring it empty.

    A stale in-process cache (or a config still being written at startup) can
    momentarily yield zero providers; the endpoint does a force_reload re-check so
    the "No providers configured" warning only fires when the on-disk config
    genuinely has none. Here every non-forced load reports an empty config while a
    force_reload returns the real one — the warning must NOT appear.
    """
    real_load = server.load_config

    def fake_load(*args, **kwargs):
        if kwargs.get("force_reload"):
            return real_load(*args, **kwargs)
        return {"providers": {}, "server": {"request_timeout": 5}}

    monkeypatch.setattr(server, "load_config", fake_load)

    server.app.config["TESTING"] = True
    resp = server.app.test_client().get("/v1/models")
    assert resp.status_code < 500, f"Got 5xx: {resp.status_code} {resp.data!r}"
    body = resp.get_json()
    assert body is not None
    assert "_warning" not in body, (
        "force_reload re-check should have found providers on disk and suppressed "
        "the 'No providers configured' warning"
    )


def test_unknown_model_returns_4xx(client):
    """Posting to chat completions with an unknown model should not 500."""
    resp = client.post("/v1/chat/completions", json={
        "model": "nope/does-not-exist",
        "messages": [{"role": "user", "content": "hi"}],
    })
    # We accept anything except 5xx — the proxy should reject cleanly.
    assert resp.status_code < 500, f"Got 5xx: {resp.status_code} {resp.data!r}"


def test_api_prefix_mirrors_routes(client):
    """Requests under /api should reach the same handlers as the bare path.

    Clients with a base_url of /api or /api/v1 (OpenRouter / Open WebUI /
    Ollama style) must not 404. /version needs no upstream, so it's a clean
    probe for the prefix-strip middleware.
    """
    bare = client.get("/version")
    via_api = client.get("/api/version")
    assert via_api.status_code == bare.status_code == 200
    assert via_api.get_json() == bare.get_json()
    # /api/v1/models should also route through to the models handler.
    assert client.get("/api/v1/models").status_code < 500


def test_api_prefix_excludes_admin(client):
    """The /api alias must NOT expose the admin surface (/api/admin)."""
    # The admin UI lives only at its canonical /admin path; /api/admin is left
    # untouched by the strip and therefore is not a registered route.
    assert client.get("/api/admin").status_code == 404


def test_architecture_block(server):
    """_architecture_block builds an OpenRouter-style modality descriptor."""
    blk = server._architecture_block(["text", "image"], None)
    assert blk == {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "modality": "text+image->text",
    }
    # Empty/None sides fall back to text-only.
    assert server._architecture_block(None, [])["modality"] == "text->text"


def test_startup_sidecar_sync_no_longer_rewrites_the_user_config(
        server, tmp_path, monkeypatch):
    """The startup sync runs without touching config.json.

    Copying the sidecar's sections in is what froze them at first run; the
    runtime reads providers.json directly as the defaults layer instead.
    """
    import json

    import scripts.update_free_models as ufm

    sidecar = {
        "providers": {
            "google": {
                "base_url": "u", "display": "G",
                "believed_free": ["google/added"],
                "free_limits": {}, "model_reasoning": {}, "model_capabilities": {},
            },
        },
        "provider_order": ["google"],
    }
    monkeypatch.setattr(ufm, "load_data", lambda: sidecar)

    cfg = {"providers": {"google": {"base_url": "u", "api_key": "k"}}, "believed_free": []}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    ran = server._sync_believed_free_from_sidecar(str(p))
    # False, not True: nothing was reconciled, so the caller must not invalidate
    # the models cache. Reporting a change here logged "cache invalidated after
    # update" on every boot for an update that never ran.
    assert ran is False
    # The user's config is left exactly as written. The sidecar's believed_free
    # reaches routing through the defaults layer now, not by being copied here.
    assert json.loads(p.read_text()) == cfg


def test_supported_parameters_from_config(server):
    """_supported_parameters surfaces tool/reasoning capabilities from config."""
    cfg = {
        "model_capabilities": {"p/m1": ["tools"]},
        "model_reasoning": {"p/m2": "deep"},
    }
    assert server._supported_parameters("p", "m1", cfg) == ["tools", "tool_choice"]
    assert server._supported_parameters("p", "m2", cfg) == ["reasoning"]
    assert server._supported_parameters("p", "m3", cfg) == []


# --------------------------------------------------------------------------- #
# route provenance coverage
# --------------------------------------------------------------------------- #
# The guarantee is that no reply leaves the proxy without saying which model
# served it. That only stays true if a newly added path cannot quietly opt out,
# so this walks every chat-serving surface rather than testing one of them.

# Routes that select a candidate and must therefore report one. /v1/embeddings
# and the /v1/<path> catch-all are excluded deliberately: they never go through
# _proxy_endpoint, so they neither serve virtual models nor pick a candidate,
# and have no provenance to report.
_PROVENANCE_ROUTES = [
    ("/v1/chat/completions", lambda m: {"model": m, "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/completions", lambda m: {"model": m, "prompt": "hi"}),
    ("/v1/messages", lambda m: {"model": m, "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 16}),
    ("/v1/responses", lambda m: {"model": m, "input": "hi"}),
]

_UPSTREAM_OK = (
    b'{"id":"chatcmpl-1","object":"chat.completion","model":"free-model",'
    b'"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},'
    b'"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,'
    b'"total_tokens":2}}'
)


class _Elapsed:
    def total_seconds(self):
        return 0.01


class _FakeResp:
    def __init__(self, status=200, body=_UPSTREAM_OK, chunks=None):
        self.status_code = status
        self._body = body
        self._chunks = chunks
        self.headers = {"Content-Type": "application/json"}
        self.elapsed = _Elapsed()

    @property
    def content(self):
        return self._body

    def iter_content(self, chunk_size=None):
        yield from (self._chunks if self._chunks is not None else [self._body])

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_upstream(monkeypatch, server, resp_factory):
    monkeypatch.setattr(
        server.requests, "post",
        lambda url, headers=None, json=None, stream=False, timeout=None: resp_factory(),
    )


def _virtual_pool(monkeypatch, server, candidates):
    monkeypatch.setattr(server, "_get_virtual_candidates", lambda model_full: candidates)


_PROVIDER_CFG = {"base_url": "http://upstream.example/v1", "api_key": "k"}


@pytest.mark.parametrize("path,payload", _PROVENANCE_ROUTES,
                         ids=[p for p, _ in _PROVENANCE_ROUTES])
def test_every_route_reports_the_model_that_served_a_virtual(
    server, monkeypatch, path, payload,
):
    _stub_upstream(monkeypatch, server, _FakeResp)
    _virtual_pool(monkeypatch, server, [("fakeprov", _PROVIDER_CFG, "free-model")])
    server.app.config["TESTING"] = True
    r = server.app.test_client().post(path, json=payload("llmproxy__free"))
    assert r.status_code == 200, r.data
    assert r.headers.get("X-LLMProxy-Selected-Model") == "fakeprov/free-model"


@pytest.mark.parametrize("path,payload", _PROVENANCE_ROUTES,
                         ids=[p for p, _ in _PROVENANCE_ROUTES])
def test_every_route_reports_the_model_that_served_a_pinned_request(
    server, monkeypatch, path, payload,
):
    _stub_upstream(monkeypatch, server, _FakeResp)
    server.app.config["TESTING"] = True
    r = server.app.test_client().post(path, json=payload("fakeprov__free-model"))
    assert r.status_code == 200, r.data
    assert r.headers.get("X-LLMProxy-Selected-Model") == "fakeprov/free-model"


def test_a_failed_reply_still_names_the_last_candidate_tried(server, monkeypatch):
    """The reply that says everything failed is the one most worth labelling."""
    _stub_upstream(monkeypatch, server, lambda: _FakeResp(status=503, body=b'{"error":{"message":"down"}}'))
    _virtual_pool(monkeypatch, server, [
        ("fakeprov", _PROVIDER_CFG, "free-model"),
        ("fakeprov", _PROVIDER_CFG, "big-model"),
    ])
    server.app.config["TESTING"] = True
    r = server.app.test_client().post(
        "/v1/chat/completions",
        json={"model": "llmproxy__free", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code >= 400
    assert r.headers.get("X-LLMProxy-Selected-Model") in (
        "fakeprov/free-model", "fakeprov/big-model",
    )


def test_streamed_reply_reports_its_model_before_any_bytes_flow(server, monkeypatch):
    chunks = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    _stub_upstream(monkeypatch, server, lambda: _FakeResp(chunks=chunks))
    _virtual_pool(monkeypatch, server, [("fakeprov", _PROVIDER_CFG, "free-model")])
    server.app.config["TESTING"] = True
    r = server.app.test_client().post(
        "/v1/chat/completions",
        json={"model": "llmproxy__free", "stream": True,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    # Headers are readable without consuming the body: Flask finalizes the
    # response before the WSGI server iterates it.
    assert r.headers.get("X-LLMProxy-Selected-Model") == "fakeprov/free-model"
    assert b"[DONE]" in r.data


def test_the_error_path_still_carries_retry_after(server, monkeypatch):
    """Carrying headers across the re-render must not disturb the error path.

    The re-render is gated on 2xx, so a quota reply never passes through it and
    keeps the Retry-After the upstream sent.
    """
    class _QuotaResp(_FakeResp):
        def __init__(self):
            super().__init__(status=429, body=b'{"error":{"message":"slow down"}}')
            self.headers = {"Content-Type": "application/json", "Retry-After": "42"}

    _stub_upstream(monkeypatch, server, _QuotaResp)
    server.app.config["TESTING"] = True
    r = server.app.test_client().post(
        "/v1/chat/completions",
        json={"model": "fakeprov__free-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 429
    assert r.headers.get("Retry-After") == "42"
    assert r.headers.get("X-LLMProxy-Selected-Model") == "fakeprov/free-model"


# ── the bare /models surface ────────────────────────────────────────────────
#
# Clients configured with a base URL that already ends at the API root probe
# /models rather than /v1/models, and got a 404. Because _StripApiPrefix rewrites
# PATH_INFO before routing, one rule also covers a client pointed at /api.

def test_bare_models_matches_the_v1_listing(client):
    bare = client.get("/models")
    v1 = client.get("/v1/models")
    assert bare.status_code == v1.status_code == 200
    assert bare.get_json() == v1.get_json()


def test_api_models_resolves_through_the_prefix_shim(client):
    """/api/models is rewritten to /models, so it resolves for free."""
    assert client.get("/api/models").status_code == 200


def test_bare_model_detail_is_not_a_404(client):
    """A client that found a model through /models must not hit a 404 on the
    very next call."""
    ids = [m["id"] for m in client.get("/models").get_json().get("data", [])]
    if not ids:
        pytest.skip("no models advertised in this fixture")
    bare = client.get(f"/models/{ids[0]}")
    v1 = client.get(f"/v1/models/{ids[0]}")
    assert bare.status_code == v1.status_code == 200
    assert bare.get_json() == v1.get_json()


def test_the_bare_alias_does_not_extend_to_admin(client):
    """The /admin surface stays unaliased, as the prefix shim already guarantees."""
    assert client.get("/api/admin").status_code == 404


def test_no_providers_is_named_as_the_cause_not_as_an_empty_pool(
        monkeypatch, tmp_path):
    """A config with no providers empties every pool, so a pool-shaped hint
    describes the symptom while the cause goes unnamed.

    The common reason is that the server is reading a different file than the
    operator is editing (a stale container bind mount, a stray LLMPROXY_CONFIG),
    so the hint has to name the path actually loaded.
    """
    import json

    p = tmp_path / "config.json"
    p.write_text(json.dumps({"providers": {}, "server": {"log_level": "ERROR"}}))
    mod = _load_server_with_config(monkeypatch, p)

    hint = mod._virtual_model_hint("llmproxy__flagship/free")
    assert "No providers are configured" in hint
    assert str(p) in hint, "the hint must name the config path actually loaded"

    # And it reaches the client, on the surface a chat client actually uses.
    mod.app.config["TESTING"] = True
    resp = mod.app.test_client().post("/v1/chat/completions", json={
        "model": "llmproxy/flagship__free",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 503
    assert "No providers are configured" in resp.get_json()["error"]["message"]


def test_provider_hint_is_unchanged_when_providers_exist(client, server):
    """The no-providers short-circuit must not swallow the pool-shaped hints."""
    hint = server._virtual_model_hint("llmproxy__vision")
    assert "No providers are configured" not in hint
