"""GET /v1/providers and GET /v1/config: read-only, unauthenticated, no secrets.

Neither route existed before. A client asking llmproxy about *itself* fell
through to the OpenAI passthrough, which demands `?provider=<name>` to know
which upstream to forward to, so the reply was `400 Supply '?provider=<name>'`
— which reads like a malformed request rather than like a missing endpoint.

These endpoints answer to anyone who can reach the port, so the constraint that
matters most here is that no credential appears in either response, in any
form. Masks are not good enough: a mask still leaks its last characters. The
tests below are written to fail loudly if that ever stops being true.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

API_KEY = "sk-thisisthesecretvalue-9876"
ADMIN_TOKEN = "admin-token-do-not-publish"


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
        "providers": {
            "paidprov": {
                "base_url": "http://paid.example/v1",
                "api_key": API_KEY,
                "model_filter": ["keep-me"],
            },
            "localprov": {"base_url": "http://127.0.0.1:11434/v1"},
            "hiddenprov": {
                "base_url": "http://hidden.example/v1",
                "api_key": API_KEY,
                "expose_to_virtual_models": False,
            },
        },
        "admin": {"token": ADMIN_TOKEN},
        "sync_believed_free_on_startup": False,
        "server": {"log_level": "ERROR", "request_log": "metadata"},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    server = _load_server(monkeypatch, tmp_path / "config.json")
    # No network: the route cache is seeded rather than fetched.
    monkeypatch.setattr(server, "_get_distinct_routes", lambda *a, **k: [
        ("paidprov", "model-a"),
        ("paidprov", "model-b"),
        ("hiddenprov", "model-c"),
    ])
    return server


@pytest.fixture
def client(S):
    return S.app.test_client()


# ── the endpoints exist at all ──────────────────────────────────────────────

@pytest.mark.parametrize("path", ["/v1/providers", "/v1/config",
                                  "/providers", "/config"])
def test_the_endpoint_answers_instead_of_demanding_a_provider_param(client, path):
    """The regression: these used to 400 with "Supply '?provider=<name>'"
    because no route matched and the passthrough caught them."""
    resp = client.get(path)
    assert resp.status_code == 200
    assert resp.get_json()["object"].startswith("llmproxy.")


# ── no credentials, in any form ─────────────────────────────────────────────

@pytest.mark.parametrize("path", ["/v1/providers", "/v1/config"])
def test_no_api_key_appears_anywhere_in_the_response(client, path):
    """The constraint that matters. Unauthenticated endpoint, real key in
    config, and not even a masked fragment may appear."""
    body = client.get(path).get_data(as_text=True)
    assert API_KEY not in body
    # A mask would leak the tail, which is exactly what is being ruled out.
    assert API_KEY[-4:] not in body


def test_no_admin_token_appears_in_the_config_response(client):
    body = client.get("/v1/config").get_data(as_text=True)
    assert ADMIN_TOKEN not in body
    assert ADMIN_TOKEN[-4:] not in body


def test_whether_a_credential_is_set_is_still_reported(client):
    """The safe form, and the only fact about a key worth publishing."""
    by_name = {p["name"]: p for p in client.get("/v1/providers").get_json()["providers"]}
    assert by_name["paidprov"]["api_key_set"] is True
    assert by_name["localprov"]["api_key_set"] is False
    assert client.get("/v1/config").get_json()["admin"]["token_set"] is True


def test_a_secret_shaped_key_in_the_server_block_is_dropped(tmp_path, monkeypatch):
    """GUARD on the belt-and-braces filter. The server block carries no secret
    today; this pins the behaviour for the one someone adds later."""
    cfg = {
        "providers": {},
        "sync_believed_free_on_startup": False,
        "server": {"log_level": "ERROR", "upstream_secret": "leak-me-please"},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    S = _load_server(monkeypatch, tmp_path / "config.json")
    body = S.app.test_client().get("/v1/config").get_data(as_text=True)
    assert "leak-me-please" not in body


@pytest.mark.parametrize("name,secret", [
    ("api_key", True), ("admin_token", True), ("client_secret", True),
    ("db_password", True), ("aws_credential", True),
    ("api_key_set", False), ("base_url_is_env", False), ("token_count", False),
    ("log_level", False), ("request_timeout", False),
])
def test_the_secret_name_filter_keeps_derived_booleans(S, name, secret):
    """`api_key_set` is a bool DERIVED from a secret, which is the safe form
    and the whole point of publishing it. Catching it would defeat the field."""
    assert S._looks_secret(name) is secret


# ── the content is actually useful ──────────────────────────────────────────

def test_providers_reports_how_many_models_each_one_serves(client):
    by_name = {p["name"]: p for p in client.get("/v1/providers").get_json()["providers"]}
    assert by_name["paidprov"]["models"] == 2
    assert by_name["localprov"]["models"] == 0


def test_providers_surfaces_the_two_switches_that_explain_a_missing_pool(client):
    """"Why is this provider not in my free pool" is nearly always one of
    these two, so both are reported rather than inferred."""
    by_name = {p["name"]: p for p in client.get("/v1/providers").get_json()["providers"]}
    assert by_name["hiddenprov"]["expose_to_virtual_models"] is False
    assert by_name["localprov"]["local"] is True
    assert by_name["paidprov"]["local"] is False


def test_config_reports_the_effective_routing_sizes_not_config_json(client):
    """Effective, not config.json's contents: the routing keys are merged from
    four layers, and this deployment writes none of them by hand. Reading
    config.json alone would report zero of everything."""
    meta = client.get("/v1/config").get_json()["routing_metadata"]
    assert meta["believed_free"] > 0
    assert meta["model_reasoning"] > 0


def test_config_reports_the_settings_most_often_being_checked(client):
    body = client.get("/v1/config").get_json()
    assert body["request_log"] == "metadata"
    assert body["free_tier_cache_affinity"] is False
    assert body["flagship_tier"]["min_flagship_free_models"] >= 0
    assert body["providers"] == ["hiddenprov", "localprov", "paidprov"]


# ── the passthrough is not broken ───────────────────────────────────────────

@pytest.mark.parametrize("path", ["/v1/providers", "/v1/config"])
def test_an_explicit_provider_param_still_passes_through(client, path, S, monkeypatch):
    """GUARD: anyone calling an UPSTREAM's own /v1/providers through llmproxy
    was relying on the passthrough. Adding a static route must not shadow it."""
    seen = {}

    def _fake(method, url, **kwargs):
        seen["url"] = url

        class _R:
            content = b'{"upstream": true}'
            status_code = 200
            headers = {"Content-Type": "application/json"}
        return _R()

    monkeypatch.setattr(S.requests, "request", _fake)
    resp = client.get(f"{path}?provider=paidprov")
    assert resp.status_code == 200
    assert resp.get_json() == {"upstream": True}
    assert seen["url"].startswith("http://paid.example/v1/")


def test_an_unknown_provider_param_still_404s(client):
    """GUARD: the passthrough's own error path is unchanged."""
    assert client.get("/v1/providers?provider=nosuchprovider").status_code == 404
