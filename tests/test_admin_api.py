"""Integration tests for the web admin API (llmproxy/admin.py).

Exercised through Flask's test client against a temp config file, mirroring the
reload pattern in test_server_routes.py. No live upstreams are contacted; model
discovery is monkeypatched where needed.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

BASE_CONFIG = {
    "providers": {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-supersecretkey123",
            "model_filter": None,
        },
    },
    "believed_free": ["openai/free-thing"],
    "model_reasoning": {"openai/gpt-x": "deep"},
    "model_capabilities": {"openai/gpt-x": ["tools"]},
    "free_limits": {},
    "server": {
        "host": "127.0.0.1",
        "port": 8080,
        "log_level": "ERROR",
        "request_timeout": 5,
        "stream_timeout": 5,
    },
}


def _make_server(monkeypatch, config_path: Path, config: dict):
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    server_mod.app.config["TESTING"] = True
    return server_mod


@pytest.fixture
def cfg_path(tmp_path) -> Path:
    return tmp_path / "config.json"


@pytest.fixture
def client(monkeypatch, cfg_path):
    server_mod = _make_server(monkeypatch, cfg_path, dict(BASE_CONFIG))
    return server_mod.app.test_client()


def _read_config(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Read

def test_admin_index_served(client):
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert b"llmproxy" in resp.data


def test_get_config(client):
    resp = client.get("/admin/api/config")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "openai" in body["providers"]
    assert body["believed_free"] == ["openai/free-thing"]
    assert body["server"]["port"] == 8080
    assert "exploratory" in body["valid_reasoning_levels"]


# --------------------------------------------------------------------------- #
# Server settings

def test_put_server_updates(client, cfg_path):
    resp = client.put("/admin/api/server", json={"port": 9001, "log_level": "DEBUG"})
    assert resp.status_code == 200
    assert _read_config(cfg_path)["server"]["port"] == 9001
    assert _read_config(cfg_path)["server"]["log_level"] == "DEBUG"


def test_put_server_rejects_bad_port(client):
    resp = client.put("/admin/api/server", json={"port": 70000})
    assert resp.status_code == 400


def test_put_server_rejects_bad_log_level(client):
    resp = client.put("/admin/api/server", json={"log_level": "TRACE"})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# Providers CRUD

def test_create_provider(client, cfg_path):
    resp = client.post("/admin/api/providers", json={
        "name": "groq", "base_url": "https://api.groq.com/openai/v1", "api_key": "gsk_abc",
    })
    assert resp.status_code == 201
    assert "groq" in _read_config(cfg_path)["providers"]


def test_create_provider_rejects_reserved_name(client):
    resp = client.post("/admin/api/providers", json={"name": "llmproxy", "base_url": "http://x/v1"})
    assert resp.status_code == 409


def test_create_provider_rejects_duplicate(client):
    resp = client.post("/admin/api/providers", json={"name": "openai", "base_url": "http://x/v1"})
    assert resp.status_code == 409


def test_create_provider_requires_base_url(client):
    resp = client.post("/admin/api/providers", json={"name": "x"})
    assert resp.status_code == 400


def test_update_provider_keeps_key_when_blank(client, cfg_path):
    resp = client.put("/admin/api/providers/openai", json={
        "base_url": "https://api.openai.com/v2", "api_key": "",
    })
    assert resp.status_code == 200
    saved = _read_config(cfg_path)["providers"]["openai"]
    assert saved["base_url"] == "https://api.openai.com/v2"
    assert saved["api_key"] == "sk-supersecretkey123"  # preserved


def test_update_provider_overwrites_key(client, cfg_path):
    resp = client.put("/admin/api/providers/openai", json={
        "base_url": "https://api.openai.com/v1", "api_key": "${OPENAI_API_KEY}",
    })
    assert resp.status_code == 200
    assert _read_config(cfg_path)["providers"]["openai"]["api_key"] == "${OPENAI_API_KEY}"


def test_update_unknown_provider_404(client):
    resp = client.put("/admin/api/providers/ghost", json={"base_url": "http://x/v1"})
    assert resp.status_code == 404


def test_delete_provider(client, cfg_path):
    resp = client.delete("/admin/api/providers/openai")
    assert resp.status_code == 200
    assert "openai" not in _read_config(cfg_path)["providers"]


def test_model_filter_validation(client):
    resp = client.put("/admin/api/providers/openai", json={
        "base_url": "https://api.openai.com/v1", "model_filter": [1, 2],
    })
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# Templates

def test_provider_templates(client):
    resp = client.get("/admin/api/provider-templates")
    assert resp.status_code == 200
    keys = [t["key"] for t in resp.get_json()["templates"]]
    assert keys  # providers.json ships several


def test_from_template_creates_provider(client, cfg_path):
    templates = client.get("/admin/api/provider-templates").get_json()["templates"]
    # pick a template that needs no account/gateway id
    simple = next(t for t in templates if not t.get("account_id_required") and not t.get("gateway_id_required"))
    resp = client.post("/admin/api/providers/from-template", json={
        "template_key": simple["key"], "name": "tmpltest", "api_key": "${SOME_KEY}",
    })
    assert resp.status_code == 201
    saved = _read_config(cfg_path)["providers"]["tmpltest"]
    assert saved["base_url"]
    assert saved["api_key"] == "${SOME_KEY}"


def test_from_template_unknown(client):
    resp = client.post("/admin/api/providers/from-template", json={"template_key": "nope"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Categorizations

def test_put_believed_free(client, cfg_path):
    resp = client.put("/admin/api/believed-free", json=["a/b", "c/d"])
    assert resp.status_code == 200
    assert _read_config(cfg_path)["believed_free"] == ["a/b", "c/d"]


def test_put_believed_free_rejects_non_list(client):
    resp = client.put("/admin/api/believed-free", json={"a": 1})
    assert resp.status_code == 400


def test_put_model_reasoning_validates_level(client):
    resp = client.put("/admin/api/model-reasoning", json={"m": "ultra"})
    assert resp.status_code == 400


def test_put_model_reasoning_ok(client, cfg_path):
    resp = client.put("/admin/api/model-reasoning", json={"m": "deep"})
    assert resp.status_code == 200
    assert _read_config(cfg_path)["model_reasoning"]["m"] == "deep"


def test_put_capabilities_validates(client):
    resp = client.put("/admin/api/model-capabilities", json={"m": ["telepathy"]})
    assert resp.status_code == 400


def test_put_capabilities_ok(client, cfg_path):
    resp = client.put("/admin/api/model-capabilities", json={"m": ["tools", "vision"]})
    assert resp.status_code == 200
    assert _read_config(cfg_path)["model_capabilities"]["m"] == ["tools", "vision"]


def test_put_free_limits_validates_key(client):
    resp = client.put("/admin/api/free-limits", json={"m": {"bogus": 1}})
    assert resp.status_code == 400


def test_put_free_limits_ok(client, cfg_path):
    resp = client.put("/admin/api/free-limits", json={"m": {"requests_per_minute": 15}})
    assert resp.status_code == 200
    assert _read_config(cfg_path)["free_limits"]["m"]["requests_per_minute"] == 15


# --------------------------------------------------------------------------- #
# Virtual-model preview + validate/heal

def test_virtual_models_preview(client):
    resp = client.get("/admin/api/virtual-models")
    assert resp.status_code == 200
    ids = [v["id"] for v in resp.get_json()["virtual_models"]]
    # BASE_CONFIG tags a deep model and a tools capability and a believed_free
    assert "llmproxy__deep" in ids
    assert "llmproxy__tools" in ids
    assert "llmproxy__free" in ids


def test_validate(client):
    resp = client.post("/admin/api/validate")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True


def test_heal_runs(client):
    resp = client.post("/admin/api/heal")
    assert resp.status_code == 200
    assert "changed" in resp.get_json()


# --------------------------------------------------------------------------- #
# Model discovery (monkeypatched upstream)

def test_provider_models_discovery(monkeypatch, cfg_path):
    server_mod = _make_server(monkeypatch, cfg_path, dict(BASE_CONFIG))
    monkeypatch.setattr(
        server_mod, "_fetch_provider_models",
        lambda name, cfg, timeout: [{"id": f"{name}__gpt-4o"}, {"id": f"{name}__gpt-4o-mini"}],
    )
    client = server_mod.app.test_client()
    resp = client.get("/admin/api/providers/openai/models")
    assert resp.status_code == 200
    assert resp.get_json()["models"] == ["openai__gpt-4o", "openai__gpt-4o-mini"]


# --------------------------------------------------------------------------- #
# Maintenance / automation settings

def test_get_config_includes_maintenance(client):
    body = client.get("/admin/api/config").get_json()
    assert "maintenance" in body
    m = body["maintenance"]
    for k in ("probe_cost", "autoremove_believed_free", "update_believed_free_on_startup",
              "pr_providers_list", "probe_frequency_days", "pr_providers_repo",
              "pr_providers_token_set"):
        assert k in m


def test_put_maintenance_sets_flags(client, cfg_path):
    resp = client.put("/admin/api/maintenance", json={
        "probe_cost": True,
        "autoremove_believed_free": True,
        "update_believed_free_on_startup": True,
        "probe_frequency_days": 7,
        "pr_providers_list": True,
        "pr_providers_repo": "BillJr99/llmproxy",
        "pr_providers_base": "main",
        "pr_providers_branch": "llmproxy-auto/providers",
        "pr_providers_token": "ghp_secret",
    })
    assert resp.status_code == 200
    saved = _read_config(cfg_path)
    assert saved["probe_cost"] is True
    assert saved["probe_frequency_days"] == 7
    assert saved["pr_providers_repo"] == "BillJr99/llmproxy"
    assert saved["pr_providers_token"] == "ghp_secret"
    # Token is never echoed back verbatim
    m = resp.get_json()["maintenance"]
    assert m["pr_providers_token"] != "ghp_secret"
    assert m["pr_providers_token_set"] is True


def test_put_maintenance_blank_token_keeps_existing(client, cfg_path):
    client.put("/admin/api/maintenance", json={"pr_providers_token": "ghp_keepme"})
    client.put("/admin/api/maintenance", json={"pr_providers_list": True, "pr_providers_token": ""})
    assert _read_config(cfg_path)["pr_providers_token"] == "ghp_keepme"


def test_put_maintenance_rejects_bad_values(client):
    assert client.put("/admin/api/maintenance", json={"probe_cost": "yes"}).status_code == 400
    assert client.put("/admin/api/maintenance", json={"probe_frequency_days": -1}).status_code == 400
    assert client.put("/admin/api/maintenance", json={"probe_frequency_days": "soon"}).status_code == 400


def test_put_maintenance_token_env_ref_flagged(client):
    resp = client.put("/admin/api/maintenance", json={"pr_providers_token": "${GH_PR_TOKEN}"})
    m = resp.get_json()["maintenance"]
    assert m["pr_providers_token_is_env"] is True
