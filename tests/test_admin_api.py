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
    # These tests assert on a static config; opt out of the on-by-default startup
    # reconcile so the background sync thread can't rewrite it mid-test.
    "sync_believed_free_on_startup": False,
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
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
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


def _read_curated(path: Path) -> dict:
    """What the admin editors persist now: the sidecar's hand-set section.

    They used to write config.json. Once the five routing keys migrated out of
    it, reading config alone showed every model as un-free and untagged, and
    saving that form wrote the blanks back as real overrides that outranked
    everything learned. The editors target the curated layer instead, and only
    record what actually differs from the layers below.
    """
    side = path.parent / "routing_metadata.json"
    if not side.exists():
        return {}
    curated = json.loads(side.read_text()).get("curated")
    return curated if isinstance(curated, dict) else {}


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
    # The EFFECTIVE view, not config.json's slice of it. The page showed every
    # model as un-free and untagged once the five keys migrated out of config,
    # and saving that form wrote the blanks back as overrides that outranked
    # everything learned.
    assert "openai/free-thing" in body["believed_free"]
    assert len(body["believed_free"]) > 1, \
        "providers.json defaults should reach the admin view"
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
    assert _read_curated(cfg_path)["believed_free"] == ["a/b", "c/d"]


def test_editors_never_write_config_json(client, cfg_path):
    """Saving a categorization must leave config.json byte-identical.

    config.json is drained into the sidecar at startup and is not a routing
    layer any more. An editor that wrote back to it would re-seed exactly the
    keys the migration strips, which is how the old local-sync kept undoing it.
    """
    before = cfg_path.read_text()
    assert client.put("/admin/api/believed-free", json=["a/b"]).status_code == 200
    assert client.put("/admin/api/model-reasoning", json={"m": "deep"}).status_code == 200
    assert client.put("/admin/api/model-capabilities", json={"m": ["tools"]}).status_code == 200
    assert client.put("/admin/api/free-limits",
                      json={"m": {"requests_per_minute": 5}}).status_code == 200
    assert cfg_path.read_text() == before


def test_put_believed_free_rejects_non_list(client):
    resp = client.put("/admin/api/believed-free", json={"a": 1})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# Favorite free models

def test_get_favorite_free_models_default_empty(client):
    resp = client.get("/admin/api/favorite-free-models")
    assert resp.status_code == 200
    assert resp.get_json()["favorite_free_models"] == []


def test_put_favorite_free_models_valid(client, cfg_path):
    favs = ["google/gemini-2.5-flash", "openai/gpt-4o-mini"]
    resp = client.put("/admin/api/favorite-free-models", json=favs)
    assert resp.status_code == 200
    assert resp.get_json()["favorite_free_models"] == favs
    assert _read_config(cfg_path)["favorite_free_models"] == favs


def test_put_favorite_free_models_rejects_non_list(client):
    resp = client.put("/admin/api/favorite-free-models", json={"model": "x"})
    assert resp.status_code == 400


def test_put_favorite_free_models_rejects_non_string_entries(client):
    resp = client.put("/admin/api/favorite-free-models", json=["ok", 42])
    assert resp.status_code == 400


def test_favorite_free_models_in_config_get(client, cfg_path):
    favs = ["google/gemini-flash"]
    client.put("/admin/api/favorite-free-models", json=favs)
    resp = client.get("/admin/api/config")
    assert resp.status_code == 200
    assert resp.get_json()["favorite_free_models"] == favs


def test_get_favorite_free_models_round_trips_empty_list(client, cfg_path):
    client.put("/admin/api/favorite-free-models", json=[])
    resp = client.get("/admin/api/favorite-free-models")
    assert resp.get_json()["favorite_free_models"] == []


def test_put_model_reasoning_validates_level(client):
    resp = client.put("/admin/api/model-reasoning", json={"m": "ultra"})
    assert resp.status_code == 400


def test_put_model_reasoning_ok(client, cfg_path):
    resp = client.put("/admin/api/model-reasoning", json={"m": "deep"})
    assert resp.status_code == 200
    assert _read_curated(cfg_path)["model_reasoning"]["m"] == "deep"


def test_put_capabilities_validates(client):
    resp = client.put("/admin/api/model-capabilities", json={"m": ["telepathy"]})
    assert resp.status_code == 400


def test_put_capabilities_ok(client, cfg_path):
    resp = client.put("/admin/api/model-capabilities", json={"m": ["tools", "vision"]})
    assert resp.status_code == 200
    assert _read_curated(cfg_path)["model_capabilities"]["m"] == ["tools", "vision"]


def test_put_free_limits_validates_key(client):
    resp = client.put("/admin/api/free-limits", json={"m": {"bogus": 1}})
    assert resp.status_code == 400


def test_put_free_limits_ok(client, cfg_path):
    resp = client.put("/admin/api/free-limits", json={"m": {"requests_per_minute": 15}})
    assert resp.status_code == 200
    assert _read_curated(cfg_path)["free_limits"]["m"]["requests_per_minute"] == 15


# --------------------------------------------------------------------------- #
# Routing metadata: the effective view, and per-model editing

def test_routing_metadata_reports_effective_values_and_their_layers(client):
    """The grid must show what is IN EFFECT and where it came from.

    Reading config.json alone rendered every model un-free and untagged once the
    five keys migrated out, and saving that view wrote the blanks back as
    overrides outranking everything learned.
    """
    resp = client.get("/admin/api/routing-metadata?limit=5")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["total"] > 0
    assert body["layers"] == ["providers.json", "learned", "listing", "curated"]
    for row in body["models"]:
        assert set(row) >= {"id", "free", "reasoning", "capabilities",
                            "free_limits", "layers", "grades"}


def test_routing_metadata_never_lists_a_normalized_join_key(client, cfg_path):
    """Rows must be routing targets, not the learned layer's join keys.

    The learned layer is keyed by normalize_model_id on purpose, so one entry
    covers every provider spelling of the same weights. Those keys are not
    callable model ids — "aionlabsaion30mini" is a join key, not something a
    request can address — and listing them filled the grid with thousands of
    phantom models and made every diagnostic built on the listing wrong.
    """
    import json as _json
    side = cfg_path.parent / "routing_metadata.json"
    side.write_text(_json.dumps({
        "by_model": {
            "aionlabsaion30mini": {"capabilities": ["tools"],
                                   "capabilities_source": "observed"},
            "llama3370b": {"reasoning": "standard", "reasoning_source": "inferred"},
        },
        "curated": {"model_reasoning": {"groq/hand-typed-model": "deep"}},
    }), encoding="utf-8")
    from llmproxy import server
    server._reset_routing_sidecar_cache()

    ids = [r["id"] for r in
           client.get("/admin/api/routing-metadata?limit=500").get_json()["models"]]
    assert "aionlabsaion30mini" not in ids
    assert "llama3370b" not in ids
    # An id a person typed by hand is still listed, even with no route yet.
    assert "groq/hand-typed-model" in ids
    server._reset_routing_sidecar_cache()


def test_routing_metadata_pages_and_filters_server_side(client):
    """Paging is server-side because a grid of thousands re-rendered per
    keystroke, with six listeners per row, is what the old page did."""
    everything = client.get("/admin/api/routing-metadata?limit=500").get_json()
    assert len(everything["models"]) <= 500
    page = client.get("/admin/api/routing-metadata?offset=1&limit=1").get_json()
    assert len(page["models"]) == 1
    assert page["offset"] == 1
    narrowed = client.get("/admin/api/routing-metadata?q=zzz-no-such-model").get_json()
    assert narrowed["total"] == 0


def test_routing_metadata_put_records_one_model_as_curated(client, cfg_path):
    """Per-model, so editing one row cannot blank the rest — which is exactly
    how the old whole-section save could wipe the learned layer in one click."""
    resp = client.put("/admin/api/routing-metadata", json={
        "model": "groq/test-model", "capabilities": ["tools", "json"],
        "reasoning": "deep", "free": True,
        "free_limits": {"requests_per_minute": 15},
    })
    assert resp.status_code == 200
    curated = _read_curated(cfg_path)
    assert curated["model_capabilities"]["groq/test-model"] == ["json", "tools"]
    assert curated["model_reasoning"]["groq/test-model"] == "deep"
    assert "groq/test-model" in curated["believed_free"]
    assert curated["free_limits"]["groq/test-model"]["requests_per_minute"] == 15


def test_clearing_a_field_hands_the_model_back_to_what_was_learned(client, cfg_path):
    """An override must be removable. Otherwise a correction made once freezes
    the model forever and no later refresh can improve it."""
    client.put("/admin/api/routing-metadata",
               json={"model": "groq/test-model", "reasoning": "deep"})
    assert _read_curated(cfg_path)["model_reasoning"].get("groq/test-model") == "deep"
    client.put("/admin/api/routing-metadata",
               json={"model": "groq/test-model", "reasoning": ""})
    assert "groq/test-model" not in _read_curated(cfg_path)["model_reasoning"]


@pytest.mark.parametrize("payload", [
    {"capabilities": ["tools"]},                       # no model
    {"model": "m", "capabilities": ["telepathy"]},     # unknown capability
    {"model": "m", "reasoning": "ultra"},              # unknown level
    {"model": "m", "reasoning": "flagship"},           # computed overlay
    {"model": "m", "free_limits": {"bogus": 1}},       # unknown limit key
    {"model": "m", "free_limits": {"requests_per_minute": "lots"}},
    {"model": "m", "free": "yes"},
])
def test_routing_metadata_put_rejects_bad_payloads(client, payload):
    assert client.put("/admin/api/routing-metadata", json=payload).status_code == 400


def test_refresh_rejects_an_unknown_target(client):
    assert client.post("/admin/api/refresh", json={"what": "nope"}).status_code == 400


# --------------------------------------------------------------------------- #
# Virtual-model preview + validate/heal

def test_virtual_models_preview(client):
    resp = client.get("/admin/api/virtual-models")
    assert resp.status_code == 200
    ids = [v["id"] for v in resp.get_json()["virtual_models"]]
    # BASE_CONFIG tags a deep model and a tools capability and a believed_free
    assert "llmproxy/deep" in ids
    assert "llmproxy/tools" in ids
    assert "llmproxy/free" in ids


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


def test_sync_believed_free_flag_defaults_true_and_toggles(client, cfg_path):
    # BASE_CONFIG sets it False explicitly; flip to absent to see the default.
    cfg = _read_config(cfg_path)
    cfg.pop("sync_believed_free_on_startup", None)
    cfg_path.write_text(json.dumps(cfg))
    m = client.get("/admin/api/config").get_json()["maintenance"]
    assert m["sync_believed_free_on_startup"] is True  # default when absent

    resp = client.put("/admin/api/maintenance", json={"sync_believed_free_on_startup": False})
    assert resp.status_code == 200
    assert _read_config(cfg_path)["free_tier"]["sync_on_startup"] is False
    assert resp.get_json()["maintenance"]["sync_believed_free_on_startup"] is False


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
    assert saved["free_tier"]["cost_probe"]["enabled"] is True
    assert saved["free_tier"]["cost_probe"]["frequency_days"] == 7
    assert saved["providers_pr"]["repo"] == "BillJr99/llmproxy"
    assert saved["providers_pr"]["token"] == "ghp_secret"
    # Token is never echoed back verbatim
    m = resp.get_json()["maintenance"]
    assert m["pr_providers_token"] != "ghp_secret"
    assert m["pr_providers_token_set"] is True


def test_put_maintenance_blank_token_keeps_existing(client, cfg_path):
    client.put("/admin/api/maintenance", json={"pr_providers_token": "ghp_keepme"})
    client.put("/admin/api/maintenance", json={"pr_providers_list": True, "pr_providers_token": ""})
    assert _read_config(cfg_path)["providers_pr"]["token"] == "ghp_keepme"


def test_put_maintenance_rejects_bad_values(client):
    assert client.put("/admin/api/maintenance", json={"probe_cost": "yes"}).status_code == 400
    assert client.put("/admin/api/maintenance", json={"probe_frequency_days": -1}).status_code == 400
    assert client.put("/admin/api/maintenance", json={"probe_frequency_days": "soon"}).status_code == 400


def test_put_maintenance_token_env_ref_flagged(client):
    resp = client.put("/admin/api/maintenance", json={"pr_providers_token": "${GH_PR_TOKEN}"})
    m = resp.get_json()["maintenance"]
    assert m["pr_providers_token_is_env"] is True
