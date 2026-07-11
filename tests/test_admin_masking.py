"""Tests that the admin API never leaks plaintext secrets and round-trips
env references verbatim."""

from __future__ import annotations

import importlib
import json

import pytest

CONFIG = {
    "providers": {
        "literal": {"base_url": "https://api.x/v1", "api_key": "sk-plaintext-abcdef123456", "model_filter": None},
        "envref": {"base_url": "https://${HOST}/v1", "api_key": "${OPENAI_API_KEY}", "model_filter": None},
        "nokey": {"base_url": "http://localhost:11434/v1", "model_filter": None},
    },
    "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR", "request_timeout": 5, "stream_timeout": 5},
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(CONFIG))
    monkeypatch.setenv("LLMPROXY_CONFIG", str(cfg_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    server_mod.app.config["TESTING"] = True
    return server_mod.app.test_client(), cfg_path


def test_plaintext_key_is_masked(client):
    c, _ = client
    body = c.get("/admin/api/providers").get_json()
    masked = body["literal"]["api_key"]
    assert masked != "sk-plaintext-abcdef123456"
    assert "plaintext" not in masked
    assert body["literal"]["api_key_set"] is True
    assert body["literal"]["api_key_is_env"] is False


def test_env_ref_returned_verbatim(client):
    c, _ = client
    body = c.get("/admin/api/providers").get_json()
    assert body["envref"]["api_key"] == "${OPENAI_API_KEY}"
    assert body["envref"]["api_key_is_env"] is True
    assert body["envref"]["base_url_is_env"] is True


def test_no_key_provider(client):
    c, _ = client
    body = c.get("/admin/api/providers").get_json()
    assert body["nokey"]["api_key"] == ""
    assert body["nokey"]["api_key_set"] is False


def test_full_config_never_contains_plaintext(client):
    c, _ = client
    raw = c.get("/admin/api/config").get_data(as_text=True)
    assert "sk-plaintext-abcdef123456" not in raw


def test_blank_submit_keeps_secret_on_disk(client):
    c, cfg_path = client
    resp = c.put("/admin/api/providers/literal", json={"base_url": "https://api.x/v2", "api_key": ""})
    assert resp.status_code == 200
    saved = json.loads(cfg_path.read_text())["providers"]["literal"]
    assert saved["api_key"] == "sk-plaintext-abcdef123456"  # untouched
    assert saved["base_url"] == "https://api.x/v2"


# ── multiple accounts per provider ───────────────────────────────────────────

MULTI_CONFIG = {
    "providers": {
        "multi": {
            "base_url": "https://api.multi/v1",
            "account_strategy": "priority",
            "accounts": [
                {"key": "sk-acct-one-secret", "label": "team-a", "priority": 1},
                {"key": "${TEAM_B_KEY}", "label": "team-b", "priority": 2},
            ],
        },
    },
    "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
               "request_timeout": 5, "stream_timeout": 5},
}


@pytest.fixture
def multi_client(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(MULTI_CONFIG))
    monkeypatch.setenv("LLMPROXY_CONFIG", str(cfg_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    server_mod.app.config["TESTING"] = True
    return server_mod.app.test_client(), cfg_path


def test_account_keys_are_masked(multi_client):
    c, _ = multi_client
    view = c.get("/admin/api/providers").get_json()["multi"]
    assert view["account_strategy"] == "priority"
    accts = view["accounts"]
    assert [a["label"] for a in accts] == ["team-a", "team-b"]
    # Literal key masked, env ref flagged, raw material never present.
    assert "secret" not in json.dumps(accts)
    assert accts[0]["key_set"] is True and accts[0]["key_is_env"] is False
    assert accts[1]["key_is_env"] is True


def test_full_config_never_leaks_account_secret(multi_client):
    c, _ = multi_client
    raw = c.get("/admin/api/config").get_data(as_text=True)
    assert "sk-acct-one-secret" not in raw


def test_post_accepts_and_persists_accounts(multi_client):
    c, cfg_path = multi_client
    resp = c.put("/admin/api/providers/multi", json={
        "base_url": "https://api.multi/v1",
        "account_strategy": "round_robin",
        "accounts": [
            {"key": "sk-new-a", "label": "a"},
            {"key": "sk-new-b", "label": "b", "priority": 3},
        ],
    })
    assert resp.status_code == 200
    saved = json.loads(cfg_path.read_text())["providers"]["multi"]
    # round_robin is the default and dropped to keep configs clean.
    assert "account_strategy" not in saved
    assert [a["key"] for a in saved["accounts"]] == ["sk-new-a", "sk-new-b"]
    assert saved["accounts"][1]["priority"] == 3
    assert saved["accounts"][0].get("label") == "a"


def test_blank_account_key_preserved_by_label(multi_client):
    c, cfg_path = multi_client
    # Re-submit team-a with a blank key (as the masked UI would) -> key kept.
    resp = c.put("/admin/api/providers/multi", json={
        "base_url": "https://api.multi/v1",
        "accounts": [{"key": "", "label": "team-a"}],
    })
    assert resp.status_code == 200
    saved = json.loads(cfg_path.read_text())["providers"]["multi"]["accounts"]
    assert saved[0]["key"] == "sk-acct-one-secret"  # preserved, not blanked


def test_invalid_account_strategy_rejected(multi_client):
    c, _ = multi_client
    resp = c.put("/admin/api/providers/multi", json={
        "base_url": "https://api.multi/v1", "account_strategy": "nonsense",
    })
    assert resp.status_code == 400
