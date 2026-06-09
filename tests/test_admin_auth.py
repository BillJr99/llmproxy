"""Tests for the admin API auth guard: localhost-only by default, token when set."""

from __future__ import annotations

import importlib
import json


def _client(monkeypatch, tmp_path, admin_block=None):
    cfg = {
        "providers": {},
        "server": {"host": "0.0.0.0", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    if admin_block is not None:
        cfg["admin"] = admin_block
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setenv("LLMPROXY_CONFIG", str(cfg_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    server_mod.app.config["TESTING"] = True
    return server_mod.app.test_client()


# --------------------------------------------------------------------------- #
# No token configured -> localhost only

def test_loopback_allowed_without_token(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    resp = c.get("/admin/api/config")  # test client defaults to 127.0.0.1
    assert resp.status_code == 200


def test_non_loopback_denied_without_token(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    resp = c.get("/admin/api/config", environ_base={"REMOTE_ADDR": "8.8.8.8"})
    assert resp.status_code == 403


def test_ui_shell_public_even_remote(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    resp = c.get("/admin", environ_base={"REMOTE_ADDR": "8.8.8.8"})
    assert resp.status_code == 200  # shell carries no secrets


# --------------------------------------------------------------------------- #
# Token configured -> any origin with token

def test_token_required_when_set(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, admin_block={"enabled": True, "token": "s3cret"})
    resp = c.get("/admin/api/config", environ_base={"REMOTE_ADDR": "8.8.8.8"})
    assert resp.status_code == 401


def test_correct_bearer_token_allows_remote(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, admin_block={"enabled": True, "token": "s3cret"})
    resp = c.get("/admin/api/config", headers={"Authorization": "Bearer s3cret"},
                 environ_base={"REMOTE_ADDR": "8.8.8.8"})
    assert resp.status_code == 200


def test_x_admin_token_header_allows(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, admin_block={"enabled": True, "token": "s3cret"})
    resp = c.get("/admin/api/config", headers={"X-Admin-Token": "s3cret"},
                 environ_base={"REMOTE_ADDR": "8.8.8.8"})
    assert resp.status_code == 200


def test_wrong_token_denied(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, admin_block={"enabled": True, "token": "s3cret"})
    resp = c.get("/admin/api/config", headers={"X-Admin-Token": "nope"})
    assert resp.status_code == 401


def test_env_token_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMPROXY_ADMIN_TOKEN", "envtok")
    c = _client(monkeypatch, tmp_path, admin_block={"enabled": True, "token": ""})
    ok = c.get("/admin/api/config", headers={"X-Admin-Token": "envtok"},
               environ_base={"REMOTE_ADDR": "8.8.8.8"})
    assert ok.status_code == 200


def test_disabled_admin_api_404(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, admin_block={"enabled": False})
    resp = c.get("/admin/api/config")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# LLMPROXY_ADMIN_ENABLED env override (set by --admin / --no-admin) must reach
# the blueprint, which reads config from disk per request.

def test_env_disable_overrides_enabled_config(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMPROXY_ADMIN_ENABLED", "0")
    c = _client(monkeypatch, tmp_path, admin_block={"enabled": True})
    assert c.get("/admin/api/config").status_code == 404
    assert c.get("/admin").status_code == 404


def test_env_enable_overrides_disabled_config(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMPROXY_ADMIN_ENABLED", "1")
    c = _client(monkeypatch, tmp_path, admin_block={"enabled": False})
    assert c.get("/admin/api/config").status_code == 200


# --------------------------------------------------------------------------- #
# Config token may itself be an env reference

def test_token_from_env_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_ADMIN_TOK", "viaref")
    c = _client(monkeypatch, tmp_path, admin_block={"enabled": True, "token": "${MY_ADMIN_TOK}"})
    resp = c.get("/admin/api/config", headers={"X-Admin-Token": "viaref"},
                 environ_base={"REMOTE_ADDR": "8.8.8.8"})
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Reverse-proxy footgun: a loopback remote_addr with forwarding headers must
# not be trusted as local when no token is configured.

def test_forwarded_header_denied_without_token(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)  # remote_addr defaults to 127.0.0.1
    resp = c.get("/admin/api/config", headers={"X-Forwarded-For": "8.8.8.8"})
    assert resp.status_code == 403


def test_real_ip_header_denied_without_token(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    resp = c.get("/admin/api/config", headers={"X-Real-IP": "8.8.8.8"})
    assert resp.status_code == 403


def test_forwarded_with_token_allowed(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, admin_block={"enabled": True, "token": "s3cret"})
    resp = c.get("/admin/api/config",
                 headers={"X-Forwarded-For": "8.8.8.8", "X-Admin-Token": "s3cret"})
    assert resp.status_code == 200
