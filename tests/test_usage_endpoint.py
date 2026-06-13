"""Tests for the /v1/usage accounting endpoint and believed_free cost flagging."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _load_server_with_config(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    return server_mod


@pytest.fixture
def usage_server(tmp_path, monkeypatch):
    cfg = {
        "providers": {
            "groq": {"base_url": "http://groq.example/v1", "api_key": "k", "model_filter": None},
        },
        "believed_free": ["groq/free-model"],
        "free_limits": {},
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5, "models_cache_ttl": 0,
                   "response_cache_ttl": 0},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    server = _load_server_with_config(monkeypatch, path)
    server._reset_usage()
    return server


def test_usage_empty_report(usage_server):
    client = usage_server.app.test_client()
    resp = client.get("/v1/usage")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["object"] == "usage.report"
    assert body["models"] == []
    assert body["totals"]["total_tokens"] == 0
    assert body["flagged_paid_free_models"] == []
    assert "since" in body


def test_usage_records_tokens_and_totals(usage_server):
    usage_server._record_usage(
        "groq", "free-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        config=usage_server.load_config(),
    )
    body = usage_server.app.test_client().get("/v1/usage").get_json()
    assert body["totals"]["requests"] == 1
    assert body["totals"]["total_tokens"] == 15
    assert len(body["models"]) == 1
    m = body["models"][0]
    assert m["model"] == "groq/free-model"
    assert m["prompt_tokens"] == 10
    assert m["believed_free"] is True
    assert m["unexpected_cost"] is False


def test_believed_free_cost_is_flagged(usage_server):
    # A believed_free model whose response reports a provider cost gets flagged.
    usage_server._record_usage(
        "groq", "free-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.002},
        config=usage_server.load_config(),
    )
    body = usage_server.app.test_client().get("/v1/usage").get_json()
    flagged = body["flagged_paid_free_models"]
    assert len(flagged) == 1
    assert flagged[0]["model"] == "groq/free-model"
    assert flagged[0]["observed_cost"] == pytest.approx(0.002)
    assert flagged[0]["cost_source"] == "provider"
    # And the per-model row marks the unexpected cost.
    assert body["models"][0]["unexpected_cost"] is True


def test_usage_reset_clears_and_requires_no_token_on_loopback(usage_server):
    usage_server._record_usage(
        "groq", "free-model",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        config=usage_server.load_config(),
    )
    client = usage_server.app.test_client()
    # No admin token configured + loopback test client → reset allowed.
    resp = client.post("/v1/usage/reset")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True
    body = client.get("/v1/usage").get_json()
    assert body["models"] == []


def test_usage_reset_rejected_without_token_when_forwarded(usage_server):
    client = usage_server.app.test_client()
    # Forwarding header present + no token configured → admin guard rejects.
    resp = client.post("/v1/usage/reset", headers={"X-Forwarded-For": "1.2.3.4"})
    assert resp.status_code == 403
