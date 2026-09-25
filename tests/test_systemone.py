"""TypeSafe Jev: the /v1/systemone route and keeping decision models out of chat.

Jev answers typed questions with probabilities and never generates text, so it
is served by its own native endpoint and must never land in a chat candidate
pool — whichever provider (TypeSafe itself, or a gateway relaying it) lists it.
"""

from __future__ import annotations

import datetime
import importlib
import json
from pathlib import Path

import pytest

from llmproxy.providers import load_data


def _write_config(tmp_path: Path) -> Path:
    cfg = {
        "providers": {
            "typesafe": {
                "base_url": "https://api.typesafe.ai/v1",
                "api_key": "ts-key",
                "model_filter": None,
                "models_id_field": "name",
            },
            "fakeprov": {
                "base_url": "http://upstream.example/v1",
                "api_key": "fake-key",
                "model_filter": None,
            },
        },
        "believed_free": [],
        "model_reasoning": {},
        "free_limits": {},
        "sync_believed_free_on_startup": False,
        "server": {"log_level": "ERROR", "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(_write_config(tmp_path)))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def client(server):
    server.app.config["TESTING"] = True
    return server.app.test_client()


class _FakeResp:
    def __init__(self, status: int, body: dict):
        self.status_code = status
        self.content = json.dumps(body).encode()
        self.headers = {"Content-Type": "application/json"}
        self.elapsed = datetime.timedelta(milliseconds=5)

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


_REQUEST = {
    "model": "typesafe/jev-latest",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
}
_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {"is_urgent": {"type": "noul", "noul": 0.95}},
    "usage": {"input_tokens": 296, "output_tokens": 20},
}


# ---------------------------------------------------------------------------
# Decision-model detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("uid", [
    "jev-latest", "jev-preview", "jev-1.13.0", "typesafe-ai/jev", "typesafe/jev-1.13.0",
])
def test_decision_models_are_recognised(server, uid):
    assert server._is_decision_model(uid)


@pytest.mark.parametrize("uid", ["gpt-4o", "jevons-7b", "anthropic/claude-sonnet-5", "free"])
def test_chat_models_are_not_decision_models(server, uid):
    assert not server._is_decision_model(uid)


# ---------------------------------------------------------------------------
# Discovery keeps Jev out of the route cache (and so out of every virtual pool)
# ---------------------------------------------------------------------------

def test_typesafe_catalog_contributes_no_chat_models(server, monkeypatch):
    body = {"models": [{"name": "jev-latest"}, {"name": "jev-preview"}]}
    monkeypatch.setattr(server.requests, "get", lambda *a, **k: _FakeResp(200, body))
    cfg = server.get_provider(server.load_config(), "typesafe")
    assert server._fetch_provider_models("typesafe", cfg, 5) == []


def test_gateway_relayed_jev_is_skipped_but_chat_models_kept(server, monkeypatch):
    body = {"data": [{"id": "typesafe-ai/jev"}, {"id": "openai/gpt-4o"}]}
    monkeypatch.setattr(server.requests, "get", lambda *a, **k: _FakeResp(200, body))
    cfg = server.get_provider(server.load_config(), "fakeprov")
    ids = [m["_upstream_id"] for m in server._fetch_provider_models("fakeprov", cfg, 5)]
    assert ids == ["openai/gpt-4o"]


# ---------------------------------------------------------------------------
# POST /v1/systemone
# ---------------------------------------------------------------------------

def test_systemone_forwards_native_body_and_records_usage(server, client, monkeypatch):
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, json=json)
        return _FakeResp(200, _RESPONSE)

    recorded = []
    monkeypatch.setattr(server.requests, "post", fake_post)
    monkeypatch.setattr(server, "_record_usage",
                        lambda pn, um, **kw: recorded.append((pn, um, kw.get("usage"))))

    resp = client.post("/v1/systemone", json=_REQUEST)

    assert resp.status_code == 200
    assert resp.get_json() == _RESPONSE
    assert sent["url"] == "https://api.typesafe.ai/v1/systemone"
    assert sent["headers"]["Authorization"] == "Bearer ts-key"
    assert sent["json"] == {**_REQUEST, "model": "jev-latest"}
    assert len(recorded) == 1
    provider, model, usage = recorded[0]
    assert (provider, model) == ("typesafe", "jev-latest")
    assert usage["prompt_tokens"] == 296 and usage["completion_tokens"] == 20


def test_systemone_accepts_display_id(server, client, monkeypatch):
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, json=json)
        return _FakeResp(200, _RESPONSE)

    monkeypatch.setattr(server.requests, "post", fake_post)
    resp = client.post("/v1/systemone", json={**_REQUEST, "model": "typesafe__jev-1.13.0"})
    assert resp.status_code == 200
    assert sent["json"]["model"] == "jev-1.13.0"


def test_systemone_passes_upstream_errors_through(server, client, monkeypatch):
    err = {"detail": "questions.is_urgent.type: invalid"}
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: _FakeResp(422, err))
    resp = client.post("/v1/systemone", json=_REQUEST)
    assert resp.status_code == 422
    assert resp.get_json() == err


def test_systemone_rejects_streaming(client):
    resp = client.post("/v1/systemone", json={**_REQUEST, "stream": True})
    assert resp.status_code == 400


def test_systemone_rejects_chat_models(server, client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not reach upstream")

    monkeypatch.setattr(server.requests, "post", boom)
    resp = client.post("/v1/systemone", json={**_REQUEST, "model": "fakeprov/gpt-4o"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Chat surfaces refuse a decision model with a pointer to /v1/systemone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["typesafe/jev-latest", "fakeprov/typesafe-ai/jev"])
def test_chat_request_for_jev_is_redirected(server, client, monkeypatch, model):
    def boom(*a, **k):
        raise AssertionError("must not reach upstream")

    monkeypatch.setattr(server.requests, "post", boom)
    resp = client.post("/v1/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 400
    assert "/v1/systemone" in resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# Sidecar / setup wizard
# ---------------------------------------------------------------------------

def test_typesafe_template_is_offered_by_the_wizard():
    from llmproxy.setup_wizard import PROVIDER_TEMPLATES
    data = load_data()
    assert "typesafe" in data["provider_order"]
    tmpl = next(t for t in PROVIDER_TEMPLATES if t["key"] == "typesafe")
    assert tmpl["base_url"] == "https://api.typesafe.ai/v1"
    assert tmpl["models_id_field"] == "name"


def test_typesafe_pricing_is_input_only():
    pricing = load_data()["pricing"]
    for model in ("jev-latest", "jev-preview", "jev-1.13.0"):
        rate = pricing[f"typesafe/{model}"]
        assert rate["input_cost_per_token"] == pytest.approx(4.2e-08)
        assert rate["output_cost_per_token"] == 0.0
