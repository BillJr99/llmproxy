"""The ``typesafe`` protocol: structured chat requests answered by Jev.

A chat request whose response_format JSON schema is made of decisions is turned
into a TypeSafe System One request (one question per field), and the typed
answers are rendered back as the JSON object the schema describes.
"""

from __future__ import annotations

import datetime
import importlib
import json
from pathlib import Path

import pytest

from llmproxy.dialects import get_outbound
from llmproxy.dialects.typesafe import to_systemone_request

ADAPTER = get_outbound("typesafe")


def _schema(properties: dict) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "triage", "strict": True,
        "schema": {"type": "object", "properties": properties,
                   "required": list(properties), "additionalProperties": False},
    }}


_TRIAGE = {
    "urgent": {"type": "boolean", "description": "Does this convey urgency?"},
    "p_churn": {"type": "number", "minimum": 0, "maximum": 1,
                "description": "Probability the customer cancels"},
    "department": {"type": "string", "enum": ["billing", "technical", "sales"],
                   "description": "Which team should handle this?"},
    "frustration": {"type": "integer", "description": "How frustrated is the customer?",
                    "oneOf": [{"const": 0, "description": "Calm"},
                              {"const": 1, "description": "Frustrated"},
                              {"const": 2, "description": "Very angry"}]},
}

_CHAT = {
    "model": "jev-latest",
    "messages": [
        {"role": "system", "content": "You triage support tickets."},
        {"role": "user", "content": "Help! My payouts have been failing for 3 days."},
    ],
    "response_format": _schema(_TRIAGE),
}


def _qid(kind, field, extra=None):
    return json.dumps([kind, field, extra])


# ---------------------------------------------------------------------------
# Request: schema -> questions
# ---------------------------------------------------------------------------

def test_request_maps_each_field_to_a_question():
    body = to_systemone_request(_CHAT)
    assert body["model"] == "jev-latest"
    assert body["state"] == [
        {"role": "system", "content": "You triage support tickets."},
        {"role": "user", "content": "Help! My payouts have been failing for 3 days."},
    ]
    q = body["questions"]
    assert q[_qid("b", "urgent")] == {"type": "noul", "instructions": "Does this convey urgency?"}
    assert q[_qid("p", "p_churn")]["type"] == "noul"
    assert q[_qid("c", "department")] == {
        "type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": None, "technical": None, "sales": None},
    }
    assert q[_qid("s", "frustration", [0, 1, 2])] == {
        "type": "score", "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    }


def test_oneof_strings_become_a_choice_with_a_rubric():
    body = to_systemone_request({**_CHAT, "response_format": _schema({"team": {
        "type": "string",
        "oneOf": [{"const": "billing", "description": "Payments, invoicing, refunds"},
                  {"const": "technical", "description": "Bugs, outages, integrations"}],
    }})})
    (qid, question), = body["questions"].items()
    assert json.loads(qid)[0] == "c"
    assert question["criteria"] == {"billing": "Payments, invoicing, refunds",
                                    "technical": "Bugs, outages, integrations"}


def test_bounded_integer_becomes_a_score_and_non_string_enum_a_json_choice():
    body = to_systemone_request({**_CHAT, "response_format": _schema({
        "stars": {"type": "integer", "minimum": 1, "maximum": 5},
        "tier": {"type": "integer", "enum": [10, 20, 30]},
    })})
    q = body["questions"]
    assert q[_qid("s", "stars", [1, 2, 3, 4, 5])]["criteria"] == ["1", "2", "3", "4", "5"]
    assert q[_qid("j", "tier")]["criteria"] == {"10": None, "20": None, "30": None}
    # A property without a description still gets a question to answer.
    assert "stars" in q[_qid("s", "stars", [1, 2, 3, 4, 5])]["instructions"]


def test_nullable_type_list_is_accepted():
    body = to_systemone_request({**_CHAT, "response_format": _schema(
        {"ok": {"type": ["boolean", "null"]}})})
    assert _qid("b", "ok") in body["questions"]


@pytest.mark.parametrize("payload, needle", [
    ({**_CHAT, "response_format": None}, "structured"),
    ({**_CHAT, "response_format": {"type": "json_object"}}, "structured"),
    ({**_CHAT, "response_format": _schema({"summary": {"type": "string"}})}, "free text"),
    ({**_CHAT, "response_format": _schema({"n": {"type": "integer", "minimum": 0, "maximum": 50}})},
     "levels"),
    ({**_CHAT, "response_format": _schema({"one": {"type": "string", "enum": ["only"]}})},
     "2 to 255"),
    ({**_CHAT, "tools": [{"type": "function", "function": {"name": "f"}}]}, "tools"),
    ({**_CHAT, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:,"}}]}]}, "text only"),
    ({**_CHAT, "messages": []}, "empty"),
])
def test_requests_jev_cannot_express_are_rejected(payload, needle):
    with pytest.raises(ValueError, match=needle):
        to_systemone_request(payload)


def test_build_request_targets_systemone_with_bearer_auth():
    url, headers, body = ADAPTER.build_request(
        "chat/completions", "https://api.typesafe.ai/v1", {"api_key": "ts-key"}, _CHAT,
        stream=False, forwarded_headers={"User-Agent": "llmproxy-test"},
    )
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"] == "Bearer ts-key"
    assert headers["User-Agent"] == "llmproxy-test"
    assert set(body) == {"model", "state", "questions"}


# ---------------------------------------------------------------------------
# Response: answers -> JSON object
# ---------------------------------------------------------------------------

_ANSWERS = {
    "model": "jev-1.13.0",
    "answers": {
        _qid("b", "urgent"): {"type": "noul", "noul": 0.95},
        _qid("p", "p_churn"): {"type": "noul", "noul": 0.31},
        _qid("c", "department"): {"type": "choice", "choice": "billing",
                                  "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
                                  "confidence": 0.81},
        _qid("s", "frustration", [0, 1, 2]): {"type": "score", "score": 1.05,
                                              "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                                              "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                                              "confidence": 0.92},
        _qid("s", "stars", [1, 2, 3, 4, 5]): {"type": "score", "score": 3.6},
        _qid("j", "tier"): {"type": "choice", "choice": "20"},
    },
    "usage": {"input_tokens": 318, "output_tokens": 34},
}


def test_answers_render_as_the_schema_object():
    out = json.loads(ADAPTER.translate_response(json.dumps(_ANSWERS).encode()))
    assert out["object"] == "chat.completion"
    assert out["model"] == "jev-1.13.0"
    assert out["choices"][0]["finish_reason"] == "stop"
    values = json.loads(out["choices"][0]["message"]["content"])
    assert values == {"urgent": True, "p_churn": 0.31, "department": "billing",
                      "frustration": 1, "stars": 5, "tier": 20}
    assert out["usage"] == {"prompt_tokens": 318, "completion_tokens": 34, "total_tokens": 352}
    # The calibrated answers behind each value stay available, keyed by field.
    assert out["typesafe"]["answers"]["department"]["confidence"] == 0.81


def test_low_probability_boolean_is_false():
    body = {"model": "jev-1.13.0", "answers": {_qid("b", "urgent"): {"type": "noul", "noul": 0.2}},
            "usage": {}}
    out = json.loads(ADAPTER.translate_response(json.dumps(body).encode()))
    assert json.loads(out["choices"][0]["message"]["content"]) == {"urgent": False}


def test_stream_is_one_chunk_then_done():
    chunks = list(ADAPTER.parse_stream(iter([json.dumps(_ANSWERS).encode()[:40],
                                             json.dumps(_ANSWERS).encode()[40:]])))
    assert chunks[-1] is None
    (chunk,) = chunks[:-1]
    assert chunk["object"] == "chat.completion.chunk"
    assert json.loads(chunk["choices"][0]["delta"]["content"])["department"] == "billing"
    assert chunk["usage"]["prompt_tokens"] == 318


# ---------------------------------------------------------------------------
# End to end through the proxy
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status: int, body: dict):
        self.status_code = status
        self.content = json.dumps(body).encode()
        self.headers = {"Content-Type": "application/json"}
        self.elapsed = datetime.timedelta(milliseconds=5)

    def iter_content(self, chunk_size=None):
        yield self.content

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def server(monkeypatch, tmp_path: Path):
    cfg = {
        "providers": {"typesafe": {
            "base_url": "https://api.typesafe.ai/v1", "api_key": "ts-key",
            "model_filter": None, "models_id_field": "name", "protocol": "typesafe",
        }},
        "believed_free": [], "model_reasoning": {}, "free_limits": {},
        "sync_believed_free_on_startup": False,
        "server": {"log_level": "ERROR", "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setenv("LLMPROXY_CONFIG", str(p))
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


def _reply_for(body: dict) -> dict:
    """What TypeSafe would answer: echo each question id with a plausible answer."""
    answers = {}
    for qid, q in body["questions"].items():
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.9}
        elif q["type"] == "choice":
            answers[qid] = {"type": "choice", "choice": next(iter(q["criteria"])), "confidence": 0.8}
        else:
            answers[qid] = {"type": "score", "score": 2.0, "confidence": 0.7}
    return {"model": "jev-1.13.0", "answers": answers,
            "usage": {"input_tokens": 300, "output_tokens": 20}}


def test_chat_completions_is_answered_by_jev(server, client, monkeypatch):
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        sent.update(url=url, json=json)
        return _FakeResp(200, _reply_for(json))

    recorded = []
    monkeypatch.setattr(server.requests, "post", fake_post)
    monkeypatch.setattr(server, "_record_usage",
                        lambda pn, um, **kw: recorded.append((pn, um, kw.get("usage"))))

    resp = client.post("/v1/chat/completions", json={**_CHAT, "model": "typesafe/jev-latest"})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert sent["url"] == "https://api.typesafe.ai/v1/systemone"
    assert sent["json"]["model"] == "jev-latest"
    values = json.loads(resp.get_json()["choices"][0]["message"]["content"])
    assert values == {"urgent": True, "p_churn": 0.9, "department": "billing", "frustration": 2}
    assert recorded and recorded[-1][2]["prompt_tokens"] == 300


def test_responses_api_text_format_is_answered_by_jev(server, client, monkeypatch):
    monkeypatch.setattr(server.requests, "post",
                        lambda url, headers=None, json=None, timeout=None, stream=False:
                        _FakeResp(200, _reply_for(json)))
    fmt = _schema({"urgent": _TRIAGE["urgent"]})["json_schema"]
    resp = client.post("/v1/responses", json={
        "model": "typesafe/jev-latest", "input": "Help! My payouts have been failing.",
        "text": {"format": {"type": "json_schema", **fmt}},
    })
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert '\\"urgent\\": true' in resp.get_data(as_text=True)


def test_streaming_chat_gets_the_answer_in_one_chunk(server, client, monkeypatch):
    monkeypatch.setattr(server.requests, "post",
                        lambda url, headers=None, json=None, timeout=None, stream=False:
                        _FakeResp(200, _reply_for(json)))
    resp = client.post("/v1/chat/completions",
                       json={**_CHAT, "model": "typesafe/jev-latest", "stream": True})
    text = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "billing" in text and "data: [DONE]" in text


def test_unstructured_chat_is_rejected_with_the_reason(server, client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not reach upstream")

    monkeypatch.setattr(server.requests, "post", boom)
    resp = client.post("/v1/chat/completions", json={
        "model": "typesafe/jev-latest", "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 400
    text = resp.get_data(as_text=True)
    assert "response_format" in text and "/v1/systemone" in text


def test_systemone_still_forwards_the_native_body_with_the_protocol_set(server, client, monkeypatch):
    native = {"model": "typesafe/jev-latest", "state": "Help!",
              "questions": {"is_urgent": {"type": "noul", "instructions": "Urgent?"}}}
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, json=json)
        return _FakeResp(200, {"model": "jev-1.13.0",
                               "answers": {"is_urgent": {"type": "noul", "noul": 0.9}},
                               "usage": {"input_tokens": 5, "output_tokens": 1}})

    monkeypatch.setattr(server.requests, "post", fake_post)
    resp = client.post("/v1/systemone", json=native)
    assert resp.status_code == 200
    assert sent["url"] == "https://api.typesafe.ai/v1/systemone"
    assert sent["json"] == {**native, "model": "jev-latest"}
    assert resp.get_json()["answers"]["is_urgent"]["noul"] == 0.9
