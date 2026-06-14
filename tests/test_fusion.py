"""Tests for the fusion (multi-model deliberation) pipeline.

Split into (1) pure helpers in llmproxy.fusion exercised directly, and
(2) the _proxy_fusion orchestration exercised through a reloaded server with a
monkeypatched _proxy_request / _proxy_streaming, mirroring test_capabilities.py.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from flask import Response

from llmproxy import fusion

# ── pure helpers ────────────────────────────────────────────────────────────

def test_get_fusion_config_defaults_and_sanitize():
    assert fusion.get_fusion_config({})["panel_size"] == 4
    assert fusion.get_fusion_config({"fusion": "nonsense"})["enabled"] is True
    cfg = fusion.get_fusion_config({"fusion": {"panel_size": 1, "diversity": "weird",
                                               "forced_capability": "x", "report": 5}})
    assert cfg["panel_size"] == fusion.MIN_PANEL  # clamped up
    assert cfg["diversity"] == "provider"
    assert cfg["forced_capability"] == "restrict"
    assert cfg["report"] == {"metadata": True}


def test_select_panel_prefers_distinct_providers():
    cands = [
        ("a", {}, "m1"), ("a", {}, "m2"), ("b", {}, "m3"), ("c", {}, "m4"),
    ]
    panel = fusion.select_panel(cands, 3, prefer_diversity=True)
    providers = [c[0] for c in panel]
    assert providers == ["a", "b", "c"]  # one per provider before doubling up


def test_select_panel_fills_when_not_enough_distinct():
    cands = [("a", {}, "m1"), ("a", {}, "m2")]
    panel = fusion.select_panel(cands, 2, prefer_diversity=True)
    assert [c[2] for c in panel] == ["m1", "m2"]  # second slot filled from leftovers


def test_select_panel_no_diversity_takes_prefix():
    cands = [("a", {}, "m1"), ("a", {}, "m2"), ("b", {}, "m3")]
    panel = fusion.select_panel(cands, 2, prefer_diversity=False)
    assert [c[2] for c in panel] == ["m1", "m2"]


def test_parse_analysis_fenced_and_prose_and_garbage():
    fenced = "```json\n{\"consensus\": [\"x\"]}\n```"
    assert fusion.parse_analysis(fenced) == {"consensus": ["x"]}
    prose = "Here is my analysis: {\"contradictions\": []} hope it helps"
    assert fusion.parse_analysis(prose) == {"contradictions": []}
    assert fusion.parse_analysis("no json here") is None
    assert fusion.parse_analysis("") is None


def test_extract_message_text_variants():
    s = json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()
    assert fusion.extract_message_text(s) == "hello"
    parts = json.dumps({"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}).encode()
    assert fusion.extract_message_text(parts) == "ab"
    assert fusion.extract_message_text(b"{bad json") == ""
    assert fusion.extract_message_text(json.dumps({"choices": []}).encode()) == ""


def test_build_messages_shape():
    orig = [{"role": "user", "content": "Q?"}]
    panel = [{"label": "a/m1", "content": "ans1"}, {"label": "b/m2", "content": "ans2"}]
    jm = fusion.build_judge_messages(orig, panel)
    assert jm[0]["role"] == "system" and "impartial judge" in jm[0]["content"]
    assert orig[0] in jm and "ans1" in jm[-1]["content"]
    sm = fusion.build_synthesizer_messages(orig, panel, {"consensus": ["c"]})
    assert "synthesizer" in sm[0]["content"]
    assert "consensus" in sm[-1]["content"]
    sm_none = fusion.build_synthesizer_messages(orig, panel, None)
    assert "comparison was unavailable" in sm_none[-1]["content"]


def test_build_and_inject_report():
    rep = fusion.build_report(panel_used=["a/m1"], judge_model="b/m2",
                              synthesizer_model="c/m3", failed_models=[],
                              analysis={"consensus": []}, fell_back=False, free=True)
    assert rep["object"] == "fusion.report" and rep["free"] is True
    body = json.dumps({"choices": [{"message": {"content": "x"}}]}).encode()
    out = json.loads(fusion.inject_report(body, rep))
    assert out["llmproxy_fusion"]["panel"] == ["a/m1"]


# ── orchestration ─────────────────────────────────────────────────────────

def _load_server(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    return server_mod


@pytest.fixture
def fusion_config(tmp_path: Path) -> Path:
    cfg = {
        "providers": {
            "pa": {"base_url": "http://pa.example/v1", "api_key": "k", "model_filter": None},
            "pb": {"base_url": "http://pb.example/v1", "api_key": "k", "model_filter": None},
            "pc": {"base_url": "http://pc.example/v1", "api_key": "k", "model_filter": None},
            "pd": {"base_url": "http://pd.example/v1", "api_key": "k", "model_filter": None},
        },
        "believed_free": ["pa/free-1", "pb/free-2"],
        "model_capabilities": {"pa/tool-1": ["tools"], "pb/tool-2": ["tools"]},
        "fusion": {"enabled": True, "panel_size": 4, "diversity": "provider"},
        "sync_believed_free_on_startup": False,
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def server(monkeypatch, fusion_config):
    return _load_server(monkeypatch, fusion_config)


def _seed(server, routes: dict[str, tuple[str, str]]):
    with server._model_route_cache_lock:
        server._model_route_cache.clear()
        server._model_route_cache.update(routes)


def _chat(content: str, status: int = 200) -> Response:
    body = json.dumps({
        "id": "x", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
    return Response(body, status=status, content_type="application/json")


def _fake_request_factory(judge_text='{"consensus": ["agree"]}', synth_text="FINAL",
                          fail_panel=(), fail_synth=False):
    """Build a fake _proxy_request that routes by the system prompt it sees."""
    seen = {"panel": [], "judge": [], "synth": []}

    def fake(endpoint, pn, cfg, payload, timeout):
        msgs = payload.get("messages", [])
        sys = msgs[0]["content"] if msgs and msgs[0].get("role") == "system" else ""
        if "impartial judge" in sys:
            seen["judge"].append(payload["model"])
            return _chat(judge_text)
        if "synthesizer" in sys:
            seen["synth"].append(payload["model"])
            return _chat(synth_text, status=500 if fail_synth else 200)
        seen["panel"].append(payload["model"])
        if payload["model"] in fail_panel:
            return _chat("", status=503)
        return _chat(f"answer from {payload['model']}")

    return fake, seen



def _run(server, model, payload, stream=False):
    """Invoke _proxy_fusion inside an app/request context (jsonify/_error need it)."""
    with server.app.test_request_context():
        return server._proxy_fusion(
            "chat/completions", model, payload,
            server.load_config(), server.get_inbound("openai"), stream,
        )


def test_fusion_fans_out_judges_and_synthesizes(server, monkeypatch):
    _seed(server, {f"p{x}__m{x}": (f"p{x}", f"m{x}") for x in ("a", "b", "c", "d")})
    fake, seen = _fake_request_factory()
    monkeypatch.setattr(server, "_proxy_request", fake)

    payload = {"model": "llmproxy__fusion", "messages": [{"role": "user", "content": "Q?"}]}
    resp = _run(server, "llmproxy__fusion", payload)
    assert resp.status_code == 200
    assert len(seen["panel"]) == 4 and len(seen["judge"]) == 1 and len(seen["synth"]) == 1
    data = json.loads(resp.get_data())
    assert data["choices"][0]["message"]["content"] == "FINAL"
    rep = data["llmproxy_fusion"]
    assert len(rep["panel"]) == 4 and rep["fell_back"] is False
    assert rep["analysis"] == {"consensus": ["agree"]}
    assert "X-LLMProxy-Fusion" in resp.headers


def test_fusion_judge_failure_still_synthesizes(server, monkeypatch):
    _seed(server, {f"p{x}__m{x}": (f"p{x}", f"m{x}") for x in ("a", "b", "c", "d")})
    fake, seen = _fake_request_factory(judge_text="not json at all")
    monkeypatch.setattr(server, "_proxy_request", fake)
    payload = {"model": "llmproxy__fusion", "messages": [{"role": "user", "content": "Q?"}]}
    resp = _run(server, "llmproxy__fusion", payload)
    data = json.loads(resp.get_data())
    assert data["choices"][0]["message"]["content"] == "FINAL"  # synth still ran
    assert "analysis" not in data["llmproxy_fusion"]  # judge produced none


def test_fusion_synth_failure_falls_back_to_panel(server, monkeypatch):
    _seed(server, {f"p{x}__m{x}": (f"p{x}", f"m{x}") for x in ("a", "b", "c", "d")})
    fake, seen = _fake_request_factory(fail_synth=True)
    monkeypatch.setattr(server, "_proxy_request", fake)
    payload = {"model": "llmproxy__fusion", "messages": [{"role": "user", "content": "Q?"}]}
    resp = _run(server, "llmproxy__fusion", payload)
    assert resp.status_code == 200
    data = json.loads(resp.get_data())
    assert data["choices"][0]["message"]["content"].startswith("answer from")
    assert data["llmproxy_fusion"]["fell_back"] is True


def test_fusion_all_panel_fail_errors(server, monkeypatch):
    _seed(server, {f"p{x}__m{x}": (f"p{x}", f"m{x}") for x in ("a", "b", "c", "d")})
    fake, _ = _fake_request_factory(fail_panel={"ma", "mb", "mc", "md"})
    monkeypatch.setattr(server, "_proxy_request", fake)
    payload = {"model": "llmproxy__fusion", "messages": [{"role": "user", "content": "Q?"}]}
    resp = _run(server, "llmproxy__fusion", payload)
    assert resp.status_code == 503


def test_fusion_backfills_failed_panel_from_reserve(server, monkeypatch):
    # Fixed ordered pool of 6 distinct providers, panel_size 4. The four chosen
    # members all fail; the panel should be rebuilt from the two-member reserve so
    # the request still succeeds, with the four failures recorded for provenance.
    pool = [(f"p{x}", {}, f"m{x}") for x in range(6)]
    monkeypatch.setattr(server, "_fusion_pool", lambda *a, **k: pool)
    fake, seen = _fake_request_factory(fail_panel={"m0", "m1", "m2", "m3"})
    monkeypatch.setattr(server, "_proxy_request", fake)
    payload = {"model": "llmproxy__fusion", "messages": [{"role": "user", "content": "Q?"}]}
    resp = _run(server, "llmproxy__fusion", payload)
    assert resp.status_code == 200
    rep = json.loads(resp.get_data())["llmproxy_fusion"]
    assert rep["panel"] == ["p4/m4", "p5/m5"]  # reserve backfilled the failed slots
    assert {f["model"] for f in rep["failed_models"]} == {"p0/m0", "p1/m1", "p2/m2", "p3/m3"}
    assert seen["panel"] == ["m0", "m1", "m2", "m3", "m4", "m5"]  # every slot attempted once


def test_fusion_all_panel_fail_error_lists_reasons(server, monkeypatch):
    _seed(server, {f"p{x}__m{x}": (f"p{x}", f"m{x}") for x in ("a", "b", "c", "d")})
    fake, _ = _fake_request_factory(fail_panel={"ma", "mb", "mc", "md"})
    monkeypatch.setattr(server, "_proxy_request", fake)
    payload = {"model": "llmproxy__fusion", "messages": [{"role": "user", "content": "Q?"}]}
    resp = _run(server, "llmproxy__fusion", payload)
    assert resp.status_code == 503
    msg = json.loads(resp.get_data())["error"]["message"]
    assert "Panel failures:" in msg and "status 503" in msg  # diagnosable, not opaque


def test_fusion_forced_tools_restricts_panel(server, monkeypatch):
    # Only pa/tool-1 and pb/tool-2 carry the tools capability.
    _seed(server, {"pa__tool-1": ("pa", "tool-1"), "pb__tool-2": ("pb", "tool-2"),
                   "pc__plain": ("pc", "plain"), "pd__plain": ("pd", "plain")})
    fake, seen = _fake_request_factory()
    monkeypatch.setattr(server, "_proxy_request", fake)
    payload = {"model": "llmproxy__fusion", "messages": [{"role": "user", "content": "Q?"}],
               "tools": [{"type": "function", "function": {"name": "x"}}], "tool_choice": "required"}
    _run(server, "llmproxy__fusion", payload)
    assert set(seen["panel"]) <= {"tool-1", "tool-2"}  # incapable models excluded


def test_fusion_free_uses_free_pool(server, monkeypatch):
    _seed(server, {"pa__free-1": ("pa", "free-1"), "pb__free-2": ("pb", "free-2"),
                   "pc__paid": ("pc", "paid")})
    fake, seen = _fake_request_factory()
    monkeypatch.setattr(server, "_proxy_request", fake)
    payload = {"model": "llmproxy__fusion/free", "messages": [{"role": "user", "content": "Q?"}]}
    _run(server, "llmproxy__fusion/free", payload)
    assert set(seen["panel"]) <= {"free-1", "free-2"}  # paid model never on the free panel


def test_fusion_streaming_sets_header(server, monkeypatch):
    _seed(server, {f"p{x}__m{x}": (f"p{x}", f"m{x}") for x in ("a", "b", "c", "d")})
    fake, _ = _fake_request_factory()
    monkeypatch.setattr(server, "_proxy_request", fake)
    monkeypatch.setattr(server, "_proxy_streaming",
                        lambda *a, **k: Response("data: {}\n\n", content_type="text/event-stream"))
    payload = {"model": "llmproxy__fusion", "stream": True,
               "messages": [{"role": "user", "content": "Q?"}]}
    resp = _run(server, "llmproxy__fusion", payload, stream=True)
    assert resp.headers["X-LLMProxy-Fusion"]
    assert resp.content_type.startswith("text/event-stream")


def test_fusion_panel_threads_inherit_request_context(server, monkeypatch):
    # Regression: the panel fans out on worker threads, but _proxy_request reads
    # request.headers (via _forwarded_client_headers). Without propagating the
    # request context into the workers, every panel member raised "working outside
    # of request context" and the whole panel collapsed to a 503. Assert the panel
    # threads see the active request context *and* its forwarded headers.
    _seed(server, {f"p{x}__m{x}": (f"p{x}", f"m{x}") for x in ("a", "b", "c", "d")})
    from flask import has_request_context

    seen_titles: list[str] = []

    def fake(endpoint, pn, cfg, payload, timeout):
        msgs = payload.get("messages", [])
        sys = msgs[0]["content"] if msgs and msgs[0].get("role") == "system" else ""
        if "impartial judge" in sys:
            return _chat('{"consensus": ["agree"]}')
        if "synthesizer" in sys:
            return _chat("FINAL")
        # Panel call: exercise the request-context-bound header forwarding the real
        # _proxy_request performs, from inside the worker thread.
        assert has_request_context(), "panel worker lost the request context"
        seen_titles.append(server._forwarded_client_headers().get("X-Title", ""))
        return _chat(f"answer from {payload['model']}")

    monkeypatch.setattr(server, "_proxy_request", fake)
    payload = {"model": "llmproxy__fusion", "messages": [{"role": "user", "content": "Q?"}]}
    with server.app.test_request_context(headers={"X-Title": "regression"}):
        resp = server._proxy_fusion(
            "chat/completions", "llmproxy__fusion", payload,
            server.load_config(), server.get_inbound("openai"), False,
        )
    assert resp.status_code == 200
    assert seen_titles and all(t == "regression" for t in seen_titles)


def test_fusion_models_advertised(server):
    # Advertisement keys on the candidate-selector counts; verify both pools
    # clear the MIN_PANEL bar so llmproxy__fusion and llmproxy__fusion/free show.
    _seed(server, {"pa__free-1": ("pa", "free-1"), "pb__free-2": ("pb", "free-2"),
                   "pc__paid": ("pc", "paid"), "pd__paid2": ("pd", "paid2")})
    assert len(server._get_all_model_candidates()) >= fusion.MIN_PANEL
    assert len(server._get_free_model_candidates()) >= fusion.MIN_PANEL
