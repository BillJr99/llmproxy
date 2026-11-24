"""Prompt caching: settings, per-provider request shaping, and usage counters.

Every upstream here is fake: requests are built and inspected in memory, and
responses are recorded provider payloads, so no test makes a network call.
"""

from __future__ import annotations

import json

import pytest

from llmproxy import caching as C
from llmproxy.dialects import anthropic as A
from llmproxy.dialects import gemini as G
from llmproxy.dialects import responses as R
from llmproxy.usage import extract_usage

ANTHROPIC = {"base_url": "https://api.anthropic.com/v1", "api_key": "k", "protocol": "anthropic"}
OPENROUTER = {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"}
OPENAI = {"base_url": "https://api.openai.com/v1", "api_key": "k", "protocol": "responses"}
GROQ = {"base_url": "https://api.groq.com/openai/v1", "api_key": "k"}

ON = C.resolve_settings({"prompt_caching": True}, {}, environ={})
ON_1H = C.resolve_settings({"prompt_caching": {"enabled": True, "ttl": "1h"}}, {}, environ={})


def _chat(**extra):
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "You are a careful assistant." * 20},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "a", "parameters": {}}},
            {"type": "function", "function": {"name": "b", "parameters": {}}},
        ],
    }
    body.update(extra)
    return body


def _markers(obj) -> list:
    found = []

    def walk(v):
        if isinstance(v, dict):
            if "cache_control" in v:
                found.append(v["cache_control"])
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(obj)
    return found


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def test_defaults_are_five_minutes_and_no_warming():
    s = C.resolve_settings({}, {}, environ={})
    assert s["enabled"] is True
    assert s["ttl"] == "5m"
    assert s["warming"] == "off"
    assert s["retention"] is None
    assert s["passthrough"] == "auto"


@pytest.mark.parametrize("raw,enabled,ttl", [
    ({"ttl": "off"}, False, "off"),
    ({"ttl": "5m"}, True, "5m"),
    ({"ttl": "1h"}, True, "1h"),
    (False, False, "off"),
    (True, True, "5m"),
    ({"enabled": False}, False, "off"),
    ({"enabled": False, "ttl": "1h"}, False, "off"),
    ({"enabled": True, "ttl": "1h"}, True, "1h"),
])
def test_each_ttl_value(raw, enabled, ttl):
    s = C.resolve_settings({"prompt_caching": raw}, {}, environ={})
    assert (s["enabled"], s["ttl"]) == (enabled, ttl)


def test_provider_ttl_overrides_global_off_and_vice_versa():
    s = C.resolve_settings({"prompt_caching": {"ttl": "off"}}, {"prompt_caching": {"ttl": "1h"}}, environ={})
    assert (s["enabled"], s["ttl"]) == (True, "1h")
    s = C.resolve_settings({"prompt_caching": {"ttl": "off"}}, {"prompt_caching": True}, environ={})
    assert (s["enabled"], s["ttl"]) == (True, "5m")
    s = C.resolve_settings({"prompt_caching": {"ttl": "1h"}}, {"prompt_caching": {"ttl": "off"}}, environ={})
    assert (s["enabled"], s["ttl"]) == (False, "off")
    # A provider bool keeps the inherited lifetime.
    s = C.resolve_settings({"prompt_caching": {"ttl": "1h"}}, {"prompt_caching": True}, environ={})
    assert s["ttl"] == "1h"


def test_env_ttl_off():
    s = C.resolve_settings({}, {}, environ={"LLMPROXY_PROMPT_CACHE_TTL": "off"})
    assert s["enabled"] is False


def test_warming_is_always_off():
    for value in ("idle", "streaming", True, "on", None):
        s = C.resolve_settings({"prompt_caching": {"warming": value}}, {}, environ={})
        assert s["warming"] == "off"


def test_off_adds_nothing_to_an_anthropic_request():
    _h, body = _anthropic_body(_chat())
    off = C.resolve_settings({"prompt_caching": {"ttl": "off"}}, {}, environ={})
    out, _ = C.apply("anthropic", ANTHROPIC, body, {}, off)
    assert _markers(out) == []


def test_default_five_minute_breakpoints():
    _h, body = _anthropic_body(_chat())
    out, _ = C.apply("anthropic", ANTHROPIC, body, {}, C.resolve_settings({}, {}, environ={}))
    assert _markers(out) and all(m == {"type": "ephemeral"} for m in _markers(out))


def test_global_provider_and_env_layers():
    cfg = {"prompt_caching": {"enabled": True, "ttl": "1h"}}
    assert C.resolve_settings(cfg, {}, environ={})["ttl"] == "1h"
    # A provider bool toggles only enabled and inherits the rest.
    s = C.resolve_settings(cfg, {"prompt_caching": False}, environ={})
    assert s["enabled"] is False and s["ttl"] == "off"
    s = C.resolve_settings(cfg, {"prompt_caching": True}, environ={})
    assert s["enabled"] is True and s["ttl"] == "1h"
    # Environment beats the file, the provider entry beats the environment.
    env = {"LLMPROXY_PROMPT_CACHING": "off", "LLMPROXY_PROMPT_CACHE_TTL": "5m"}
    assert C.resolve_settings(cfg, {}, environ=env)["enabled"] is False
    assert C.resolve_settings(cfg, {}, environ=env)["ttl"] == "off"
    assert C.resolve_settings(cfg, {}, environ={"LLMPROXY_PROMPT_CACHE_TTL": "5m"})["ttl"] == "5m"
    assert C.resolve_settings(cfg, {"prompt_caching": True}, environ=env)["enabled"] is True
    env = {"LLMPROXY_PROMPT_CACHE_RETENTION": "24h"}
    assert C.resolve_settings({}, {}, environ=env)["retention"] == "24h"


def test_invalid_values_fall_back_to_defaults():
    s = C.resolve_settings(
        {"prompt_caching": {"enabled": True, "ttl": "2d", "retention": "forever",
                            "inject_markers": "sometimes"}}, {}, environ={})
    assert s["ttl"] == "5m"
    assert s["retention"] is None
    assert s["inject_markers"] == "auto"


# ---------------------------------------------------------------------------
# Anthropic Messages upstream
# ---------------------------------------------------------------------------

def _anthropic_body(payload):
    url, headers, body = A.AnthropicOutbound().build_request(
        "chat/completions", "https://api.anthropic.com/v1", ANTHROPIC, payload,
        stream=False, forwarded_headers={})
    return headers, body


def test_disabled_anthropic_request_is_unchanged():
    _h, body = _anthropic_body(_chat())
    out, _ = C.apply("anthropic", ANTHROPIC, body, {},
                     C.resolve_settings({"prompt_caching": False}, {}, environ={}))
    assert out is body
    assert _markers(out) == []


def test_enabled_anthropic_gets_three_breakpoints():
    _h, body = _anthropic_body(_chat())
    out, _ = C.apply("anthropic", ANTHROPIC, body, {}, ON)
    assert out["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in out["tools"][0]
    assert out["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert out["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert len(_markers(out)) == 3 <= C.MAX_BREAKPOINTS
    # The original is not mutated (the cycling loop may retry it elsewhere).
    assert _markers(body) == []


def test_one_hour_ttl_rides_on_every_breakpoint():
    _h, body = _anthropic_body(_chat())
    out, headers = C.apply("anthropic", ANTHROPIC, body, {}, ON_1H)
    assert {json.dumps(m) for m in _markers(out)} == {json.dumps({"type": "ephemeral", "ttl": "1h"})}
    # The 1-hour cache is generally available; no beta header is added.
    assert "anthropic-beta" not in headers


def test_client_markers_are_left_alone():
    payload = _chat()
    payload["messages"][1]["content"] = [
        {"type": "text", "text": "first question", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    _h, body = _anthropic_body(payload)
    out, _ = C.apply("anthropic", ANTHROPIC, body, {}, ON)
    assert _markers(out) == [{"type": "ephemeral", "ttl": "1h"}]


def test_top_level_automatic_cache_control_round_trips():
    inbound = A._anthropic_to_openai_request({
        "model": "claude", "max_tokens": 10, "cache_control": {"type": "ephemeral"},
        "messages": [{"role": "user", "content": "hi"}]})
    assert inbound["cache_control"] == {"type": "ephemeral"}
    _h, body = _anthropic_body(inbound)
    assert body["cache_control"] == {"type": "ephemeral"}
    # It counts as the client's own choice: nothing else is added.
    out, _ = C.apply("anthropic", ANTHROPIC, body, {}, ON)
    assert _markers(out) == [{"type": "ephemeral"}]


def test_tool_result_marker_survives_the_round_trip():
    inbound = A._anthropic_to_openai_request({
        "model": "claude", "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "a", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "result",
                 "cache_control": {"type": "ephemeral"}}]},
        ]})
    _h, body = _anthropic_body(inbound)
    last = body["messages"][-1]["content"][-1]
    assert last["type"] == "tool_result"
    assert last["cache_control"] == {"type": "ephemeral"}


def test_last_turn_marker_skips_thinking_and_empty_blocks():
    body = {"model": "m", "messages": [
        {"role": "assistant", "content": [
            {"type": "text", "text": "answer"},
            {"type": "thinking", "thinking": "...", "signature": "s"},
            {"type": "text", "text": ""}]}]}
    out = C.add_anthropic_breakpoints(body)
    blocks = out["messages"][0]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1] and "cache_control" not in blocks[2]


def test_anthropic_beta_header_reaches_the_anthropic_upstream():
    _u, headers, _b = A.AnthropicOutbound().build_request(
        "chat/completions", "https://api.anthropic.com/v1", ANTHROPIC, _chat(),
        stream=False, forwarded_headers={"anthropic-beta": "some-feature-2026-01-01"})
    assert headers["anthropic-beta"] == "some-feature-2026-01-01"


# ---------------------------------------------------------------------------
# OpenAI-compatible upstreams
# ---------------------------------------------------------------------------

def test_openrouter_claude_gets_content_part_markers():
    body = _chat(model="anthropic/claude-sonnet-5")
    out, _ = C.apply("openai", OPENROUTER, body, {}, ON_1H)
    msgs = out["messages"]
    marker = {"type": "ephemeral", "ttl": "1h"}
    assert msgs[0]["content"][-1]["cache_control"] == marker
    assert msgs[-1]["content"][-1]["cache_control"] == marker
    assert isinstance(body["messages"][0]["content"], str)  # original untouched


def test_openrouter_automatic_cache_families_are_not_marked():
    for model in ("deepseek/deepseek-v4", "openai/gpt-5.5", "google/gemini-3.1-pro"):
        body = _chat(model=model)
        out, _ = C.apply("openai", OPENROUTER, body, {}, ON)
        assert _markers(out) == [], model


def test_generic_openai_compatible_provider_is_untouched_unless_opted_in():
    body = _chat(model="claude-sonnet-5")
    out, _ = C.apply("openai", GROQ, body, {}, ON)
    assert out == body
    forced = C.resolve_settings({"prompt_caching": True},
                                {"prompt_caching": {"inject_markers": True}}, environ={})
    out, _ = C.apply("openai", GROQ, _chat(model="any"), {}, forced)
    assert len(_markers(out)) == 2


def test_openai_gets_cache_key_and_retention():
    s = C.resolve_settings({"prompt_caching": {"enabled": True, "retention": "24h"}}, {}, environ={})
    body = R._canonical_to_responses_request(_chat(model="gpt-5.5"))
    out, _ = C.apply("responses", OPENAI, body, {}, s, affinity_key="conversation-root")
    assert out["prompt_cache_retention"] == "24h"
    assert out["prompt_cache_key"] == C.derive_cache_key("conversation-root")
    assert out["prompt_cache_key"].startswith("llmproxy-") and len(out["prompt_cache_key"]) < 64


def test_openai_client_fields_win():
    s = C.resolve_settings({"prompt_caching": {"enabled": True, "retention": "24h"}}, {}, environ={})
    body = _chat(model="gpt-5.5", prompt_cache_key="mine", prompt_cache_retention="in_memory")
    out, _ = C.apply("openai", {**OPENAI, "protocol": "openai"}, body, {}, s, affinity_key="x")
    assert out["prompt_cache_key"] == "mine"
    assert out["prompt_cache_retention"] == "in_memory"


def test_cache_key_and_retention_only_go_to_openai_itself():
    s = C.resolve_settings({"prompt_caching": {"enabled": True, "retention": "24h"}}, {}, environ={})
    out, _ = C.apply("openai", GROQ, _chat(), {}, s, affinity_key="x")
    assert "prompt_cache_key" not in out and "prompt_cache_retention" not in out


def test_passthrough_false_strips_client_cache_fields():
    s = C.resolve_settings({}, {"prompt_caching": {"passthrough": False}}, environ={})
    body = _chat(prompt_cache_key="k", prompt_cache_retention="24h",
                 prompt_cache_options={"ttl": "1h"}, cache_control={"type": "ephemeral"})
    body["messages"][1]["content"] = [
        {"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}]
    out, _ = C.apply("openai", GROQ, body, {}, s)
    for key in ("prompt_cache_key", "prompt_cache_retention", "prompt_cache_options", "cache_control"):
        assert key not in out
    assert _markers(out) == []
    assert out["messages"][1]["content"][0]["text"] == "q"


def test_auto_passthrough_relays_everything_to_caching_providers():
    body = _chat(model="anthropic/claude-sonnet-5", prompt_cache_options={"ttl": "1h"},
                 cache_control={"type": "ephemeral"})
    out, _ = C.apply("openai", OPENROUTER, body, {}, C.resolve_settings({}, {}, environ={}))
    assert out["prompt_cache_options"] == {"ttl": "1h"}
    assert out["cache_control"] == {"type": "ephemeral"}


def test_auto_passthrough_drops_risky_fields_on_generic_servers():
    body = _chat(prompt_cache_options={"ttl": "1h"}, prompt_cache_retention="24h",
                 cache_control={"type": "ephemeral"}, prompt_cache_key="k")
    body["messages"].append({"role": "tool", "tool_call_id": "t", "content": [
        {"type": "text", "text": "result", "cache_control": {"type": "ephemeral"}}]})
    body["messages"][1]["content"] = [
        {"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}]
    out, _ = C.apply("openai", GROQ, body, {}, C.resolve_settings({}, {}, environ={}))
    for key in ("prompt_cache_options", "prompt_cache_retention", "cache_control"):
        assert key not in out
    assert out["prompt_cache_key"] == "k"                       # always relayed
    assert out["messages"][-1]["content"] == "result"           # tool content flattened
    assert out["messages"][1]["content"][0]["cache_control"]    # user markers kept
    assert isinstance(body["messages"][-1]["content"], list)    # original untouched


def test_passthrough_true_relays_everything_everywhere():
    s = C.resolve_settings({}, {"prompt_caching": {"passthrough": True}}, environ={})
    body = _chat(prompt_cache_options={"ttl": "1h"})
    out, _ = C.apply("openai", GROQ, body, {}, s)
    assert out["prompt_cache_options"] == {"ttl": "1h"}


def test_passthrough_false_strips_responses_input_markers():
    s = C.resolve_settings({}, {"prompt_caching": {"passthrough": False}}, environ={})
    body = {"model": "m", "input": [{"role": "user", "content": [
        {"type": "input_text", "text": "x", "cache_control": {"type": "ephemeral"}}]}]}
    out, _ = C.apply("responses", OPENAI, body, {}, s)
    assert _markers(out) == []


def test_openrouter_marks_the_newest_tool_result():
    body = _chat(model="anthropic/claude-sonnet-5")
    body["messages"] += [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t", "type": "function", "function": {"name": "a", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t", "content": "tool output"},
    ]
    out, _ = C.apply("openai", OPENROUTER, body, {}, ON)
    assert out["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_non_chat_endpoints_are_never_shaped():
    body = {"model": "e", "input": "text"}
    out, _ = C.apply("openai", OPENROUTER, body, {}, ON, endpoint="embeddings")
    assert out is body


# ---------------------------------------------------------------------------
# Responses inbound / outbound passthrough
# ---------------------------------------------------------------------------

def test_responses_inbound_keeps_cache_fields():
    payload = R._responses_to_canonical({
        "model": "m", "input": "hi", "prompt_cache_key": "k",
        "prompt_cache_retention": "24h", "prompt_cache_options": {"mode": "implicit"},
        "cache_control": {"type": "ephemeral"}})
    assert payload["prompt_cache_key"] == "k"
    assert payload["prompt_cache_retention"] == "24h"
    assert payload["prompt_cache_options"] == {"mode": "implicit"}
    assert payload["cache_control"] == {"type": "ephemeral"}


def test_responses_outbound_relays_retention_and_options():
    body = R._canonical_to_responses_request(_chat(
        prompt_cache_retention="24h", prompt_cache_options={"ttl": "12h"}))
    assert body["prompt_cache_retention"] == "24h"
    assert body["prompt_cache_options"] == {"ttl": "12h"}


# ---------------------------------------------------------------------------
# usage counters (recorded provider responses)
# ---------------------------------------------------------------------------

ANTHROPIC_RESPONSE = {
    "id": "msg_1", "type": "message", "model": "claude-opus-5-5",
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 1800,
              "cache_creation_input_tokens": 300,
              "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 300}},
}


def test_anthropic_usage_counts_cached_tokens_in_prompt_tokens():
    out = A._anthropic_response_to_openai(ANTHROPIC_RESPONSE)
    u = out["usage"]
    assert u["prompt_tokens"] == 12 + 1800 + 300
    assert u["total_tokens"] == 12 + 1800 + 300 + 5
    assert u["prompt_tokens_details"] == {"cached_tokens": 1800, "cache_write_tokens": 300}
    assert u["cache_creation"]["ephemeral_1h_input_tokens"] == 300
    got = extract_usage(out)
    assert got["cache_read_tokens"] == 1800
    assert got["cache_creation_tokens"] == 300


def test_anthropic_client_sees_anthropic_counters_again():
    canonical = A._anthropic_response_to_openai(ANTHROPIC_RESPONSE)
    back = A._openai_response_to_anthropic(canonical)
    assert back["usage"]["input_tokens"] == 12
    assert back["usage"]["cache_read_input_tokens"] == 1800
    assert back["usage"]["cache_creation_input_tokens"] == 300


def test_anthropic_stream_usage_carries_cache_counters():
    events = [
        ("message_start", {"type": "message_start", "message": {
            "model": "claude", "usage": {"input_tokens": 3, "cache_read_input_tokens": 500,
                                         "cache_creation_input_tokens": 0}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "hi"}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 2}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    raw = "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()
    chunks = [c for c in A.AnthropicOutbound().parse_stream([raw]) if c]
    usage = chunks[-1]["usage"]
    assert usage["prompt_tokens"] == 503
    assert usage["prompt_tokens_details"]["cached_tokens"] == 500


def test_openrouter_usage_counters():
    got = extract_usage({"usage": {"prompt_tokens": 2000, "completion_tokens": 10, "total_tokens": 2010,
                                   "prompt_tokens_details": {"cached_tokens": 1500,
                                                             "cache_write_tokens": 100}}})
    assert got["cache_read_tokens"] == 1500
    assert got["cache_creation_tokens"] == 100


def test_deepseek_usage_counters():
    got = extract_usage({"usage": {"prompt_tokens": 900, "completion_tokens": 10, "total_tokens": 910,
                                   "prompt_cache_hit_tokens": 640, "prompt_cache_miss_tokens": 260}})
    assert got["cache_read_tokens"] == 640
    assert "cache_creation_tokens" not in got


def test_gemini_usage_counters():
    usage = G._usage({"promptTokenCount": 5000, "candidatesTokenCount": 20,
                      "totalTokenCount": 5020, "cachedContentTokenCount": 4096})
    assert usage["prompt_tokens_details"] == {"cached_tokens": 4096}
    assert extract_usage({"usage": usage})["cache_read_tokens"] == 4096


def test_responses_usage_cache_write_round_trip():
    canonical = R._responses_usage_to_canonical({
        "input_tokens": 100, "output_tokens": 5,
        "input_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 10}})
    assert canonical["prompt_tokens_details"] == {"cached_tokens": 80, "cache_write_tokens": 10}
    back = R._usage_to_responses(canonical)
    assert back["input_tokens_details"] == {"cached_tokens": 80, "cache_write_tokens": 10}


def test_no_cache_counters_when_none_reported():
    got = extract_usage({"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    assert "cache_read_tokens" not in got and "cache_creation_tokens" not in got


# ---------------------------------------------------------------------------
# /v1/usage and the server hook
# ---------------------------------------------------------------------------

@pytest.fixture
def caching_server(tmp_path, monkeypatch):
    from tests.test_usage_endpoint import _load_server_with_config
    cfg = {
        "providers": {
            "anthropic": {**ANTHROPIC},
            "groq": {**GROQ, "model_filter": None},
        },
        "prompt_caching": {"enabled": True, "ttl": "1h"},
        "sync_believed_free_on_startup": False,
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5, "models_cache_ttl": 0,
                   "response_cache_ttl": 0},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    server = _load_server_with_config(monkeypatch, path)
    server._reset_usage()
    return server


def test_usage_report_includes_cache_fields(caching_server):
    s = caching_server
    s._record_usage("anthropic", "claude-opus-5-5",
                    usage=extract_usage(A._anthropic_response_to_openai(ANTHROPIC_RESPONSE)),
                    config=s.load_config())
    body = s.app.test_client().get("/v1/usage").get_json()
    m = body["models"][0]
    assert m["cache_read_tokens"] == 1800
    assert m["cache_creation_tokens"] == 300
    assert body["totals"]["cache_read_tokens"] == 1800
    assert body["totals"]["cache_creation_tokens"] == 300


def test_streaming_raw_usage_is_counted(caching_server):
    """The translated streaming path records the raw canonical usage block."""
    s = caching_server
    raw = A._anthropic_response_to_openai(ANTHROPIC_RESPONSE)["usage"]
    s._record_usage("anthropic", "claude-opus-5-5", usage=raw, config=s.load_config(),
                    count_request=False)
    m = s.app.test_client().get("/v1/usage").get_json()["models"][0]
    assert m["cache_read_tokens"] == 1800
    assert m["cache_creation_tokens"] == 300


def test_empty_usage_report_has_zero_cache_totals(caching_server):
    body = caching_server.app.test_client().get("/v1/usage").get_json()
    assert body["totals"]["cache_read_tokens"] == 0
    assert body["totals"]["cache_creation_tokens"] == 0


def test_server_hook_shapes_the_anthropic_request(caching_server):
    from llmproxy.dialects import get_outbound
    s = caching_server
    cfg = s.load_config()["providers"]["anthropic"]
    _u, _h, body = s._build_upstream_request(
        get_outbound("anthropic"), "chat/completions", "https://api.anthropic.com/v1",
        cfg, _chat(model="claude-opus-5-5"), stream=False, forwarded_headers={})
    assert len(_markers(body)) == 3
    assert all(m.get("ttl") == "1h" for m in _markers(body))


def test_server_hook_leaves_other_providers_alone(caching_server):
    from llmproxy.dialects import get_outbound
    s = caching_server
    cfg = s.load_config()["providers"]["groq"]
    payload = _chat(model="llama")
    _u, _h, body = s._build_upstream_request(
        get_outbound("openai"), "chat/completions", GROQ["base_url"], cfg, payload,
        stream=False, forwarded_headers={})
    assert body == payload


def test_sqlite_backend_records_cache_counters(tmp_path):
    from llmproxy.state import SqliteState
    be = SqliteState(str(tmp_path / "state.db"))
    be.record_usage("p/m", requests=1, prompt=100, completion=1, total=101,
                    cache_read=60, cache_creation=20)
    be.record_usage("p/m", prompt=100, completion=1, total=101, cache_read=40)
    rows = dict(be.usage_rows())
    assert rows["p/m"]["cache_read_tokens"] == 100
    assert rows["p/m"]["cache_creation_tokens"] == 20
