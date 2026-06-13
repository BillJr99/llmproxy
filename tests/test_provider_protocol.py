"""Tests for the provider ``protocol`` field plumbing and adapter tool handling."""

from __future__ import annotations

import json

from llmproxy.admin import _PROVIDER_PROTOCOLS, _clean_provider_payload
from llmproxy.dialects import get_inbound, get_outbound
from llmproxy.providers import get_provider_templates

# --- protocol config plumbing ----------------------------------------------

def test_admin_accepts_valid_protocol():
    cfg, err = _clean_provider_payload(
        {"base_url": "https://x/v1", "api_key": "k", "protocol": "anthropic"}, None)
    assert err is None
    assert cfg["protocol"] == "anthropic"


def test_admin_drops_openai_protocol_to_keep_configs_clean():
    cfg, err = _clean_provider_payload(
        {"base_url": "https://x/v1", "api_key": "k", "protocol": "openai"}, None)
    assert err is None
    assert "protocol" not in cfg


def test_admin_rejects_unknown_protocol():
    cfg, err = _clean_provider_payload(
        {"base_url": "https://x/v1", "api_key": "k", "protocol": "bogus"}, None)
    assert cfg is None
    assert "protocol" in err


def test_protocols_match_registered_adapters():
    for proto in _PROVIDER_PROTOCOLS:
        assert get_outbound(proto).name == proto


def test_templates_carry_protocol_for_native_providers():
    by_key = {t["key"]: t for t in get_provider_templates()}
    assert by_key["anthropic"].get("protocol") == "anthropic"
    assert by_key["gemini"].get("protocol") == "gemini"
    # OpenAI-compatible providers omit protocol (default openai)
    assert "protocol" not in by_key["scaleway"]


# --- adapter tool-call translation -----------------------------------------

def test_anthropic_inbound_tool_result_becomes_openai_tool_role():
    inb = get_inbound("anthropic")
    body = {
        "model": "m", "max_tokens": 10,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"city": "SF"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "72F"}]},
        ],
        "tools": [{"name": "get_weather", "description": "w",
                   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
    }
    canon = inb.to_canonical_request(body)
    roles = [m["role"] for m in canon["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assistant = canon["messages"][1]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "SF"}
    assert canon["messages"][2]["tool_call_id"] == "tu1"
    assert canon["tools"][0]["type"] == "function"


def test_anthropic_inbound_renders_tool_calls_as_tool_use_blocks():
    inb = get_inbound("anthropic")
    oai = {
        "id": "x", "model": "m",
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "lookup", "arguments": "{\"q\": 1}"}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    out = json.loads(inb.render_response(json.dumps(oai).encode()))
    assert out["stop_reason"] == "tool_use"
    tu = out["content"][0]
    assert tu["type"] == "tool_use"
    assert tu["name"] == "lookup"
    assert tu["input"] == {"q": 1}


def test_anthropic_outbound_builds_native_tool_request():
    out = get_outbound("anthropic")
    payload = {
        "model": "claude-x",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "f", "description": "d",
            "parameters": {"type": "object", "properties": {}}}}],
    }
    _url, _h, body = out.build_request(
        "chat/completions", "https://api.anthropic.com/v1", {"api_key": "k"},
        payload, stream=False, forwarded_headers={})
    assert body["tools"][0]["name"] == "f"
    assert "input_schema" in body["tools"][0]
    assert body["max_tokens"]  # required field defaulted
