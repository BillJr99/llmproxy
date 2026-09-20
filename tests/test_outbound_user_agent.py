"""llmproxy identifies itself on every outbound request.

It had no identity at all. Where a client sent no User-Agent, whatever HTTP
library was in use filled one in, so upstreams saw `python-requests/2.33.1` — a
library default leaking out rather than a decision. Where a client DID send one
it was relayed verbatim, so a caller using urllib got a Cloudflare block page
instead of an answer: measured through the proxy against a real provider,
`Python-urllib/3.11` returns 403 and 4KB of HTML while `curl/8.19.0` and
`OpenAI/Python 2.24.0` return 200.

A caller's choice of HTTP library should not decide whether an upstream
responds. These tests pin both halves of the rule: bare library defaults are
replaced, and anything naming a product is left alone.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from llmproxy import USER_AGENT


def _load_server(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


def _server(tmp_path: Path, monkeypatch, **server_over):
    cfg = {
        "providers": {"p1": {"base_url": "http://p1.example/v1", "api_key": "k"}},
        "sync_believed_free_on_startup": False,
        "server": {"log_level": "ERROR", **server_over},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return _load_server(monkeypatch, tmp_path / "config.json")


@pytest.fixture
def S(tmp_path: Path, monkeypatch):
    return _server(tmp_path, monkeypatch)


# ── the rewrite half ────────────────────────────────────────────────────────

# Bare runtime and library defaults. `Python-urllib/3.11` is the one measured
# getting a Cloudflare 403 through the proxy; the rest are the same shape.
GENERIC = [
    "Python-urllib/3.11",
    "python-urllib/3.12",
    "python-requests/2.33.1",
    "python-httpx/0.27.0",
    "httpx/0.28.1",
    "aiohttp/3.9.5",
    "Go-http-client/1.1",
    "Java/17.0.2",
    "okhttp/4.12.0",
    "libwww-perl/6.72",
    "node-fetch/2.6.7",
    "axios/1.7.2",
    "Apache-HttpClient/4.5.13",
    "GuzzleHttp/7",
]


@pytest.mark.parametrize("ua", GENERIC)
def test_a_bare_library_default_is_replaced_with_our_identity(S, ua):
    """The regression. These identify an HTTP stack, not a client, and they are
    exactly what CDN bot filters match on."""
    assert S._outbound_user_agent(ua) == USER_AGENT


def test_a_missing_user_agent_yields_our_identity_not_the_library_default(S):
    """The case that sent `python-requests/2.33.1`: nothing to relay, so the
    HTTP library filled one in and llmproxy went out unidentified."""
    for absent in (None, "", "   "):
        assert S._outbound_user_agent(absent) == USER_AGENT


# ── the passthrough half ────────────────────────────────────────────────────

# Strings that name a product. Hermes sends the first of these and it returns
# 200, so replacing it would discard real attribution to fix nothing.
NAMED = [
    "OpenAI/Python 2.24.0",
    "curl/8.19.0",
    "Wget/1.21.4",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "llmproxy/1.0.0",
    "MyAgent/3.1 (+https://example.com)",
]


@pytest.mark.parametrize("ua", NAMED)
def test_a_client_that_names_itself_is_passed_through_untouched(S, ua):
    """GUARD, and the half that keeps this from being a blunt instrument.
    curl and wget are deliberately here: both pass Cloudflare's check, and
    rewriting them would mislead anyone reproducing a problem by hand."""
    assert S._outbound_user_agent(ua) == ua


def test_the_rule_matches_a_prefix_not_a_substring(S):
    """GUARD: a product whose name merely contains 'java' or 'requests' is a
    real client and must survive."""
    for ua in ("JavaScriptRuntime/2.0", "SuperRequests/9", "MyPython-Requests-Tool/1"):
        assert S._outbound_user_agent(ua) == ua


# ── the three modes ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", [True, "true", "TRUE"])
def test_forward_true_reproduces_the_behaviour_that_shipped_before(tmp_path, monkeypatch, mode):
    """The escape hatch has to be an exact escape: relay what arrived, and
    return None when nothing did so the library default stands, byte for byte
    what llmproxy did before this setting existed."""
    S = _server(tmp_path, monkeypatch, forward_user_agent=mode)
    assert S._outbound_user_agent("Python-urllib/3.11") == "Python-urllib/3.11"
    assert S._outbound_user_agent("OpenAI/Python 2.24.0") == "OpenAI/Python 2.24.0"
    assert S._outbound_user_agent(None) is None


@pytest.mark.parametrize("mode", [False, "false", "False"])
def test_forward_false_always_sends_our_identity(tmp_path, monkeypatch, mode):
    S = _server(tmp_path, monkeypatch, forward_user_agent=mode)
    assert S._outbound_user_agent("OpenAI/Python 2.24.0") == USER_AGENT
    assert S._outbound_user_agent(None) == USER_AGENT


def test_an_unrecognised_mode_reads_as_auto(tmp_path, monkeypatch):
    """GUARD: a typo must not silently restore the behaviour this exists to
    fix, so anything unrecognised falls back to the safe default."""
    S = _server(tmp_path, monkeypatch, forward_user_agent="yes-please")
    assert S._outbound_user_agent("Python-urllib/3.11") == USER_AGENT
    assert S._outbound_user_agent("OpenAI/Python 2.24.0") == "OpenAI/Python 2.24.0"


def test_a_custom_user_agent_string_is_honoured(tmp_path, monkeypatch):
    S = _server(tmp_path, monkeypatch, user_agent="acme-gateway/4.2")
    assert S._outbound_user_agent("Python-urllib/3.11") == "acme-gateway/4.2"
    assert S._outbound_user_agent(None) == "acme-gateway/4.2"


def test_a_blank_custom_string_falls_back_rather_than_sending_nothing(tmp_path, monkeypatch):
    """GUARD: `"user_agent": ""` must not mean 'send an empty header'."""
    S = _server(tmp_path, monkeypatch, user_agent="   ")
    assert S._outbound_user_agent(None) == USER_AGENT


# ── it reaches the actual outbound headers ──────────────────────────────────

def test_the_resolved_agent_lands_in_the_forwarded_header_set(S):
    """Resolved in the single producer feeding all four outbound builder sites
    (buffered, streaming, cycling, fusion), so they cannot drift apart."""
    with S.app.test_request_context("/v1/chat/completions",
                                    headers={"User-Agent": "Python-urllib/3.11"}):
        assert S._forwarded_client_headers()["User-Agent"] == USER_AGENT
    with S.app.test_request_context("/v1/chat/completions",
                                    headers={"User-Agent": "OpenAI/Python 2.24.0"}):
        assert S._forwarded_client_headers()["User-Agent"] == "OpenAI/Python 2.24.0"


def test_off_the_request_thread_we_still_identify_ourselves(S):
    """The fusion-worker case. This used to return {}, so those requests went
    out as whatever the HTTP library called itself."""
    assert S._forwarded_client_headers() == {"User-Agent": USER_AGENT}


def test_other_forwarded_headers_are_still_relayed_verbatim(S):
    """GUARD: only the User-Agent is resolved. OpenRouter attribution headers
    are the reason this forwarding exists at all."""
    with S.app.test_request_context(
        "/v1/chat/completions",
        headers={"User-Agent": "python-requests/2.33.1",
                 "HTTP-Referer": "https://example.com", "X-Title": "My App"},
    ):
        out = S._forwarded_client_headers()
    assert out["HTTP-Referer"] == "https://example.com"
    assert out["X-Title"] == "My App"
    assert out["User-Agent"] == USER_AGENT


# ── the non-OpenAI dialects ─────────────────────────────────────────────────

def _provider():
    return {"base_url": "http://up.example/v1", "api_key": "k"}


def _payload():
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize("module,cls", [
    ("llmproxy.dialects.anthropic", "AnthropicOutbound"),
    ("llmproxy.dialects.gemini", "GeminiOutbound"),
])
def test_the_native_dialects_now_identify_themselves(module, cls):
    """Both ignored forwarded headers entirely, so those upstreams only ever
    saw the HTTP library's default."""
    mod = importlib.import_module(module)
    _url, headers, _body = getattr(mod, cls)().build_request(
        "chat/completions", "http://up.example/v1", _provider(), _payload(),
        stream=False, forwarded_headers={"User-Agent": USER_AGENT},
    )
    assert headers["User-Agent"] == USER_AGENT


@pytest.mark.parametrize("module,cls", [
    ("llmproxy.dialects.anthropic", "AnthropicOutbound"),
    ("llmproxy.dialects.gemini", "GeminiOutbound"),
])
def test_the_native_dialects_still_ignore_every_other_forwarded_header(module, cls):
    """GUARD: they drop OpenAI-gateway attribution headers on purpose. Taking
    the User-Agent must not turn into taking the lot."""
    mod = importlib.import_module(module)
    _url, headers, _body = getattr(mod, cls)().build_request(
        "chat/completions", "http://up.example/v1", _provider(), _payload(),
        stream=False,
        forwarded_headers={"User-Agent": USER_AGENT,
                           "HTTP-Referer": "https://example.com",
                           "X-Title": "My App"},
    )
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers


@pytest.mark.parametrize("module,cls", [
    ("llmproxy.dialects.anthropic", "AnthropicOutbound"),
    ("llmproxy.dialects.gemini", "GeminiOutbound"),
])
def test_the_native_dialects_tolerate_an_empty_forwarded_set(module, cls):
    """GUARD: they are called with {} from several paths and must not raise."""
    mod = importlib.import_module(module)
    _url, headers, _body = getattr(mod, cls)().build_request(
        "chat/completions", "http://up.example/v1", _provider(), _payload(),
        stream=False, forwarded_headers={},
    )
    assert "User-Agent" not in headers


# ── the route-cache builder ─────────────────────────────────────────────────

def test_the_model_listing_fetch_identifies_itself(S, monkeypatch):
    """The worst of the unidentified paths: it builds the route cache, so a CDN
    refusing it makes a provider vanish from every pool at once — which reads
    as 'that provider has no models' rather than as a block."""
    seen = {}

    def _fake_get(url, headers=None, timeout=None, **kw):
        seen["headers"] = headers or {}

        class _R:
            status_code = 200
            def json(self):
                return {"data": []}
            def raise_for_status(self):
                pass
        return _R()

    monkeypatch.setattr(S.requests, "get", _fake_get)
    S._fetch_provider_models("p1", {"base_url": "http://p1.example/v1",
                                    "api_key": "k"}, 5)
    assert seen["headers"].get("User-Agent") == USER_AGENT
