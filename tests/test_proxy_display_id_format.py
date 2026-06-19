"""Pin the display-id format returned by /v1/models.

Regression guards for:
- the original "model (provider)" form (rejected by Hermes for the spaces);
- the PR #27 "model__provider" form (correct chars, but provider on the wrong side);
- the current "provider__model" form (matches the canonical slash order).
All three legacy input forms must continue to resolve as input on chat/completions.
"""

from __future__ import annotations

import importlib
import re

import pytest

_DISPLAY_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+__[^\s()]+$")


@pytest.fixture
def server(monkeypatch, minimal_config):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(minimal_config))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    # Suppress background daemon threads that can race with cache-sensitive tests:
    # the startup warmup (step 5 of _run_startup_tasks_once) and the per-request
    # interval probe check both write to _models_list_cache, and a stale thread
    # from a prior test can overwrite the cache after the test sets it to None.
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


def _stub_response(model_ids):
    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"data": [{"id": mid, "object": "model"} for mid in model_ids]}
    return _R()


def test_proxy_id_uses_provider_first_double_underscore(server, monkeypatch):
    """Every non-virtual model id must use the `provider__model` form —
    provider on the left, no spaces, no parens, no leading slash, and at most
    one `/` (multi-slash upstream ids are flattened)."""
    captured = _stub_response([
        "qwen2.5vl:3b",
        "llama3.2-3b-instruct",
        "nested/path/model-x",
    ])
    monkeypatch.setattr(server.requests, "get", lambda *a, **kw: captured)

    models = server._fetch_provider_models(
        "ollama",
        {"base_url": "http://upstream.example/v1", "api_key": "x"},
        timeout=1,
    )

    assert models, "expected the stubbed upstream to yield models"
    by_upstream = {m["_upstream_id"]: m["id"] for m in models}
    for m in models:
        mid = m["id"]
        assert " " not in mid, f"display id contains a space: {mid!r}"
        assert "(" not in mid and ")" not in mid, f"display id contains parens: {mid!r}"
        assert _DISPLAY_ID_RE.match(mid), (
            f"display id {mid!r} does not match expected `provider__model` shape"
        )
        assert mid.startswith("ollama__"), f"provider must come first: {mid!r}"
        assert mid.count("/") <= 1, f"display id must carry at most one slash: {mid!r}"
    # 0-slash and 1-slash upstreams are unchanged; the 2-slash one is flattened.
    assert by_upstream["qwen2.5vl:3b"] == "ollama__qwen2.5vl:3b"
    assert by_upstream["nested/path/model-x"] == "ollama__nested_path/model-x"
    # The route still forwards under the original (un-flattened) upstream id.
    nested = next(m for m in models if m["_upstream_id"] == "nested/path/model-x")
    assert nested["_route"] == ("ollama", "nested/path/model-x")


def test_flatten_display_model(server):
    """_flatten_display_model collapses all but the last '/' into '_'."""
    f = server._flatten_display_model
    assert f("gpt-4o") == "gpt-4o"                       # 0 slashes
    assert f("anthropic/claude-3.5") == "anthropic/claude-3.5"   # 1 slash unchanged
    assert f("meta-llama/llama-3/instruct") == "meta-llama_llama-3/instruct"  # 2
    assert f("a/b/c/d") == "a_b_c/d"                     # 3 slashes


def test_resolver_resolves_flattened_multislash_via_cache(server):
    """A flattened display id resolves to the original upstream via the cache."""
    with server._model_route_cache_lock:
        server._model_route_cache.clear()
        server._model_route_cache["fakeprov__meta-llama_llama-3/instruct"] = (
            "fakeprov", "meta-llama/llama-3/instruct",
        )
    provider_name, _cfg, upstream_model, err = server._resolve_provider(
        "fakeprov__meta-llama_llama-3/instruct"
    )
    assert err is None, f"unexpected error: {err}"
    assert provider_name == "fakeprov"
    assert upstream_model == "meta-llama/llama-3/instruct"
    with server._model_route_cache_lock:
        server._model_route_cache.clear()


def test_resolver_rebuilds_cache_for_cold_flattened_id(server, monkeypatch):
    """On a cold cache miss for a flattened multi-slash id whose left token is a
    configured provider, _resolve_provider rebuilds the route cache once and
    retries so routing recovers the true upstream id."""
    with server._model_route_cache_lock:
        server._model_route_cache.clear()

    calls = {"n": 0}

    def _fake_rebuild(providers_cfg, timeout, only_if_empty=False):
        calls["n"] += 1
        with server._model_route_cache_lock:
            server._model_route_cache["fakeprov__meta-llama_llama-3/instruct"] = (
                "fakeprov", "meta-llama/llama-3/instruct",
            )
        return []

    monkeypatch.setattr(server, "_rebuild_route_cache", _fake_rebuild)

    provider_name, _cfg, upstream_model, err = server._resolve_provider(
        "fakeprov__meta-llama_llama-3/instruct"
    )
    assert calls["n"] == 1, "expected exactly one route-cache rebuild"
    assert err is None, f"unexpected error: {err}"
    assert provider_name == "fakeprov"
    assert upstream_model == "meta-llama/llama-3/instruct"
    with server._model_route_cache_lock:
        server._model_route_cache.clear()


def test_resolver_accepts_new_format(server, monkeypatch):
    """_resolve_provider must parse the current `provider__model` form."""
    with server._model_route_cache_lock:
        server._model_route_cache.clear()

    provider_name, _provider_cfg, upstream_model, err = server._resolve_provider(
        "fakeprov__qwen2.5vl:3b"
    )
    assert err is None, f"unexpected error: {err}"
    assert provider_name == "fakeprov"
    assert upstream_model == "qwen2.5vl:3b"


def test_resolver_still_accepts_pr27_format(server, monkeypatch):
    """Legacy `model__provider` ids from PR #27 must still resolve."""
    with server._model_route_cache_lock:
        server._model_route_cache.clear()

    provider_name, _provider_cfg, upstream_model, err = server._resolve_provider(
        "qwen2.5vl:3b__fakeprov"
    )
    assert err is None, f"unexpected error: {err}"
    assert provider_name == "fakeprov"
    assert upstream_model == "qwen2.5vl:3b"


def test_spaces_in_upstream_model_name_are_sanitized(server, monkeypatch):
    """If an upstream id contains spaces, they must be replaced with `_` so the
    display id stays whitespace-free."""
    captured = _stub_response(["My Cool Model v1"])
    monkeypatch.setattr(server.requests, "get", lambda *a, **kw: captured)

    models = server._fetch_provider_models(
        "ollama",
        {"base_url": "http://upstream.example/v1", "api_key": "x"},
        timeout=1,
    )
    assert models
    mid = models[0]["id"]
    assert " " not in mid
    assert mid == "ollama__My_Cool_Model_v1"
    # The original upstream id is preserved on the route so requests still
    # forward to the upstream under its true name.
    assert models[0]["_upstream_id"] == "My Cool Model v1"
    assert models[0]["_route"] == ("ollama", "My Cool Model v1")


def test_spaces_in_provider_name_are_sanitized(server, monkeypatch):
    """If a provider name contains spaces, the display id replaces them with `_`."""
    captured = _stub_response(["llama3"])
    monkeypatch.setattr(server.requests, "get", lambda *a, **kw: captured)

    models = server._fetch_provider_models(
        "my provider",
        {"base_url": "http://upstream.example/v1", "api_key": "x"},
        timeout=1,
    )
    assert models
    mid = models[0]["id"]
    assert " " not in mid
    assert mid.startswith("my_provider__")


def test_resolver_still_accepts_legacy_paren_format(server, monkeypatch):
    """Pre-PR #27 `model (provider)` ids must still resolve (backward compat)."""
    with server._model_route_cache_lock:
        server._model_route_cache.clear()

    provider_name, _provider_cfg, upstream_model, err = server._resolve_provider(
        "qwen2.5vl:3b (fakeprov)"
    )
    assert err is None, f"unexpected error: {err}"
    assert provider_name == "fakeprov"
    assert upstream_model == "qwen2.5vl:3b"


def test_bare_list_models_payload_is_accepted(server, monkeypatch):
    """Some upstreams (e.g. Together) return /models as a bare JSON array
    instead of `{"data": [...]}`. These must be parsed, not dropped with a
    `'list' object has no attribute 'get'` error."""
    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return [{"id": "meta-llama/Llama-3-8b", "object": "model"}]
    monkeypatch.setattr(server.requests, "get", lambda *a, **kw: _R())

    models = server._fetch_provider_models(
        "together",
        {"base_url": "http://upstream.example/v1", "api_key": "x"},
        timeout=1,
    )
    assert models, "bare-list /models payload should yield models"
    assert models[0]["_upstream_id"] == "meta-llama/Llama-3-8b"
    assert models[0]["id"] == "together__meta-llama/Llama-3-8b"


def test_fetch_failure_logs_status_and_body_without_secrets(server, monkeypatch, caplog):
    """A failed /models fetch should log the HTTP status, content-type, and a
    body snippet for diagnosis — but never the Authorization header / api_key."""
    import requests as _requests

    class _R:
        status_code = 404
        headers = {"Content-Type": "text/html"}
        text = "<html><body>Not Found</body></html>"
        def raise_for_status(self):
            raise _requests.HTTPError("404 Client Error: Not Found", response=self)
        def json(self): return {}
    monkeypatch.setattr(server.requests, "get", lambda *a, **kw: _R())

    with caplog.at_level("WARNING"):
        models = server._fetch_provider_models(
            "github",
            {"base_url": "http://upstream.example/inference", "api_key": "super-secret-token"},
            timeout=1,
        )
    assert models == []
    text = caplog.text
    assert "status=404" in text
    assert "content_type=text/html" in text
    assert "Not Found" in text
    assert "url=http://upstream.example/inference/models" in text
    # The bearer token must never leak into the logs.
    assert "super-secret-token" not in text
    assert "Authorization" not in text


def test_models_url_override_is_used_for_fetch(server, monkeypatch):
    """When a provider sets models_url, that URL is fetched instead of
    <base_url>/models (e.g. GitHub serves its catalog at a different path)."""
    requested = {}

    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return [{"id": "openai/gpt-4.1", "object": "model"}]
    def _fake_get(url, *a, **kw):
        requested["url"] = url
        return _R()
    monkeypatch.setattr(server.requests, "get", _fake_get)

    models = server._fetch_provider_models(
        "github",
        {
            "base_url": "https://models.github.ai/inference",
            "api_key": "x",
            "models_url": "https://models.github.ai/catalog/models",
        },
        timeout=1,
    )
    assert requested["url"] == "https://models.github.ai/catalog/models"
    assert models and models[0]["_upstream_id"] == "openai/gpt-4.1"
    assert models[0]["id"] == "github__openai/gpt-4.1"


def test_models_id_field_and_keep_task_adapt_cloudflare_shape(server, monkeypatch):
    """Cloudflare's models/search puts the usable id in 'name' (not 'id') and
    mixes tasks. models_id_field + models_keep_task must read 'name' and drop
    non-Text-Generation entries."""
    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"success": True, "result": [
                {"id": "uuid-1", "name": "@cf/meta/llama-3.1-8b-instruct",
                 "task": {"name": "Text Generation"}},
                {"id": "uuid-2", "name": "@cf/baai/bge-base-en-v1.5",
                 "task": {"name": "Text Embeddings"}},
                {"id": "uuid-3", "name": "@cf/black-forest-labs/flux-1-schnell",
                 "task": {"name": "Text-to-Image"}},
            ]}
    monkeypatch.setattr(server.requests, "get", lambda *a, **kw: _R())

    models = server._fetch_provider_models(
        "cloudflare-workers",
        {
            "base_url": "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1",
            "api_key": "x",
            "models_url": "https://api.cloudflare.com/client/v4/accounts/acct/ai/models/search",
            "models_id_field": "name",
            "models_keep_task": "Text Generation",
        },
        timeout=1,
    )
    # Only the Text Generation model survives; its upstream id is the 'name'.
    upstream = [m["_upstream_id"] for m in models]
    assert upstream == ["@cf/meta/llama-3.1-8b-instruct"]
    # Routing keeps the true upstream id; the display id flattens extra slashes.
    assert models[0]["_route"] == ("cloudflare-workers", "@cf/meta/llama-3.1-8b-instruct")
    assert models[0]["id"] == "cloudflare-workers__@cf_meta/llama-3.1-8b-instruct"


def test_canonical_virtual_ids_use_double_underscore(server):
    """Every virtual id in _NEW_VIRTUAL_MODELS (the canonical internal set)
    must start with `llmproxy__`.  The advertised form uses `llmproxy/` but
    internal routing state stays in canonical form."""
    new = server._NEW_VIRTUAL_MODELS
    assert new, "expected _NEW_VIRTUAL_MODELS to be non-empty"
    for vid in new:
        assert vid.startswith("llmproxy__"), f"virtual id should use __ prefix: {vid!r}"
        assert " " not in vid and "(" not in vid and ")" not in vid


def test_legacy_virtual_ids_still_in_membership_set(server):
    """The legacy `llmproxy/...` virtual ids must still resolve as input —
    they remain in _VIRTUAL_MODELS so chat/completions dispatches them to the
    virtual handler instead of treating them as provider/model pairs."""
    legacy = server._LEGACY_VIRTUAL_MODELS
    assert "llmproxy/free" in legacy
    assert "llmproxy/deep/free" in legacy
    # All legacy ids must be in the combined membership set.
    for vid in legacy:
        assert vid in server._VIRTUAL_MODELS


def test_virtual_candidates_dispatch_matches_for_legacy_and_new(server):
    """`_get_virtual_candidates("llmproxy/free")` and
    `_get_virtual_candidates("llmproxy__free")` must produce the same list."""
    new = server._get_virtual_candidates("llmproxy__free")
    legacy = server._get_virtual_candidates("llmproxy/free")
    assert new == legacy

    new_tiered = server._get_virtual_candidates("llmproxy__deep/free")
    legacy_tiered = server._get_virtual_candidates("llmproxy/deep/free")
    assert new_tiered == legacy_tiered


# ── slash-form id round-trip (_canonicalize_model_id) ─────────────────────────
# Some clients display and send the first `__` of an advertised id rewritten to
# `/` (the `provider/model` slash form) on each request. _canonicalize_model_id
# must invert that (first `/` -> `__`) so the existing resolver/virtual machinery
# sees the canonical form — but only for ids whose leading token is a configured
# provider or the `llmproxy` virtual namespace.

def test_canonicalize_rewrites_first_slash_for_known_provider(server):
    config = server.load_config()
    f = server._canonicalize_model_id
    # 0-slash model part: provider/model -> provider__model
    assert f("fakeprov/qwen2.5vl:3b", config) == "fakeprov__qwen2.5vl:3b"
    # 1-slash model part: only the FIRST slash (the shim separator) is rewritten,
    # preserving the interior slash of the upstream id.
    assert f("fakeprov/meta-llama/llama-3.3", config) == "fakeprov__meta-llama/llama-3.3"


def test_canonicalize_rewrites_llmproxy_virtuals(server):
    config = server.load_config()
    f = server._canonicalize_model_id
    assert f("llmproxy/free", config) == "llmproxy__free"
    assert f("llmproxy/loadbalanced", config) == "llmproxy__loadbalanced"
    assert f("llmproxy/deep/free", config) == "llmproxy__deep/free"


def test_canonicalize_leaves_unknown_and_canonical_ids_untouched(server):
    config = server.load_config()
    f = server._canonicalize_model_id
    # Already-canonical (__) ids pass through unchanged.
    assert f("fakeprov__qwen2.5vl:3b", config) == "fakeprov__qwen2.5vl:3b"
    assert f("llmproxy__free", config) == "llmproxy__free"
    # Bare ids with no slash are unchanged.
    assert f("gpt-4", config) == "gpt-4"
    # A leading token that is neither a configured provider nor `llmproxy` is left
    # alone so ordinary slash ids still flow through parse_model_string.
    assert f("unknownprov/some-model", config) == "unknownprov/some-model"


def test_canonicalized_slash_id_resolves_to_same_route(server):
    """A real model requested in the pi slash form resolves to the same upstream
    as its canonical `__` spelling."""
    config = server.load_config()
    with server._model_route_cache_lock:
        server._model_route_cache.clear()
        server._model_route_cache["fakeprov__meta-llama/llama-3.3"] = (
            "fakeprov", "meta-llama/llama-3.3",
        )
    canonical = server._canonicalize_model_id("fakeprov/meta-llama/llama-3.3", config)
    provider_name, _cfg, upstream_model, err = server._resolve_provider(canonical)
    assert err is None, f"unexpected error: {err}"
    assert provider_name == "fakeprov"
    assert upstream_model == "meta-llama/llama-3.3"
    with server._model_route_cache_lock:
        server._model_route_cache.clear()


def test_canonicalized_virtual_slash_id_is_recognized_as_virtual(server):
    """`llmproxy/free` canonicalizes to `llmproxy__free` and is recognized as a
    virtual model so it dispatches to the cycling handler, not provider routing."""
    config = server.load_config()
    canonical = server._canonicalize_model_id("llmproxy/free", config)
    assert canonical == "llmproxy__free"
    assert server._is_virtual_model(canonical)


# ── slash display form helper (_display_id) ──────────────────────────────────
# _display_id converts a canonical id to the "provider/model" slash form. It is
# used both inbound (to dual-key the route cache so an advertised id round-trips
# via _canonicalize_model_id) and outbound: VIRTUAL model ids are advertised on
# /v1/models in this slash form (e.g. "llmproxy/deep__free") so clients like
# opencode show a distinct label per virtual. REAL model ids stay canonical
# ("provider__model") so opencode does not collapse every model from one provider
# under a single "provider/" group.

def test_display_id_swaps_separators(server):
    f = server._display_id
    assert f("openrouter__deepseek/deepseek-chat-v3") == "openrouter/deepseek__deepseek-chat-v3"
    assert f("llmproxy__exploratory/free") == "llmproxy/exploratory__free"
    assert f("llmproxy__free") == "llmproxy/free"            # no interior slash
    assert f("ollama__qwen2.5vl:3b") == "ollama/qwen2.5vl:3b"
    # already-foreign ids (no "__") are returned unchanged
    assert f("gpt-4") == "gpt-4"


def test_display_id_round_trips_through_canonicalize(server):
    """_canonicalize_model_id inverts _display_id back to the canonical form so
    an advertised id sent back by a client resolves to the same route. Verified
    on a cold cache (the string-level path), where the leading token is a provider."""
    config = server.load_config()
    with server._model_route_cache_lock:
        server._model_route_cache.clear()
    for canonical in (
        "fakeprov__qwen2.5vl:3b",
        "fakeprov__meta-llama/llama-3.3",
        "llmproxy__deep/free",
    ):
        advertised = server._display_id(canonical)
        assert server._canonicalize_model_id(advertised, config) == canonical


def test_models_list_advertises_canonical_form_no_leading_provider_slash(server, monkeypatch):
    """The /v1/models payload advertises the canonical "provider__model" form.
    No advertised id begins with "<provider>/" — the provider separator is "__",
    so clients that group by the segment before the first "/" (e.g. opencode) do
    not collapse every model under one provider group."""
    routes = {
        "openrouter__deepseek/deepseek-chat-v3:free": ("openrouter", "deepseek/deepseek-chat-v3:free"),
        "openrouter__qwen/qwen-2.5:free": ("openrouter", "qwen/qwen-2.5:free"),
    }

    def _fake_rebuild(providers_cfg, timeout, only_if_empty=False):
        with server._model_route_cache_lock:
            server._model_route_cache.clear()
            server._model_route_cache.update(routes)
            for cid, route in list(routes.items()):
                server._model_route_cache[server._display_id(cid)] = route
        return [{"id": k, "name": k, "object": "model"} for k in routes]

    monkeypatch.setattr(server, "_rebuild_route_cache", _fake_rebuild)
    monkeypatch.setattr(server, "_sync_local_provider_models_once", lambda: None)
    server._models_list_cache = None

    data = server.app.test_client().get("/v1/models").get_json()["data"]
    ids = [m["id"] for m in data]

    # canonical form: ids are advertised with the "__" provider separator, never
    # a leading "provider/" segment.
    assert "openrouter__deepseek/deepseek-chat-v3:free" in ids
    assert "openrouter__qwen/qwen-2.5:free" in ids
    for mid in ids:
        assert not mid.startswith("openrouter/"), \
            f"advertised id must not start with a provider/ segment: {mid!r}"

    with server._model_route_cache_lock:
        server._model_route_cache.clear()


def test_advertised_real_id_resolves_losslessly_via_dual_keyed_cache(server):
    """An inbound advertised id whose upstream legitimately contains "__" resolves
    to the exact upstream via the dual-keyed route cache (the hot path), with no
    string-level reverse ambiguity."""
    config = server.load_config()
    canonical = "fakeprov__weird__name"          # upstream literally contains "__"
    advertised = server._display_id(canonical)   # -> "fakeprov/weird__name"
    route = ("fakeprov", "weird__name")
    with server._model_route_cache_lock:
        server._model_route_cache.clear()
        server._model_route_cache[canonical] = route
        server._model_route_cache[advertised] = route
    # canonicalize returns the advertised id unchanged (it is a cache key), so the
    # resolver's cache lookup hits the exact upstream.
    assert server._canonicalize_model_id(advertised, config) == advertised
    pn, _cfg, upstream, err = server._resolve_provider(advertised)
    assert err is None, f"unexpected error: {err}"
    assert (pn, upstream) == ("fakeprov", "weird__name")
    with server._model_route_cache_lock:
        server._model_route_cache.clear()


# ── virtual model friendly names (_virtual_display_name) ─────────────────────
# Virtual models must have a `name` that contains no "/" so client pickers that
# derive a label by stripping to the last "/" (e.g. opencode) show the full
# label verbatim rather than a bare word like "free".

def test_virtual_display_name_simple(server):
    f = server._virtual_display_name
    assert f("llmproxy__free") == "[llmproxy] Free"
    assert f("llmproxy__local") == "[llmproxy] Local"
    assert f("llmproxy__loadbalanced") == "[llmproxy] Loadbalanced"
    assert f("llmproxy__tools") == "[llmproxy] Tools"
    assert f("llmproxy__fusion") == "[llmproxy] Fusion"


def test_virtual_display_name_compound(server):
    f = server._virtual_display_name
    assert f("llmproxy__exploratory/free") == "[llmproxy] Exploratory — Free"
    assert f("llmproxy__deep/free") == "[llmproxy] Deep — Free"
    assert f("llmproxy__standard/local") == "[llmproxy] Standard — Local"
    assert f("llmproxy__tools/free") == "[llmproxy] Tools — Free"
    assert f("llmproxy__openrouter/free") == "[llmproxy] Openrouter — Free"


def test_virtual_display_name_has_no_slash(server):
    """name must contain no "/" so strip-to-last-slash clients show it verbatim."""
    for vid in server._NEW_VIRTUAL_MODELS:
        name = server._virtual_display_name(vid)
        assert "/" not in name, f"virtual name contains slash: {vid!r} -> {name!r}"
        assert name.startswith("[llmproxy]"), f"virtual name missing prefix: {vid!r} -> {name!r}"


def test_models_list_virtual_names_are_friendly(server, monkeypatch):
    """/v1/models emits human-readable names for virtual models with no "/"."""
    # Seed the route cache with a believed_free model so llmproxy/free appears.
    seed_cid = "fakeprov__free-model"
    seed_route = ("fakeprov", "free-model")

    def _fake_rebuild(providers_cfg, timeout, only_if_empty=False):
        with server._model_route_cache_lock:
            server._model_route_cache.clear()
            server._model_route_cache[seed_cid] = seed_route
            server._model_route_cache[server._display_id(seed_cid)] = seed_route
        return [{"id": seed_cid, "name": seed_cid, "object": "model"}]

    monkeypatch.setattr(server, "_rebuild_route_cache", _fake_rebuild)
    monkeypatch.setattr(server, "_sync_local_provider_models_once", lambda: None)
    server._models_list_cache = None

    data = server.app.test_client().get("/v1/models").get_json()["data"]
    # Virtual ids are advertised in the "llmproxy/..." slash form.
    virtual = [m for m in data if m["id"].startswith("llmproxy/")]
    assert virtual, "expected virtual models in /v1/models listing"
    for m in virtual:
        assert "/" not in m["name"], \
            f"virtual model name contains slash: id={m['id']!r} name={m['name']!r}"
        assert m["name"].startswith("[llmproxy]"), \
            f"virtual model name missing [llmproxy] prefix: {m['name']!r}"

    with server._model_route_cache_lock:
        server._model_route_cache.clear()


def test_get_model_virtual_name_is_friendly(server):
    """GET /v1/models/<virtual_id> returns a friendly name with no "/"."""
    resp = server.app.test_client().get("/v1/models/llmproxy__free")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["name"] == "[llmproxy] Free"
    assert "/" not in body["name"]

    resp2 = server.app.test_client().get("/v1/models/llmproxy__deep/free")
    body2 = resp2.get_json()
    assert resp2.status_code == 200
    assert body2["name"] == "[llmproxy] Deep — Free"
    assert "/" not in body2["name"]


def test_real_model_name_unchanged(server, monkeypatch):
    """Real model names from upstream are passed through without modification."""
    routes = {"openrouter__aion-labs/aion-1.0": ("openrouter", "aion-labs/aion-1.0")}

    def _fake_rebuild(providers_cfg, timeout, only_if_empty=False):
        with server._model_route_cache_lock:
            server._model_route_cache.clear()
            server._model_route_cache.update(routes)
            for cid, route in list(routes.items()):
                server._model_route_cache[server._display_id(cid)] = route
        return [{"id": "openrouter__aion-labs/aion-1.0",
                 "name": "Aion Labs Aion 1.0", "object": "model"}]

    monkeypatch.setattr(server, "_rebuild_route_cache", _fake_rebuild)
    monkeypatch.setattr(server, "_sync_local_provider_models_once", lambda: None)
    server._models_list_cache = None

    data = server.app.test_client().get("/v1/models").get_json()["data"]
    real = next((m for m in data if m["id"] == "openrouter__aion-labs/aion-1.0"), None)
    assert real is not None, "expected the real model in the listing"
    assert real["name"] == "Aion Labs Aion 1.0", \
        f"real model name should be unchanged: {real['name']!r}"

    with server._model_route_cache_lock:
        server._model_route_cache.clear()


def test_models_list_advertises_virtual_ids_in_slash_form(server, monkeypatch):
    """Virtual model ids are advertised as "llmproxy/<suffix>" with any interior
    "/" encoded as "__" (e.g. "llmproxy/deep__free"), so opencode groups every
    virtual under one "llmproxy/" provider with a distinct label per entry. Real
    model ids stay canonical "provider__model" (no leading "provider/")."""
    seed_cid = "fakeprov__free-model"
    seed_route = ("fakeprov", "free-model")

    def _fake_rebuild(providers_cfg, timeout, only_if_empty=False):
        with server._model_route_cache_lock:
            server._model_route_cache.clear()
            server._model_route_cache[seed_cid] = seed_route
            server._model_route_cache[server._display_id(seed_cid)] = seed_route
        return [{"id": seed_cid, "name": seed_cid, "object": "model"}]

    monkeypatch.setattr(server, "_rebuild_route_cache", _fake_rebuild)
    monkeypatch.setattr(server, "_sync_local_provider_models_once", lambda: None)
    server._models_list_cache = None

    ids = [m["id"] for m in server.app.test_client().get("/v1/models").get_json()["data"]]
    # The general free virtual is advertised in slash form.
    assert "llmproxy/free" in ids
    # No virtual id retains the canonical "llmproxy__" prefix in the advertised list.
    assert not any(i.startswith("llmproxy__") for i in ids), \
        f"virtual ids must be advertised in slash form: {[i for i in ids if i.startswith('llmproxy')]!r}"
    # Any compound virtual carries exactly one "/" (right after "llmproxy").
    for i in ids:
        if i.startswith("llmproxy/"):
            assert i.count("/") == 1, f"virtual id must carry one slash: {i!r}"
    # The real model stays canonical (no leading "fakeprov/").
    assert seed_cid in ids

    with server._model_route_cache_lock:
        server._model_route_cache.clear()


def test_canonicalize_fuzzy_matches_underscores_for_slash(server):
    """Robustness: an inbound virtual id that uses "__" where the canonical form
    has "/" (e.g. a client that double-underscores everything) still resolves to
    the canonical virtual id."""
    config = server.load_config()
    f = server._canonicalize_model_id
    # "__" used in place of the "/" separator between dimension and tier.
    assert f("llmproxy__deep__free", config) == "llmproxy__deep/free"
    assert f("llmproxy__standard__local", config) == "llmproxy__standard/local"
    # The advertised slash form and the legacy all-slash form still work too.
    assert f("llmproxy/deep__free", config) == "llmproxy__deep/free"
    assert f("llmproxy/deep/free", config) == "llmproxy__deep/free"
    # A genuine canonical id is returned unchanged.
    assert f("llmproxy__deep/free", config) == "llmproxy__deep/free"
