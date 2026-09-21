"""What an exhausted pool returns, and what it records while getting there.

Failover always worked: every `status_code >= 400` falls through to the next
candidate. What broke the client was the *final* status. `_proxy_cycling_*`
replayed the last candidate's status verbatim, so a pool whose last member
answered `404 No endpoints found that support tool use` handed the caller a
404 — which an OpenAI-compatible client reads as "no such model", classifies as
non-retryable, and gives up on with its retry budget untouched.

The companion half is the record: health scores store a bare boolean, so
nothing could say WHICH models failed and why. `GET /v1/failures` answers that
without needing the request log turned on.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from flask import Response


def _load_server(monkeypatch, config_path: Path):
    """Load the server against an isolated config with startup tasks neutered.

    Importing the module is not enough: hitting any route through the test
    client fires the startup hooks, which reach the network. Tests must never
    do that.
    """
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def S(tmp_path, monkeypatch):
    cfg = {
        "providers": {
            "p1": {"base_url": "http://p1.example/v1", "api_key": "k", "model_filter": None},
        },
        "sync_believed_free_on_startup": False,
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5,
                   "models_cache_ttl": 0, "response_cache_ttl": 0},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    server = _load_server(monkeypatch, path)
    server._reset_failures()
    server._capability_gap_registry.clear()
    return server


def _json_resp(body: dict, status: int) -> Response:
    return Response(json.dumps(body), status=status, content_type="application/json")


def _cycle(S, monkeypatch, per_candidate, candidates):
    """Run the non-streaming loop with each candidate's response scripted."""
    replies = dict(per_candidate)
    monkeypatch.setattr(
        S, "_proxy_request",
        lambda endpoint, pn, cfg, payload, timeout: replies[(pn, payload["model"])],
    )
    return S._proxy_cycling_non_streaming(
        "chat/completions", "flagship__free",
        [(pn, {}, um) for pn, um in candidates], {}, 5,
        route_reason="flagship_scored", virtual_model="llmproxy__flagship/free",
    )


# ── the status an exhausted pool reports ────────────────────────────────────

@pytest.mark.parametrize("statuses, expected", [
    ([404, 404], 502),              # the shipped bug: unanimous but misleading
    ([504, 504, 404], 502),         # Bill's real pool: two timeouts then the 404
    ([401], 502),                   # llmproxy's credentials, not the caller's
    ([403], 502),                   # likewise
    ([429, 503], 502),              # a mixture speaks for no one
    ([400, 400], 400),              # unanimous client fault is honest
    ([413, 413], 413),
    ([422], 422),
    ([429, 429], 429),              # accurate, and it clears by itself
    ([503, 503], 503),              # already says "server side"
    ([502], 502),
])
def test_the_exhausted_pool_status_is_only_sanitised_when_it_misleads(S, statuses, expected):
    attempted = [(f"p{i}", f"m{i}", st) for i, st in enumerate(statuses)]
    assert S._exhausted_pool_status(attempted) == expected


def test_a_relayed_404_becomes_502_so_the_client_still_retries(S, monkeypatch):
    """The exact failure Hermes hit, end to end through the cycling loop."""
    body = {"error": {"message": "No endpoints found that support tool use."}}
    resp = _cycle(S, monkeypatch,
                  {("openrouter", "a:free"): _json_resp(body, 404),
                   ("openrouter", "b:free"): _json_resp(body, 404)},
                  [("openrouter", "a:free"), ("openrouter", "b:free")])
    assert resp.status_code == 502


def test_the_error_names_every_candidate_that_was_tried(S, monkeypatch):
    """"All candidates failed" with no roll-call is what sends you to the logs."""
    resp = _cycle(S, monkeypatch,
                  {("p1", "m1"): _json_resp({"error": {"message": "nope"}}, 404),
                   ("p2", "m2"): _json_resp({"error": {"message": "gone"}}, 410)},
                  [("p1", "m1"), ("p2", "m2")])
    payload = json.loads(resp.get_data())
    listed = payload["error"]["llmproxy_candidates"]
    assert [c["target"] for c in listed] == ["p1/m1", "p2/m2"]
    assert [c["status"] for c in listed] == [404, 410]
    assert payload["error"]["code"] == "all_candidates_failed"


def test_a_unanimous_client_fault_is_relayed_untouched(S, monkeypatch):
    """If every candidate rejects it identically, the request really is at fault."""
    resp = _cycle(S, monkeypatch,
                  {("p1", "m1"): _json_resp({"error": {"message": "bad request"}}, 400),
                   ("p2", "m2"): _json_resp({"error": {"message": "bad request"}}, 400)},
                  [("p1", "m1"), ("p2", "m2")])
    assert resp.status_code == 400
    assert b"bad request" in resp.get_data()


def test_an_empty_pool_still_reports_503(S, monkeypatch):
    """503 keeps meaning "nothing to try", so it stays distinct from 502.

    Inside an app context because the empty-pool reply is built with jsonify,
    and in production this path only ever runs while serving a request.
    """
    with S.app.app_context():
        resp = S._proxy_cycling_non_streaming(
            "chat/completions", "flagship__free", [], {}, 5)
    assert resp.status_code == 503


# ── the failure record ──────────────────────────────────────────────────────

def test_each_failed_candidate_is_recorded(S, monkeypatch):
    _cycle(S, monkeypatch,
           {("p1", "m1"): _json_resp({"error": {"message": "boom"}}, 500),
            ("p2", "m2"): _json_resp({"error": {"message": "nope"}}, 404)},
           [("p1", "m1"), ("p2", "m2")])
    rows = S._failure_records()
    assert {r["target"] for r in rows} == {"p1/m1", "p2/m2"}
    assert rows[0]["virtual_model"] == "llmproxy__flagship/free"


def test_a_capability_rejection_is_classified_and_learned(S, monkeypatch):
    """One 404 should both explain itself and stop the model being picked again."""
    body = {"error": {"message": "No endpoints found that support tool use.",
                      "metadata": {"failed_routing_step": "Filter by Tool Compatibility"}}}
    _cycle(S, monkeypatch, {("openrouter", "z-ai/glm-9.9:free"): _json_resp(body, 404)},
           [("openrouter", "z-ai/glm-9.9:free")])
    row = S._failure_records()[0]
    assert row["kind"] == "capability"
    assert S._learned_capability_gaps("openrouter", "z-ai/glm-9.9:free") == {"tools"}


@pytest.mark.parametrize("status, body, kind", [
    (413, {"error": {"message": "too large"}}, "oversize"),
    (429, {"error": {"message": "rate limited"}}, "quota"),
    (500, {"error": {"message": "boom"}}, "server"),
    (418, {"error": {"message": "odd"}}, "upstream"),
])
def test_failures_are_classified_by_cause(S, status, body, kind):
    assert S._classify_failure(status, json.dumps(body).encode()) == kind


def test_a_timeout_outranks_every_other_classification(S):
    assert S._classify_failure(504, b"{}", timed_out=True) == "timeout"


# ── secrets never reach the report ──────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    b'{"error":{"message":"bad key sk-abcdef0123456789abcdef"}}',
    b'{"error":{"message":"Authorization: Bearer zzzsecretvalue123"}}',
    b'{"error":{"message":"api_key=AKIAIOSFODNN7EXAMPLEKEYMATERIAL"}}',
])
def test_credential_shaped_text_is_scrubbed_from_the_detail(S, raw):
    detail = S._failure_detail(raw)
    assert "[redacted]" in detail
    for leak in ("sk-abcdef", "zzzsecretvalue", "AKIAIOSFODNN7"):
        assert leak not in detail


def test_an_ordinary_message_survives_scrubbing_intact(S):
    """Over-redaction would make the report useless, which is its own failure."""
    detail = S._failure_detail(
        b'{"error":{"message":"model accounts/fireworks/models/llama-v3p1-70b-instruct '
        b'is not available"}}')
    assert detail == "model accounts/fireworks/models/llama-v3p1-70b-instruct is not available"


def test_the_detail_is_truncated(S):
    detail = S._failure_detail(b'{"error":{"message":"' + b"a " * 500 + b'"}}')
    assert len(detail) <= S._FAILURE_DETAIL_MAX_CHARS + 1


def test_a_non_json_body_is_still_usable(S):
    assert "gateway" in S._failure_detail(b"<!doctype html><p>gateway down</p>")


# ── the endpoint ────────────────────────────────────────────────────────────

def _get(S, path):
    with S.app.test_client() as c:
        return c.get(path)


def test_the_report_aggregates_repeat_failures_under_one_target(S):
    for _ in range(3):
        S._record_failure("openrouter", "m1", status=404, kind="capability", detail=b"x")
    S._record_failure("otherprov", "m2", status=504, kind="timeout", detail=b"y")
    data = _get(S, "/v1/failures").get_json()
    assert data["total"] == 4
    assert data["distinct_targets"] == 2
    top = data["by_model"][0]
    assert top["target"] == "openrouter/m1"
    assert top["failures"] == 3
    assert top["kinds"] == {"capability": 3}
    assert top["statuses"] == {"404": 3}


def test_the_recent_list_is_newest_first_and_honours_limit(S):
    for i in range(5):
        S._record_failure("p", f"m{i}", status=500, kind="server")
    data = _get(S, "/v1/failures?limit=2").get_json()
    assert [r["model"] for r in data["recent"]] == ["m4", "m3"]
    assert data["total"] == 5


@pytest.mark.parametrize("since", ["1h", "30m", "3600", "nonsense"])
def test_the_since_parameter_is_accepted_and_never_errors(S, since):
    """A diagnostic endpoint must not 400 on a typo."""
    S._record_failure("p", "m", status=500, kind="server")
    assert _get(S, f"/v1/failures?since={since}").status_code == 200


def test_the_ring_buffer_respects_its_cap(S):
    for i in range(S._FAILURE_LOG_MAX + 25):
        S._record_failure("p", f"m{i}", status=500, kind="server")
    rows = S._failure_records()
    assert len(rows) == S._FAILURE_LOG_MAX
    assert rows[0]["model"] == f"m{S._FAILURE_LOG_MAX + 24}"


def test_reset_is_gated_by_admin_auth(S, monkeypatch):
    S._record_failure("p", "m", status=500, kind="server")
    monkeypatch.setattr("llmproxy.admin.enforce_admin_auth", lambda: ({"error": "nope"}, 401))
    with S.app.test_client() as c:
        assert c.post("/v1/failures/reset").status_code == 401
    assert S._failure_records()

    monkeypatch.setattr("llmproxy.admin.enforce_admin_auth", lambda: None)
    with S.app.test_client() as c:
        assert c.post("/v1/failures/reset").status_code == 200
    assert S._failure_records() == []


# ── a CDN block is not an API error ─────────────────────────────────────────
#
# An upstream behind a CDN answers a refused request with an HTML interstitial,
# not with an API error. Recorded raw, that lands in the ring as "Backend
# request failed with status 403" plus four kilobytes of markup, and the one
# fact that would explain it — that the CDN, not the API, said no — is the fact
# that gets lost. This session spent an afternoon rediscovering it by hand.

CLOUDFLARE_1010 = (
    '<!doctype html><html class="no-js" lang="en-US"><head>'
    '<title>Attention Required! | Cloudflare</title></head><body>'
    '<h1>Access denied</h1><p>Error code: 1010</p>'
    '<p>The owner of this website has banned your access based on your '
    "browser's signature.</p><p>Cloudflare Ray ID: 8f2b1c</p></body></html>"
)


def test_a_cloudflare_block_page_is_named_rather_than_quoted(S):
    """The regression: the detail said 'status 403' and then raw markup."""
    detail = S._failure_detail(CLOUDFLARE_1010)
    assert detail.startswith("CDN blocked the request")
    assert "1010" in detail and "browser signature" in detail
    assert "<html" not in detail and "doctype" not in detail.lower()


def test_an_unnamed_cdn_code_still_reports_the_block(S):
    """A code we have no gloss for is still a CDN block, and saying so beats
    handing back HTML."""
    page = CLOUDFLARE_1010.replace("Error code: 1010", "Error code: 1104")
    detail = S._failure_detail(page)
    assert detail.startswith("CDN blocked the request")
    assert "1104" in detail


def test_a_block_page_with_no_code_is_still_recognised(S):
    page = ('<!doctype html><html><head><title>Attention Required! | Cloudflare'
            '</title></head><body>Sorry, you have been blocked</body></html>')
    assert S._failure_detail(page).startswith("CDN blocked the request")


def test_an_ordinary_api_error_is_untouched(S):
    """GUARD: the existing behaviour is the common case and must not change."""
    body = json.dumps({"error": {"message": "model not found"}})
    assert S._failure_detail(body) == "model not found"


def test_a_model_named_after_a_cdn_is_not_mistaken_for_one(S):
    """GUARD, and the reason two independent markers are required. An upstream
    serving 'cloudflare/llama-3' would otherwise have every one of its ordinary
    JSON errors relabelled — worse than the bare status this replaces."""
    body = json.dumps({"error": {"message": "cloudflare/llama-3 is unavailable"}})
    assert S._failure_detail(body) == "cloudflare/llama-3 is unavailable"


def test_an_html_page_that_is_not_a_cdn_block_is_not_relabelled(S):
    """GUARD: an upstream's own HTML error page is not a CDN refusal."""
    page = "<!doctype html><html><body><h1>502 Bad Gateway</h1>nginx</body></html>"
    assert not S._failure_detail(page).startswith("CDN blocked")


def test_a_cdn_block_gets_its_own_failure_kind(S):
    """So /v1/failures separates it in `kinds` and a CDN refusal stops being
    indistinguishable from an API error at the same status code."""
    S._reset_failures()
    S._record_failure("p1", "m1", status=403, detail=CLOUDFLARE_1010)
    rows = S._failure_records()
    assert rows[0]["kind"] == "cdn_block"
    assert rows[0]["detail"].startswith("CDN blocked the request")


def test_an_ordinary_upstream_failure_keeps_its_kind(S):
    """GUARD on the other branch of the same condition."""
    S._reset_failures()
    S._record_failure("p1", "m1", status=400,
                      detail=json.dumps({"error": {"message": "bad request"}}))
    assert S._failure_records()[0]["kind"] == "upstream"


def test_a_caller_supplied_kind_is_not_overwritten(S):
    """GUARD: a caller that already named the kind knows more than the body
    sniffer does, so only an unclassified 'upstream' failure is upgraded."""
    S._reset_failures()
    S._record_failure("p1", "m1", status=403, kind="timeout",
                      detail=CLOUDFLARE_1010)
    assert S._failure_records()[0]["kind"] == "timeout"


def test_secrets_in_a_cdn_body_are_still_scrubbed(S):
    """GUARD: the new early return must not skip the redaction the old path
    applied. A block page can echo the request, key and all."""
    page = CLOUDFLARE_1010.replace(
        "Access denied", "Access denied for Bearer sk-live-abcdef1234567890")
    assert "sk-live-abcdef1234567890" not in S._failure_detail(page)
