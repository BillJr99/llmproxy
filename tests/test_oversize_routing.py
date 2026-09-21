"""Routing around a candidate that refused a request as too large (HTTP 413).

Cycling past a 413 already worked — it is not a transient status, so the loop
fails straight over. What did not exist was memory: the candidate kept its rank
and was tried first again on the next request. Under a random rotation that cost
one wasted call in N; under the deterministic flagship ranking it costs one on
every request.

A 413 is a property of the request's SIZE, not of the candidate's availability,
so the fix is a size watermark rather than a cooldown: remember the smallest
body each target has refused, and route around it only for requests at least
that large. These tests pin that distinction, and the invariant that a candidate
is demoted but never dropped.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _make_server(monkeypatch, tmp_path: Path):
    cfg = {
        "providers": {
            "p1": {"base_url": "http://p1.example/v1", "api_key": "k"},
            "p2": {"base_url": "http://p2.example/v1", "api_key": "k"},
        },
        "believed_free": [], "model_reasoning": {},
        "server": {"log_level": "ERROR", "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("LLMPROXY_CONFIG", str(p))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def server(monkeypatch, tmp_path: Path):
    s = _make_server(monkeypatch, tmp_path)
    s._reset_oversize()
    yield s
    s._reset_oversize()


CANDS = [
    ("p1", {"base_url": "http://p1.example/v1", "api_key": "k"}, "m1"),
    ("p2", {"base_url": "http://p2.example/v1", "api_key": "k"}, "m2"),
]


def _payload(n_chars: int) -> dict:
    return {"messages": [{"role": "user", "content": "x" * n_chars}]}


# ── the watermark ───────────────────────────────────────────────────────────

def test_a_watermark_is_recorded_and_only_ever_tightens(server):
    """The limit can only be bounded from above by what was actually refused, so
    a smaller rejection lowers it and a larger one must not raise it."""
    server._record_oversize("p1", "m1", 5000)
    assert server._is_oversize_for("p1", "m1", 5000) is True

    server._record_oversize("p1", "m1", 3000)       # tighter evidence
    assert server._is_oversize_for("p1", "m1", 3000) is True
    assert server._is_oversize_for("p1", "m1", 2999) is False

    server._record_oversize("p1", "m1", 9000)       # weaker — must not loosen
    assert server._is_oversize_for("p1", "m1", 3000) is True


def test_a_request_below_the_watermark_is_unaffected(server):
    server._record_oversize("p1", "m1", 5000)
    assert server._is_oversize_for("p1", "m1", 4999) is False
    assert server._is_oversize_for("p1", "m1", 5001) is True


def test_an_unknown_target_has_no_limit(server):
    assert server._is_oversize_for("p1", "m1", 10**9) is False


def test_the_watermark_is_shared_across_accounts_but_not_providers(server):
    """A body limit belongs to the endpoint, not the credential — so one
    account's 413 informs the rest, and a different provider learns nothing."""
    server._record_oversize("p1", "m1", 5000)
    assert server._oversize_key("p1", "m1") == server._usage_key("p1", "m1", None)
    assert server._is_oversize_for("p2", "m1", 5000) is False
    assert server._is_oversize_for("p1", "m2", 5000) is False


def test_size_is_measured_in_serialized_bytes_not_message_text(server):
    """Tool definitions and image payloads are exactly what push a request over a
    byte limit, and the token estimator counts none of them."""
    payload = {"messages": [{"role": "user", "content": "hi"}],
               "tools": [{"function": {"name": "f", "parameters": {"x": "y" * 4000}}}]}
    assert server._payload_size_bytes(payload) > 4000
    assert server._estimate_payload_tokens(payload) < 10


def test_an_unserializable_payload_disables_the_check_rather_than_guessing(server):
    class Weird:
        pass
    assert server._payload_size_bytes({"messages": [], "x": Weird()}) > 0  # default=str
    assert server._is_oversize_for("p1", "m1", 0) is False


# ── the ordering pass ───────────────────────────────────────────────────────

def _order(server, payload):
    ordered, n = server._demote_oversize_candidates(list(CANDS), payload)
    return [(pn, um) for pn, _pc, um in ordered], n


def test_an_empty_registry_is_a_free_no_op(server, monkeypatch):
    """The common case — no 413 has ever happened — must not even serialize."""
    called = []
    monkeypatch.setattr(server, "_payload_size_bytes",
                        lambda p: called.append(1) or 999)
    cands = list(CANDS)
    ordered, n = server._demote_oversize_candidates(cands, _payload(100))
    assert ordered is cands and n == 0
    assert called == []


def test_an_oversized_candidate_is_demoted_not_dropped(server):
    """Never drop: a pool where everything has 413'd must still serve, and a
    demoted candidate reached as a last resort can disprove its own watermark."""
    big = _payload(5000)
    server._record_oversize("p1", "m1", server._payload_size_bytes(big))
    ordered, n = _order(server, big)
    assert n == 1
    assert ordered == [("p2", "m2"), ("p1", "m1")]
    assert len(ordered) == len(CANDS)


def test_a_smaller_request_still_prefers_the_same_candidate(server):
    """The whole point of a size watermark rather than a cooldown."""
    big = _payload(5000)
    server._record_oversize("p1", "m1", server._payload_size_bytes(big))
    ordered, n = _order(server, _payload(10))
    assert n == 0
    assert ordered[0] == ("p1", "m1")


def test_relative_order_survives_among_candidates_that_still_fit(server):
    """Stable, so the benchmark rank / capacity decision made before this one is
    preserved among the candidates that are still plausible."""
    big = _payload(5000)
    server._record_oversize("p2", "m2", server._payload_size_bytes(big))
    ordered, _ = _order(server, big)
    assert ordered == [("p1", "m1"), ("p2", "m2")]


def test_a_pool_where_everything_is_oversized_still_serves(server):
    big = _payload(5000)
    size = server._payload_size_bytes(big)
    server._record_oversize("p1", "m1", size)
    server._record_oversize("p2", "m2", size)
    ordered, n = _order(server, big)
    assert n == 2
    assert len(ordered) == 2, "candidates must never be dropped"


# ── self-correction ─────────────────────────────────────────────────────────

def test_a_success_at_or_above_the_watermark_clears_it(server):
    """One spurious 413 must not sideline a model until the process restarts."""
    server._record_oversize("p1", "m1", 5000)
    server._note_accepted_size("p1", "m1", 5000)
    assert server._is_oversize_for("p1", "m1", 5000) is False


def test_a_smaller_success_proves_nothing_and_leaves_it_alone(server):
    server._record_oversize("p1", "m1", 5000)
    server._note_accepted_size("p1", "m1", 4000)
    assert server._is_oversize_for("p1", "m1", 5000) is True


def test_nothing_is_persisted_across_a_restart(server, monkeypatch, tmp_path):
    """Deliberately in memory: a body limit is cheap to relearn, and persisting
    it would buy a schema and a staleness problem and nothing else.

    A restart is a new process, so the routing state goes with it. Reloading the
    module alone does not model that -- the state backend is process-scoped and
    deliberately outlives a reload -- so this drops it the way a fork or a fresh
    boot does.
    """
    from llmproxy import state

    server._record_oversize("p1", "m1", 5000)
    state.reset_for_worker()
    fresh = _make_server(monkeypatch, tmp_path)
    assert fresh._is_oversize_for("p1", "m1", 5000) is False


# ── end to end through the cycling loop ─────────────────────────────────────

class _Resp:
    def __init__(self, status, content=b'{"choices":[{"message":{"content":"hi"}}]}'):
        import datetime
        self.status_code = status
        self.content = content
        self.headers = {"Content-Type": "application/json"}
        self.elapsed = datetime.timedelta(milliseconds=5)

    def get_data(self):
        return self.content


def test_a_413_records_the_size_and_the_next_request_routes_around_it(
        server, monkeypatch):
    seen = []

    def _post(url, **kw):
        seen.append(url)
        return _Resp(413, b'{"error":{"message":"request entity too large"}}') \
            if "p1" in url else _Resp(200)

    monkeypatch.setattr(server.requests, "post", _post)
    big = _payload(5000)
    with server.app.test_request_context():
        resp = server._proxy_cycling_non_streaming(
            "chat/completions", "t", list(CANDS), big, 5, config=None)

    assert resp.status_code == 200, "must fail over, as it already did"
    assert len(seen) == 2, "413 is not transient — no same-candidate retry"
    # And the size is now remembered, so an equally large request skips p1.
    ordered, n = _order(server, big)
    assert n == 1 and ordered[0] == ("p2", "m2")


def test_the_recorded_size_matches_what_the_ordering_pass_measures(server, monkeypatch):
    """The measure must be the CLIENT payload at both ends. Recording the
    per-candidate upstream body instead makes the watermark wrong by the length
    of the substituted model id, so a request of exactly the refused size no
    longer trips it."""
    monkeypatch.setattr(server.requests, "post",
                        lambda url, **kw: _Resp(413, b'{"error":{"message":"too large"}}'))
    big = _payload(5000)
    with server.app.test_request_context():
        server._proxy_cycling_non_streaming(
            "chat/completions", "t", [CANDS[0]], big, 5, config=None)

    # The exact same payload must now be recognised as oversized for p1/m1.
    assert server._is_oversize_for("p1", "m1", server._payload_size_bytes(big)) is True
    ordered, n = _order(server, big)
    assert n == 1 and ordered[0] == ("p2", "m2")


def test_a_413_does_not_cool_the_candidate_like_a_429(server, monkeypatch):
    """A 413 says the request was too big, not that the model is unavailable."""
    monkeypatch.setattr(server.requests, "post",
                        lambda url, **kw: _Resp(413, b'{"error":{"message":"too large"}}'))
    with server.app.test_request_context():
        server._proxy_cycling_non_streaming(
            "chat/completions", "t", list(CANDS), _payload(5000), 5, config=None)
    assert server._is_candidate_saturated("p1", "m1", None) is False
