"""Tests for the /v1/usage accounting endpoint and believed_free cost flagging."""

from __future__ import annotations

import datetime
import importlib
import json
import threading
from pathlib import Path

import pytest


def _load_server_with_config(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def usage_server(tmp_path, monkeypatch):
    cfg = {
        "providers": {
            "groq": {"base_url": "http://groq.example/v1", "api_key": "k", "model_filter": None},
        },
        "believed_free": ["groq/free-model"],
        "free_limits": {},
        # Keep the on-disk config stable: don't let the background startup sync
        # rewrite it (it would race the runtime cost_observed_free_tier write).
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


def test_usage_key_anonymous_matches_legacy_form(usage_server):
    # No account -> byte-identical to the historical provider/model key.
    assert usage_server._usage_key("groq", "free-model") == "groq/free-model"
    assert usage_server._usage_key("groq", "free-model", None) == "groq/free-model"


def test_usage_key_scopes_by_account(usage_server):
    assert usage_server._usage_key("groq", "free-model", "team-a") == "groq#team-a/free-model"


def test_per_account_usage_is_metered_separately(usage_server):
    # Two accounts of the same provider/model each keep their own counters.
    common = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    cfg = usage_server.load_config()
    usage_server._record_usage("groq", "free-model", usage=common, config=cfg, account_id="a")
    usage_server._record_usage("groq", "free-model", usage=common, config=cfg, account_id="a")
    usage_server._record_usage("groq", "free-model", usage=common, config=cfg, account_id="b")

    reg = usage_server._usage_registry
    assert "groq#a/free-model" in reg
    assert "groq#b/free-model" in reg
    assert reg["groq#a/free-model"].snapshot()[1] == 2  # requests_today
    assert reg["groq#b/free-model"].snapshot()[1] == 1
    # The anonymous per-model key is untouched when accounts are used.
    assert "groq/free-model" not in reg


def test_per_account_cost_flag_is_model_level(usage_server):
    # A believed_free model reporting cost flags at MODEL granularity even when
    # the request was recorded under a specific account.
    usage_server._record_usage(
        "groq", "free-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.002},
        config=usage_server.load_config(),
        account_id="team-a",
    )
    body = usage_server.app.test_client().get("/v1/usage").get_json()
    flagged = body["flagged_paid_free_models"]
    assert len(flagged) == 1
    assert flagged[0]["model"] == "groq/free-model"


def test_usage_report_breaks_down_by_account(usage_server):
    cfg = usage_server.load_config()
    common = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    usage_server._record_usage("groq", "free-model", usage=common, config=cfg, account_id="team-a")
    usage_server._record_usage("groq", "free-model", usage=common, config=cfg, account_id="team-b")
    usage_server._record_usage("groq", "free-model", usage=common, config=cfg, account_id="team-b")
    body = usage_server.app.test_client().get("/v1/usage").get_json()
    by_acct = {m["account"]: m for m in body["models"] if "account" in m}
    assert set(by_acct) == {"team-a", "team-b"}
    # Same clean model id on each row, distinct per-account request counts.
    assert all(m["model"] == "groq/free-model" for m in by_acct.values())
    assert by_acct["team-a"]["requests"] == 1
    assert by_acct["team-b"]["requests"] == 2
    # Totals aggregate across accounts.
    assert body["totals"]["requests"] == 3


def test_usage_report_anonymous_has_no_account_field(usage_server):
    usage_server._record_usage(
        "groq", "free-model",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        config=usage_server.load_config(),
    )
    m = usage_server.app.test_client().get("/v1/usage").get_json()["models"][0]
    assert m["model"] == "groq/free-model"
    assert "account" not in m


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
    # The observation reclassifies the model as paid (recorded in
    # cost_observed_free_tier), so the per-model row is no longer believed_free
    # and the cost is no longer "unexpected" — the durable signal is the
    # flagged_paid_free_models list above.
    assert body["models"][0]["believed_free"] is False
    assert body["models"][0]["unexpected_cost"] is False


def test_cost_observed_persisted_to_config(usage_server, tmp_path):
    # First observation of a cost on a believed_free model is appended to the live
    # config's cost_observed_free_tier (original-cased qualified id), exactly once.
    usage_server._record_usage(
        "groq", "free-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.002},
        config=usage_server.load_config(),
    )
    written = json.loads((tmp_path / "config.json").read_text())
    assert written["cost_observed_free_tier"] == ["groq/free-model"]

    # A second hit must not duplicate the entry.
    usage_server._record_usage(
        "groq", "free-model",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.003},
        config=usage_server.load_config(),
    )
    written = json.loads((tmp_path / "config.json").read_text())
    assert written["cost_observed_free_tier"] == ["groq/free-model"]


def test_flag_paid_free_reports_first_observation_only(usage_server):
    assert usage_server._flag_paid_free("groq/m", 0.002, "provider") is True
    assert usage_server._flag_paid_free("groq/m", 0.004, "provider") is False


def test_cost_observed_makes_model_not_free_immediately(usage_server):
    s = usage_server
    cfg = {"believed_free": ["groq/m"], "cost_observed_free_tier": ["groq/m"]}
    # Even though it's still in believed_free, the cost-observed entry wins.
    assert s._is_model_free("groq", "m", cfg) is False
    # And without the cost-observed entry it would be free.
    assert s._is_model_free("groq", "m", {"believed_free": ["groq/m"]}) is True


def test_cost_observed_overrides_free_substring(usage_server):
    s = usage_server
    cfg = {"believed_free": [], "cost_observed_free_tier": ["groq/some-free-model"]}
    # 'free' substring would normally force free; cost-observed overrides it.
    assert s._is_model_free("groq", "some-free-model", cfg) is False


def test_cost_observed_is_paid_tier_in_loadbalancer(usage_server):
    s = usage_server
    pc = {"base_url": "http://groq.example/v1", "api_key": "k"}
    cfg = {"believed_free": ["groq/m"], "cost_observed_free_tier": ["groq/m"]}
    assert s._cost_tier("groq", "m", pc, cfg) == s._TIER_PAID


def test_no_cost_observed_when_model_not_free(usage_server, tmp_path):
    # A model that isn't believed_free reporting a cost is expected — don't record it.
    usage_server._record_usage(
        "groq", "paid-model",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.01},
        config=usage_server.load_config(),
    )
    written = json.loads((tmp_path / "config.json").read_text())
    assert "paid-model" not in str(written.get("cost_observed_free_tier", []))


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


# ---------------------------------------------------------------------------
# _react_to_cost_observed_async — background propagation to sidecar / PR
# ---------------------------------------------------------------------------

def test_react_to_cost_observed_skips_when_flags_off(usage_server, monkeypatch):
    """Background propagation is a no-op when neither update_on_startup nor pr.enabled is set."""
    calls = []
    monkeypatch.setattr(usage_server, "_run_free_models_update", lambda *a, **kw: calls.append(1))
    usage_server._cost_observed_reaction_inflight = False
    usage_server._react_to_cost_observed_async()
    # The default fixture config has neither flag set, so no thread should have been spawned.
    # Give threads a brief window just in case, then assert nothing ran.
    import time
    time.sleep(0.05)
    assert calls == []


def test_react_to_cost_observed_calls_updater_when_update_on_startup(usage_server, monkeypatch):
    """Background thread calls _run_free_models_update when free_tier.update_on_startup is True."""
    base_cfg = dict(usage_server.load_config())
    patched_cfg = {**base_cfg, "free_tier": {"update_on_startup": True}}
    monkeypatch.setattr(usage_server, "load_config", lambda: patched_cfg)

    done = threading.Event()
    calls = []

    def _fake_update(*a, **kw):
        calls.append(1)
        done.set()

    monkeypatch.setattr(usage_server, "_run_free_models_update", _fake_update)
    usage_server._cost_observed_reaction_inflight = False
    usage_server._react_to_cost_observed_async()
    done.wait(timeout=2.0)
    assert calls == [1]


def test_react_to_cost_observed_inflight_guard_deduplicates(usage_server, monkeypatch):
    """Concurrent observations fire the updater exactly once; the inflight guard drops the second."""
    import time

    base_cfg = dict(usage_server.load_config())
    patched_cfg = {**base_cfg, "free_tier": {"update_on_startup": True}}
    monkeypatch.setattr(usage_server, "load_config", lambda: patched_cfg)

    done = threading.Event()
    calls = []

    def _slow_update(*a, **kw):
        calls.append(1)
        time.sleep(0.05)
        done.set()

    monkeypatch.setattr(usage_server, "_run_free_models_update", _slow_update)
    usage_server._cost_observed_reaction_inflight = False
    usage_server._react_to_cost_observed_async()
    usage_server._react_to_cost_observed_async()  # second call — inflight guard must drop this
    done.wait(timeout=2.0)
    assert calls == [1]


def test_react_to_cost_observed_calls_updater_when_pr_enabled(usage_server, monkeypatch):
    """Background thread fires when providers_pr.enabled is True even without update_on_startup."""
    base_cfg = dict(usage_server.load_config())
    patched_cfg = {**base_cfg, "providers_pr": {"enabled": True}}
    monkeypatch.setattr(usage_server, "load_config", lambda: patched_cfg)

    done = threading.Event()
    calls = []

    def _fake_update(*a, **kw):
        calls.append(1)
        done.set()

    monkeypatch.setattr(usage_server, "_run_free_models_update", _fake_update)
    usage_server._cost_observed_reaction_inflight = False
    usage_server._react_to_cost_observed_async()
    done.wait(timeout=2.0)
    assert calls == [1]


def test_maybe_open_providers_pr_respects_frequency_days(usage_server, monkeypatch):
    """PR creation is skipped when frequency_days has not elapsed since the last PR.

    The updater still runs to keep the sidecar fresh; only the GitHub API call
    is suppressed by the throttle.
    """
    from llmproxy import config as config_mod

    # Simulate a PR opened moments ago so frequency_days=7 has not elapsed.
    recent = datetime.datetime.now(datetime.UTC).isoformat()
    monkeypatch.setattr(config_mod, "load_pr_state", lambda *a, **kw: {"last_pr_at": recent})

    pr_create_calls = []

    def _fake_create_or_update_pr(**kw):
        pr_create_calls.append(kw)
        return None

    # Patch create_or_update_pr at the module level so the lazy import inside
    # _maybe_open_providers_pr picks it up.
    from llmproxy import github_pr as github_pr_mod
    monkeypatch.setattr(github_pr_mod, "create_or_update_pr", _fake_create_or_update_pr)

    cfg = {
        "providers_pr": {
            "enabled": True,
            "frequency_days": 7,
            "repo": "owner/repo",
            "token": "fake-token",
        }
    }
    usage_server._maybe_open_providers_pr(cfg, '{"providers":{}}')
    assert pr_create_calls == []
