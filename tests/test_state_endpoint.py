"""GET /v1/state — whether sharing is actually on.

The question has no other symptom. A deployment that asked for four workers and
fell back to one because its state directory was unwritable behaves perfectly
correctly, just without the parallelism, and nothing else in the proxy would say
so. The same endpoint answers "why did that scheduled job not run", by naming
who holds its lease.
"""

from __future__ import annotations

import importlib
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
def server(monkeypatch, minimal_config):
    return _load_server_with_config(monkeypatch, minimal_config)


@pytest.fixture
def client(server):
    server.app.config["TESTING"] = True
    return server.app.test_client()


def test_it_is_admin_gated(client, server, monkeypatch):
    """It names a filesystem path and the worker identities, which is more than
    an unauthenticated endpoint should hand out -- unlike /v1/usage."""
    from llmproxy import admin
    monkeypatch.setattr(admin, "enforce_admin_auth", lambda: ({"error": "nope"}, 403))
    assert client.get("/v1/state").status_code == 403


def test_it_reports_the_in_process_backend(client):
    body = client.get("/v1/state").get_json()
    assert body["object"] == "llmproxy.state"
    assert body["backend"] == "memory"
    assert body["shared"] is False


def test_it_is_reachable_bare_and_under_api(client):
    assert client.get("/state").status_code == 200
    assert client.get("/api/v1/state").status_code == 200


def test_it_reports_a_shared_backend_and_its_path(client, tmp_path):
    from llmproxy import state
    backend = state.SqliteState(tmp_path / "shared_state.db")
    state.set_backend(backend)
    try:
        body = client.get("/v1/state").get_json()
        assert body["backend"] == "sqlite"
        assert body["shared"] is True
        assert body["path"] == str(tmp_path / "shared_state.db")
        assert body["journal_mode"] == "wal", "WAL is what lets readers not block"
    finally:
        state.set_backend(None)
        backend.close()


def test_it_names_who_holds_a_lease(client, tmp_path):
    """The question being asked is why a scheduled job did not run."""
    from llmproxy import state
    backend = state.SqliteState(tmp_path / "shared_state.db")
    state.set_backend(backend)
    try:
        backend.acquire_lease("free-models-update", 300)
        held = client.get("/v1/state").get_json()["leases"]
        assert [row["job"] for row in held] == ["free-models-update"]
        assert held[0]["holder"]
        assert held[0]["expires_in"] > 0
    finally:
        state.set_backend(None)
        backend.close()


def test_it_counts_what_is_held(client, server):
    server._record_usage("fakeprov", "free-model", usage=None)
    server._mark_saturated("fakeprov/free-model")
    body = client.get("/v1/state").get_json()
    assert body["metered_keys"] >= 1
    assert body["cooling_keys"] >= 1


def test_a_broken_backend_does_not_500_the_endpoint(client, monkeypatch, server):
    """A diagnostic that crashes when things are wrong is worse than none."""
    class _Broken:
        kind = "broken"
        def diagnostics(self):
            raise RuntimeError("disk gone")

    monkeypatch.setattr(server, "get_backend", lambda: _Broken())
    body = client.get("/v1/state").get_json()
    assert body["object"] == "llmproxy.state"
    assert "disk gone" in body["error"]
