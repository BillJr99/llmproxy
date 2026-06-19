"""Regression tests: local providers must never leak into believed_free.

Local providers (localhost / loopback) participate in the dedicated
llmproxy/local virtual-endpoint family. Their models must never appear in
config['believed_free'] (which feeds llmproxy/free) nor in the candidate
list for the /free family, even if a stale config still has them there.

These tests pin three independent paths that historically polluted
believed_free with local-provider models:

  1. _sync_local_provider_models_once: startup poll added local models
     to believed_free. (server.py)
  2. _get_free_model_candidates: didn't filter local providers, so any
     model with "free" in its ID served from localhost would route via
     llmproxy/free. (server.py)
  3. _offer_free_tier_auto_populate: ran for every newly-added provider,
     including ones pointed at a local URL. (setup_wizard.py)
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ─── 1. Sync routine no longer adds to believed_free; prunes existing pollution ───

def _load_server(monkeypatch, config_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(config_path))
    from llmproxy import config as config_mod
    importlib.reload(config_mod)
    from llmproxy import server as server_mod
    importlib.reload(server_mod)
    monkeypatch.setattr(server_mod, "_run_startup_tasks_once", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_maybe_fire_interval_probes", lambda *a, **k: None)
    return server_mod


@pytest.fixture
def config_with_polluted_local(tmp_path: Path) -> Path:
    """A config where a local provider has already polluted believed_free + free_limits."""
    cfg = {
        "providers": {
            "ollama": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model_filter": None,
            },
            "groq": {  # cloud — should not be touched
                "base_url": "https://api.groq.com/openai/v1",
                "api_key": "gsk_...",
                "model_filter": None,
            },
        },
        # Stale pollution from before the fix:
        "believed_free": [
            "ollama/llama3.2-3b",      # local → must be pruned
            "ollama/qwen2.5-14b",      # local → must be pruned
            "groq/llama-3.3-70b",      # cloud → must be retained
        ],
        "model_reasoning": {
            "ollama/llama3.2-3b": "exploratory",
            "groq/llama-3.3-70b": "standard",
        },
        "free_limits": {
            "ollama/llama3.2-3b": {"requests_per_minute": 30,
                                   "requests_per_day": None,
                                   "tokens_per_minute": None,
                                   "tokens_per_day": None},
            "groq/llama-3.3-70b": {"requests_per_minute": 30,
                                    "requests_per_day": 1000,
                                    "tokens_per_minute": 6000,
                                    "tokens_per_day": 500000},
        },
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def _fake_ollama_models_resp(model_ids):
    class _Resp:
        status_code = 200
        def raise_for_status(self): return None
        def json(self): return {"data": [{"id": m} for m in model_ids]}
    return _Resp()


def test_sync_prunes_local_from_believed_free(monkeypatch, config_with_polluted_local):
    """The startup sync must remove every ollama/* entry from believed_free."""
    server_mod = _load_server(monkeypatch, config_with_polluted_local)

    # Stub the HTTP call so the sync function thinks ollama serves these.
    with patch.object(server_mod, "requests") as req_mock:
        req_mock.get.return_value = _fake_ollama_models_resp(["llama3.2-3b", "qwen2.5-14b"])
        # Reset the once-flag so the sync actually runs (module reload already did).
        server_mod._local_sync_done = False
        # Patch threading so _run() executes synchronously (deterministic test).
        import threading as _real_threading

        class _ImmediateThread:
            def __init__(self, target, daemon=False, name=""):
                self._target = target
            def start(self):
                self._target()

        with patch.object(_real_threading, "Thread", _ImmediateThread):
            server_mod._sync_local_provider_models_once()

    # Re-read what got saved
    saved = json.loads(config_with_polluted_local.read_text())
    assert "ollama/llama3.2-3b" not in saved["believed_free"], \
        f"local model leaked into believed_free: {saved['believed_free']}"
    assert "ollama/qwen2.5-14b" not in saved["believed_free"]
    # Cloud entry must survive untouched.
    assert "groq/llama-3.3-70b" in saved["believed_free"]


def test_sync_prunes_local_from_free_limits(monkeypatch, config_with_polluted_local):
    """free_limits for local providers must also be pruned."""
    server_mod = _load_server(monkeypatch, config_with_polluted_local)
    import threading as _real_threading

    class _ImmediateThread:
        def __init__(self, target, daemon=False, name=""):
            self._target = target
        def start(self):
            self._target()

    with patch.object(server_mod, "requests") as req_mock:
        req_mock.get.return_value = _fake_ollama_models_resp(["llama3.2-3b"])
        server_mod._local_sync_done = False
        with patch.object(_real_threading, "Thread", _ImmediateThread):
            server_mod._sync_local_provider_models_once()

    saved = json.loads(config_with_polluted_local.read_text())
    assert "ollama/llama3.2-3b" not in saved["free_limits"]
    # Cloud entry retained
    assert "groq/llama-3.3-70b" in saved["free_limits"]


def test_sync_still_populates_model_reasoning(monkeypatch, config_with_polluted_local):
    """The /local/<level> family still needs model_reasoning entries — these
    must be added even though believed_free is not."""
    server_mod = _load_server(monkeypatch, config_with_polluted_local)
    import threading as _real_threading

    class _ImmediateThread:
        def __init__(self, target, daemon=False, name=""):
            self._target = target
        def start(self):
            self._target()

    with patch.object(server_mod, "requests") as req_mock:
        req_mock.get.return_value = _fake_ollama_models_resp(["new-model-3b", "huge-70b"])
        server_mod._local_sync_done = False
        with patch.object(_real_threading, "Thread", _ImmediateThread):
            server_mod._sync_local_provider_models_once()

    saved = json.loads(config_with_polluted_local.read_text())
    # Newly-discovered local models get a reasoning level so /local/<level> works.
    assert "ollama/new-model-3b" in saved["model_reasoning"]
    assert "ollama/huge-70b" in saved["model_reasoning"]
    # And nothing got added to believed_free.
    assert "ollama/new-model-3b" not in saved["believed_free"]
    assert "ollama/huge-70b" not in saved["believed_free"]


# ─── 2. _get_free_model_candidates filters local providers ───

def test_get_free_model_candidates_excludes_local(tmp_path, monkeypatch):
    """Even if a stale config has a local model in believed_free, the runtime
    free-candidate selector must filter it out (defence-in-depth)."""
    cfg = {
        "providers": {
            "ollama": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model_filter": None,
            },
            "groq": {
                "base_url": "https://api.groq.com/openai/v1",
                "api_key": "gsk_...",
                "model_filter": None,
            },
        },
        "believed_free": [
            "ollama/llama3.2-3b",       # stale pollution — must be skipped at runtime
            "groq/llama-3.3-70b",       # legitimate cloud free model
        ],
        "model_reasoning": {},
        "free_limits": {},
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    server_mod = _load_server(monkeypatch, p)

    # Prime the route cache so _get_route_cache_snapshot returns our two models.
    with patch.object(server_mod, "_get_route_cache_snapshot") as snap_mock:
        snap_mock.return_value = {
            "ollama/llama3.2-3b": ("ollama", "llama3.2-3b"),
            "groq/llama-3.3-70b": ("groq", "llama-3.3-70b"),
        }
        candidates = server_mod._get_free_model_candidates()

    provider_names = {c[0] for c in candidates}
    assert "ollama" not in provider_names, \
        "_get_free_model_candidates must skip local providers"
    assert "groq" in provider_names, "cloud free model still eligible for /free"


def test_get_free_model_candidates_excludes_local_even_with_free_in_id(tmp_path, monkeypatch):
    """A locally-served model whose id contains 'free' must STILL be excluded
    from /free routing."""
    cfg = {
        "providers": {
            "ollama": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model_filter": None,
            },
        },
        "believed_free": [],
        "model_reasoning": {},
        "free_limits": {},
        "server": {"host": "127.0.0.1", "port": 8080, "log_level": "ERROR",
                   "request_timeout": 5, "stream_timeout": 5},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    server_mod = _load_server(monkeypatch, p)

    with patch.object(server_mod, "_get_route_cache_snapshot") as snap_mock:
        snap_mock.return_value = {
            "ollama/free-thinking-model": ("ollama", "free-thinking-model"),
        }
        candidates = server_mod._get_free_model_candidates()
    assert candidates == [], \
        "local model with 'free' in its id must NOT route via /free"


# ─── 3. setup_wizard skips free-tier auto-populate for local providers ───

def test_local_provider_does_not_trigger_free_tier_prompt(monkeypatch):
    """When the user adds a local provider, _offer_free_tier_auto_populate
    must NOT be called — believed_free is irrelevant for local routes."""
    from llmproxy import setup_wizard as sw

    # Stub the interactive bits the wizard relies on.
    call_log: list[tuple[str, dict]] = []

    def fake_offer_free_tier(name, config):
        call_log.append(("offer_free", {"name": name}))
        return False

    def fake_auto_register_local(name, cfg, config):
        call_log.append(("auto_register_local", {"name": name}))
        return False

    monkeypatch.setattr(sw, "_offer_free_tier_auto_populate", fake_offer_free_tier)
    monkeypatch.setattr(sw, "_auto_register_local_models", fake_auto_register_local)

    # Simulate the branch in _providers_menu that fires after a provider is added.
    # We invoke the conditional directly so we don't need to mock the full TTY.
    cfg_local = {"base_url": "http://localhost:11434/v1", "api_key": "ollama"}
    cfg_cloud = {"base_url": "https://api.groq.com/openai/v1", "api_key": "gsk_..."}
    config: dict = {"providers": {}, "believed_free": [], "model_reasoning": {}}

    # Replicate the branch from setup_wizard._providers_menu (choice == 0 path):
    def _post_add(name, cfg):
        if sw._is_local_url(cfg.get("base_url", "")):
            sw._auto_register_local_models(name, cfg, config)
        else:
            sw._offer_free_tier_auto_populate(name, config)

    _post_add("ollama", cfg_local)
    _post_add("groq", cfg_cloud)

    # Local provider must only have called auto_register_local, never offer_free.
    local_calls = [c for c in call_log if c[1]["name"] == "ollama"]
    assert ("offer_free", {"name": "ollama"}) not in local_calls
    assert ("auto_register_local", {"name": "ollama"}) in local_calls

    # Cloud provider must call offer_free but not auto_register_local.
    cloud_calls = [c for c in call_log if c[1]["name"] == "groq"]
    assert ("offer_free", {"name": "groq"}) in cloud_calls
    assert ("auto_register_local", {"name": "groq"}) not in cloud_calls
