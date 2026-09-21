"""Shared fixtures for the llmproxy test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def sidecar_data() -> dict:
    """Live contents of llmproxy/providers.json."""
    from llmproxy.providers import load_data
    return load_data()


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """A minimal but valid llmproxy config on disk, returned as a path."""
    cfg = {
        "providers": {
            "fakeprov": {
                "base_url": "http://upstream.example/v1",
                "api_key": "fake-key",
                "model_filter": None,
            },
        },
        "believed_free": ["fakeprov/free-model"],
        "model_reasoning": {
            "fakeprov/free-model": "standard",
            "fakeprov/big-model": "deep",
        },
        "free_limits": {},
        # Keep the static fixture stable: don't let the on-by-default startup
        # reconcile rewrite it. (fakeprov isn't in the sidecar so this is a no-op
        # in practice, but it's explicit and future-proofs new providers.)
        "sync_believed_free_on_startup": False,
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "log_level": "ERROR",
            "request_timeout": 5,
            "stream_timeout": 5,
        },
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture(autouse=True)
def _fresh_state():
    """Give every test its own routing state.

    Quota counters, health windows, saturation cooldowns, capability gaps,
    oversize watermarks, affinity pins and the failure ring live in
    ``llmproxy.state`` behind one backend object. That object is process-scoped
    by design, which is correct in production and wrong in a test session: it
    outlives the ``importlib.reload(server)`` that several suites use to rebuild
    the module against a new config, so without this one test's cooldowns and
    counters would leak into the next.

    Autouse, because the leak is silent and order-dependent — the kind of thing
    that passes locally and fails in CI on a different shard.
    """
    from llmproxy import state
    state.set_backend(None)
    yield
    state.set_backend(None)
