"""_auto_register_local_models: local models go into model_reasoning only,
NOT believed_free (they have their own /local virtual endpoint family)."""

from __future__ import annotations

from unittest.mock import patch

from llmproxy.setup_wizard import _auto_register_local_models


def _fake_models_response(model_ids):
    class _Resp:
        status_code = 200
        def raise_for_status(self): return None
        def json(self): return {"data": [{"id": m} for m in model_ids]}
    return _Resp()


def test_local_models_added_to_reasoning_only():
    config: dict = {"believed_free": [], "model_reasoning": {}}
    provider_cfg = {"base_url": "http://localhost:11434/v1", "api_key": "ollama"}

    with patch("llmproxy.setup_wizard._requests.get",
               return_value=_fake_models_response(["llama3.2-3b", "qwen2.5-14b"])):
        changed = _auto_register_local_models("ollama", provider_cfg, config)

    assert changed is True
    # believed_free must stay empty — local models do NOT go here.
    assert config["believed_free"] == []
    # model_reasoning must contain both qualified IDs with inferred levels.
    assert config["model_reasoning"] == {
        "ollama/llama3.2-3b": "exploratory",
        "ollama/qwen2.5-14b": "exploratory",
    }


def test_slash_models_are_skipped():
    """Models like 'openai/gpt-4o' piped through OpenWebUI must be ignored —
    they're cloud-backed, not local."""
    config: dict = {"believed_free": [], "model_reasoning": {}}
    provider_cfg = {"base_url": "http://localhost:3000/v1", "api_key": "x"}

    with patch("llmproxy.setup_wizard._requests.get",
               return_value=_fake_models_response(["llama3", "openai/gpt-4o"])):
        _auto_register_local_models("openwebui", provider_cfg, config)

    assert "openwebui/llama3" in config["model_reasoning"]
    assert "openwebui/openai/gpt-4o" not in config["model_reasoning"]


def test_existing_reasoning_entries_preserved():
    config: dict = {
        "believed_free": [],
        "model_reasoning": {"ollama/llama3": "deep"},  # manually tagged
    }
    provider_cfg = {"base_url": "http://localhost:11434/v1", "api_key": "ollama"}

    with patch("llmproxy.setup_wizard._requests.get",
               return_value=_fake_models_response(["llama3", "phi3"])):
        _auto_register_local_models("ollama", provider_cfg, config)

    assert config["model_reasoning"]["ollama/llama3"] == "deep"  # unchanged
    assert "ollama/phi3" in config["model_reasoning"]  # newly added
    assert config["believed_free"] == []


def test_unreachable_provider_returns_false():
    config: dict = {"believed_free": [], "model_reasoning": {}}
    provider_cfg = {"base_url": "http://localhost:11434/v1", "api_key": "x"}

    with patch("llmproxy.setup_wizard._requests.get",
               side_effect=Exception("connection refused")):
        changed = _auto_register_local_models("ollama", provider_cfg, config)

    assert changed is False
    assert config == {"believed_free": [], "model_reasoning": {}}
