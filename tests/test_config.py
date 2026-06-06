"""Coverage of llmproxy/config.py — load, save, hot-reload, schema fallback."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from llmproxy import config as config_mod


def test_load_returns_defaults_when_file_missing(tmp_path: Path):
    path = tmp_path / "nope.json"
    cfg = config_mod.load_config(str(path), force_reload=True)
    assert cfg["providers"] == {}
    assert cfg["believed_free"] == []


def test_save_then_load_roundtrip(tmp_path: Path):
    path = tmp_path / "cfg.json"
    src = {
        "providers": {"p": {"base_url": "http://x", "api_key": "k", "model_filter": None}},
        "believed_free": ["p/m"],
    }
    assert config_mod.save_config(src, str(path)) is True
    cfg = config_mod.load_config(str(path), force_reload=True)
    assert cfg["providers"]["p"]["base_url"] == "http://x"
    assert cfg["believed_free"] == ["p/m"]


def test_hot_reload_picks_up_changes(tmp_path: Path):
    path = tmp_path / "cfg.json"
    config_mod.save_config({"providers": {}, "believed_free": ["a"]}, str(path))
    first = config_mod.load_config(str(path), force_reload=True)
    assert first["believed_free"] == ["a"]

    # Sleep ensures mtime advances on filesystems with 1s resolution.
    time.sleep(1.05)
    config_mod.save_config({"providers": {}, "believed_free": ["b"]}, str(path))
    second = config_mod.load_config(str(path))
    assert second["believed_free"] == ["b"]


def test_parse_model_string_basic():
    p, m = config_mod.parse_model_string("openai/gpt-4o")
    assert p == "openai"
    assert m == "gpt-4o"


def test_parse_model_string_nested_slashes():
    p, m = config_mod.parse_model_string("openrouter/foo/bar:free")
    assert p == "openrouter"
    assert m == "foo/bar:free"


def test_parse_model_string_no_slash_raises():
    with pytest.raises(ValueError):
        config_mod.parse_model_string("noslash")


def test_model_is_allowed_filter_semantics():
    assert config_mod.model_is_allowed({"model_filter": None}, "x") is True
    assert config_mod.model_is_allowed({"model_filter": []}, "x") is False
    assert config_mod.model_is_allowed({"model_filter": ["x"]}, "x") is True
    assert config_mod.model_is_allowed({"model_filter": ["y"]}, "x") is False


def test_get_provider_helper():
    cfg = {"providers": {"p": {"base_url": "u"}}}
    assert config_mod.get_provider(cfg, "p") == {"base_url": "u"}
    assert config_mod.get_provider(cfg, "missing") is None


def test_heal_adds_models_url_for_github():
    cfg = {"providers": {"github": {
        "base_url": "https://models.github.ai/inference",
        "api_key": "ghp_x",
        "model_filter": None,
    }}}
    cfg, changed, messages = config_mod.heal_config(cfg)
    assert changed is True
    assert cfg["providers"]["github"]["models_url"] == "https://models.github.ai/catalog/models"
    assert any(level == "info" for level, _ in messages)


def test_heal_substitutes_account_id_for_cloudflare():
    cfg = {"providers": {"cloudflare-workers": {
        "base_url": "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1",
        "api_key": "k",
        "model_filter": None,
    }}}
    cfg, changed, _ = config_mod.heal_config(cfg)
    prov = cfg["providers"]["cloudflare-workers"]
    assert changed is True
    assert prov["models_url"] == (
        "https://api.cloudflare.com/client/v4/accounts/abc123/ai/models/search?per_page=100"
    )
    assert prov["models_id_field"] == "name"
    assert prov["models_keep_task"] == "Text Generation"


def test_heal_matches_renamed_provider_by_base_url():
    cfg = {"providers": {"my-gh": {
        "base_url": "https://models.github.ai/inference",
        "api_key": "ghp_x",
    }}}
    cfg, changed, _ = config_mod.heal_config(cfg)
    assert changed is True
    assert cfg["providers"]["my-gh"]["models_url"] == "https://models.github.ai/catalog/models"


def test_heal_warns_when_account_id_unrecoverable():
    # A cloudflare-workers provider whose base_url doesn't carry a parseable
    # account id: the static fields heal, but models_url needs {account_id}
    # we can't recover, so it is warned about rather than fabricated.
    cfg = {"providers": {"cloudflare-workers": {
        "base_url": "https://custom.example.com/v1",
        "api_key": "k",
    }}}
    cfg, _, messages = config_mod.heal_config(cfg)
    prov = cfg["providers"]["cloudflare-workers"]
    assert "models_url" not in prov
    assert prov["models_id_field"] == "name"
    assert any(level == "warning" and "models_url" in text for level, text in messages)


def test_heal_never_overwrites_existing_field():
    cfg = {"providers": {"github": {
        "base_url": "https://models.github.ai/inference",
        "api_key": "ghp_x",
        "models_url": "https://my.custom/models",
    }}}
    cfg, changed, _ = config_mod.heal_config(cfg)
    assert changed is False
    assert cfg["providers"]["github"]["models_url"] == "https://my.custom/models"


def test_heal_is_idempotent():
    cfg = {"providers": {"github": {
        "base_url": "https://models.github.ai/inference",
        "api_key": "ghp_x",
    }}}
    cfg, changed1, _ = config_mod.heal_config(cfg)
    assert changed1 is True
    cfg, changed2, messages2 = config_mod.heal_config(cfg)
    assert changed2 is False
    assert messages2 == []


def test_heal_ignores_unknown_provider():
    cfg = {"providers": {"custom": {
        "base_url": "https://api.unknown-vendor.example/v1",
        "api_key": "k",
    }}}
    cfg, changed, messages = config_mod.heal_config(cfg)
    assert changed is False
    assert messages == []
    assert "models_url" not in cfg["providers"]["custom"]


def test_heal_skips_null_template_field_without_warning(monkeypatch):
    # A template that carries an explicit null/empty value for a healable field
    # (copied verbatim from providers.json) must be skipped silently rather than
    # warned about — there's nothing usable to backfill.
    monkeypatch.setattr(config_mod._providers, "get_provider_templates", lambda: [
        {"key": "nullprov", "base_url": "https://api.nullprov.example/v1",
         "models_url": None, "models_id_field": ""},
    ])
    cfg = {"providers": {"nullprov": {
        "base_url": "https://api.nullprov.example/v1",
        "api_key": "k",
    }}}
    cfg, changed, messages = config_mod.heal_config(cfg)
    assert changed is False
    assert messages == []
    assert "models_url" not in cfg["providers"]["nullprov"]
    assert "models_id_field" not in cfg["providers"]["nullprov"]


def test_get_config_path_priority(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LLMPROXY_CONFIG", str(tmp_path / "env.json"))
    # Explicit override wins over env var
    p = config_mod.get_config_path(str(tmp_path / "cli.json"))
    assert p == tmp_path / "cli.json"
    # Env var beats default
    p = config_mod.get_config_path(None)
    assert p == tmp_path / "env.json"
