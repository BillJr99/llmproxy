"""Verify the setup_wizard module loads cleanly from the sidecar."""

from __future__ import annotations

from llmproxy.providers import load_data


def test_provider_templates_loaded():
    from llmproxy.setup_wizard import PROVIDER_TEMPLATES
    data = load_data()
    assert len(PROVIDER_TEMPLATES) == len(data["provider_order"])
    keys = [t["key"] for t in PROVIDER_TEMPLATES]
    assert keys == data["provider_order"]


def test_provider_free_info_shapes_match_legacy():
    """Every provider entry must have the metadata fields, even if empty."""
    from llmproxy.setup_wizard import PROVIDER_FREE_INFO
    for _pkey, info in PROVIDER_FREE_INFO.items():
        assert isinstance(info["believed_free"], list)
        assert isinstance(info["model_reasoning"], dict)
        assert isinstance(info["model_capabilities"], dict)
        assert isinstance(info["free_limits"], dict)


def test_each_template_has_display_and_base_url():
    from llmproxy.setup_wizard import PROVIDER_TEMPLATES
    for t in PROVIDER_TEMPLATES:
        assert t.get("display"), f"{t['key']} missing display"
        assert t.get("base_url"), f"{t['key']} missing base_url"


def test_account_id_required_templates_use_placeholder():
    from llmproxy.setup_wizard import PROVIDER_TEMPLATES
    for t in PROVIDER_TEMPLATES:
        if t.get("account_id_required"):
            assert "{account_id}" in t["base_url"]
        if t.get("gateway_id_required"):
            assert "{gateway_id}" in t["base_url"]


def test_docker_model_runner_template_prompts_for_url_and_optional_key():
    from llmproxy.setup_wizard import PROVIDER_TEMPLATES, _is_local_url
    tmpl = next(t for t in PROVIDER_TEMPLATES if t["key"] == "docker-model-runner")
    assert tmpl["base_url"] == "http://localhost:12434/engines/v1"
    assert tmpl["url_prompt"] is True
    assert not tmpl.get("key_required")
    assert _is_local_url(tmpl["base_url"])
    assert _is_local_url("http://model-runner.docker.internal/engines/v1")


def test_url_prompt_template_takes_entered_url_and_blank_key(monkeypatch):
    from llmproxy import setup_wizard as sw
    idx = next(i for i, t in enumerate(sw.PROVIDER_TEMPLATES)
               if t["key"] == "docker-model-runner")
    answers = iter(["", "http://model-runner.docker.internal/engines/v1/"])
    monkeypatch.setattr(sw, "_pick", lambda *a, **k: idx)
    monkeypatch.setattr(sw, "_prompt", lambda label, default=None, secret=False:
                        next(answers) or (default or ""))
    monkeypatch.setattr(sw.getpass, "getpass", lambda *a, **k: "")
    key, cfg = sw._setup_from_template({})
    assert key == "docker-model-runner"
    assert cfg["base_url"] == "http://model-runner.docker.internal/engines/v1"
    assert cfg["api_key"] == ""
