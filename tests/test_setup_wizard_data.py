"""Verify the setup_wizard module loads cleanly from the sidecar."""

from __future__ import annotations

from llmproxy.free_models import load_data


def test_provider_templates_loaded():
    from llmproxy.setup_wizard import PROVIDER_TEMPLATES
    data = load_data()
    assert len(PROVIDER_TEMPLATES) == len(data["provider_order"])
    keys = [t["key"] for t in PROVIDER_TEMPLATES]
    assert keys == data["provider_order"]


def test_provider_free_info_shapes_match_legacy():
    """Every provider entry must have the three legacy fields, even if empty."""
    from llmproxy.setup_wizard import PROVIDER_FREE_INFO
    for _pkey, info in PROVIDER_FREE_INFO.items():
        assert isinstance(info["believed_free"], list)
        assert isinstance(info["model_reasoning"], dict)
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
