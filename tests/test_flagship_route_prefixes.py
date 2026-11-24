"""Gateway route prefixes (coding-, cc-) must not split a model from its scores."""

from __future__ import annotations

import pytest

from llmproxy.flagship import normalize_model_id


@pytest.mark.parametrize("upstream, key", [
    ("coding-glm-5.4", "glm54"),
    ("coding-glm-5.4-free", "glm54"),
    ("coding-glm-5.3-flash-free", "glm53flash"),
    ("coding-kimi-k3-free", "kimik3"),
    ("coding-minimax-m3-free", "minimaxm3"),
    ("cc-glm-5", "glm5"),
    ("cc-glm-5-turbo", "glm5turbo"),
    ("vendor/coding-glm-5.4", "glm54"),
])
def test_route_prefixes_join_the_plain_model(upstream, key):
    assert normalize_model_id(upstream) == key


def test_prefixed_and_plain_ids_share_a_key():
    assert normalize_model_id("coding-glm-5.4-free") == normalize_model_id("z-ai/glm-5.4")
    assert normalize_model_id("cc-glm-5") == normalize_model_id("zai-org/glm-5")


@pytest.mark.parametrize("upstream", [
    "qwen3-coder",          # "coder" mid-name is a different model line
    "kimi-for-coding",      # "coding" not at the start
    "codestral-latest",     # no hyphenated prefix
    "ccx-model",            # only the exact "cc-" prefix is a route marker
    "vendor-coding-model",
])
def test_names_that_merely_contain_the_words_are_untouched(upstream):
    import re
    expected = re.sub(r"[^a-z0-9]", "", re.sub(r"-(?:latest)$", "", upstream))
    assert normalize_model_id(upstream) == expected
