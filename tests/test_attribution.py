"""Guards that adapted third-party work stays attributed.

llmproxy carries code adapted from Apache-2.0 and MIT projects. Those licences
require the notices to travel with the code, and the easy way to break that is
for a later port to land without anyone noticing. This asserts the paperwork
exists, in the same spirit as the config.example.json drift guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTICES = REPO_ROOT / "THIRD_PARTY_NOTICES.md"
UPSTREAMS = ("Switchyard", "OmniRoute")


def _source_files():
    for path in sorted((REPO_ROOT / "llmproxy").rglob("*.py")):
        yield path
    for path in sorted((REPO_ROOT / "scripts").rglob("*.py")):
        yield path


def test_license_file_exists_and_is_apache_2():
    text = (REPO_ROOT / "LICENSE").read_text()
    assert "Apache License" in text
    assert "Version 2.0" in text
    # The appendix placeholder must be filled in, not shipped as a template.
    assert "[yyyy]" not in text and "[name of copyright owner]" not in text


def test_notices_file_exists():
    assert NOTICES.is_file(), "THIRD_PARTY_NOTICES.md is required"


@pytest.mark.parametrize("upstream", UPSTREAMS)
def test_each_upstream_is_documented_with_its_licence(upstream):
    text = NOTICES.read_text()
    assert upstream in text
    assert re.search(r"Apache License 2\.0|MIT License", text)


def test_switchyard_notice_is_reproduced():
    """Apache-2.0 s4(d): the upstream NOTICE contents must travel with the code."""
    text = NOTICES.read_text()
    assert "NVIDIA CORPORATION & AFFILIATES" in text
    assert "http://www.apache.org/licenses/LICENSE-2.0" in text


def test_omniroute_mit_notice_is_reproduced():
    """MIT requires the copyright line and permission notice in full."""
    text = NOTICES.read_text()
    assert "Copyright (c) 2026 diegosouzapw" in text
    assert "WITHOUT WARRANTY OF ANY KIND" in text


def test_adapted_files_are_listed_in_the_notices():
    """Any source file naming an upstream must have a notices entry."""
    notices = NOTICES.read_text()
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not any(u in text for u in UPSTREAMS):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        assert rel in notices, (
            f"{rel} references an upstream project but has no "
            f"THIRD_PARTY_NOTICES.md entry"
        )


def test_signals_module_states_its_origin_and_changes():
    """Apache-2.0 s4(b): derivative files must carry a change statement."""
    text = (REPO_ROOT / "llmproxy" / "signals.py").read_text()
    assert "Switchyard" in text
    assert "Apache-2.0" in text
    assert "THIRD_PARTY_NOTICES.md" in text
    assert "Translated from Rust" in text


def test_readme_acknowledges_both_upstreams():
    text = (REPO_ROOT / "README.md").read_text()
    assert "## Acknowledgements" in text
    for upstream in UPSTREAMS:
        assert upstream in text
