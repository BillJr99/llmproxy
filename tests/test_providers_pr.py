"""_maybe_open_providers_pr: opens the providers PR from computed content,
so a read-only deployment can still PR providers.json + config.example.json."""

from __future__ import annotations

import pytest

from llmproxy import github_pr, server


@pytest.fixture
def captured_pr(monkeypatch):
    captured: dict = {}

    def fake_create(*, token, owner, repo, base, branch, files, title, body):
        captured.update(token=token, owner=owner, repo=repo, base=base,
                        branch=branch, files=files, title=title, body=body)
        return "https://github.com/o/r/pull/1"

    monkeypatch.setattr(github_pr, "create_or_update_pr", fake_create)
    # Make sure no ambient token leaks in/out of the test.
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    return captured


def test_opens_pr_with_both_files_from_content(captured_pr):
    cfg = {"providers_pr": {"enabled": True, "repo": "o/r"}}
    server._maybe_open_providers_pr(cfg, '{"providers":{}}\n', '{"example":true}\n')
    assert captured_pr["owner"] == "o" and captured_pr["repo"] == "r"
    assert captured_pr["files"]["llmproxy/providers.json"] == '{"providers":{}}\n'
    assert captured_pr["files"]["config.example.json"] == '{"example":true}\n'


def test_omits_example_when_not_provided(captured_pr):
    cfg = {"providers_pr": {"enabled": True, "repo": "o/r"}}
    server._maybe_open_providers_pr(cfg, '{"providers":{}}\n', None)
    assert "config.example.json" not in captured_pr["files"]
    assert "llmproxy/providers.json" in captured_pr["files"]


def test_skipped_when_flag_off(captured_pr):
    server._maybe_open_providers_pr({"providers_pr": {"enabled": False}}, "x", "y")
    assert captured_pr == {}  # create_or_update_pr never called


def test_skipped_when_no_token(monkeypatch, captured_pr):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    cfg = {"providers_pr": {"enabled": True, "repo": "o/r"}}
    server._maybe_open_providers_pr(cfg, "x", "y")
    assert captured_pr == {}  # no token -> skipped before calling the API


def test_skipped_when_no_repo(captured_pr):
    server._maybe_open_providers_pr({"providers_pr": {"enabled": True}}, "x", "y")
    assert captured_pr == {}  # missing providers_pr.repo -> skipped
