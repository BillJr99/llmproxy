"""Tests for llmproxy.github_pr — PR creation via the GitHub API."""

from __future__ import annotations

import responses

from llmproxy.github_pr import _git_blob_sha, create_or_update_pr, parse_github_slug

API = "https://api.github.com"
OWNER, REPO = "billjr99", "llmproxy"


def test_parse_github_slug_https():
    assert parse_github_slug("https://github.com/billjr99/llmproxy.git") == ("billjr99", "llmproxy")


def test_parse_github_slug_ssh():
    assert parse_github_slug("git@github.com:billjr99/llmproxy.git") == ("billjr99", "llmproxy")


def test_parse_github_slug_non_github():
    assert parse_github_slug("https://gitlab.com/a/b.git") is None
    assert parse_github_slug("") is None


def _register_happy_path(pr_status=201):
    base = f"{API}/repos/{OWNER}/{REPO}"
    responses.add(responses.GET, f"{base}/git/ref/heads/main",
                  json={"object": {"sha": "basesha"}}, status=200)
    responses.add(responses.GET, f"{base}/git/commits/basesha",
                  json={"tree": {"sha": "basetree"}}, status=200)
    # Base tree has no matching blob → content differs → proceed to open the PR.
    responses.add(responses.GET, f"{base}/git/trees/basetree",
                  json={"tree": [], "truncated": False}, status=200)
    responses.add(responses.POST, f"{base}/git/blobs", json={"sha": "blob1"}, status=201)
    responses.add(responses.POST, f"{base}/git/trees", json={"sha": "newtree"}, status=201)
    responses.add(responses.POST, f"{base}/git/commits", json={"sha": "newcommit"}, status=201)
    responses.add(responses.POST, f"{base}/git/refs", json={}, status=201)
    responses.add(responses.POST, f"{base}/pulls",
                  json={"html_url": f"https://github.com/{OWNER}/{REPO}/pull/99"}, status=pr_status)


@responses.activate
def test_create_pr_happy_path():
    _register_happy_path()
    url = create_or_update_pr(
        token="t", owner=OWNER, repo=REPO, base="main", branch="auto/providers",
        files={"llmproxy/providers.json": "{}"}, title="x", body="y",
    )
    assert url == f"https://github.com/{OWNER}/{REPO}/pull/99"


@responses.activate
def test_no_pr_when_files_match_base():
    """If every file already matches the base branch, no commit/branch/PR is made."""
    content = '{"providers": {}}\n'
    base = f"{API}/repos/{OWNER}/{REPO}"
    responses.add(responses.GET, f"{base}/git/ref/heads/main",
                  json={"object": {"sha": "basesha"}}, status=200)
    responses.add(responses.GET, f"{base}/git/commits/basesha",
                  json={"tree": {"sha": "basetree"}}, status=200)
    # Base tree already holds this exact content (matching git blob sha).
    responses.add(responses.GET, f"{base}/git/trees/basetree",
                  json={"tree": [{"path": "llmproxy/providers.json", "type": "blob",
                                  "sha": _git_blob_sha(content)}],
                        "truncated": False}, status=200)

    url = create_or_update_pr(
        token="t", owner=OWNER, repo=REPO, base="main", branch="auto/providers",
        files={"llmproxy/providers.json": content}, title="x", body="y",
    )
    assert url is None
    # No write calls (blobs/trees/commits/refs/pulls) were made.
    assert all(c.request.method == "GET" for c in responses.calls)


@responses.activate
def test_derived_file_change_alone_does_not_open_pr():
    """When only a non-decisive (derived) file differs, no PR is opened.

    providers.json matches base but config.example.json differs — with
    decisive_paths restricted to providers.json, that must be treated as "no
    change" so a regenerated/version-skewed example can't churn a PR each run.
    """
    providers = '{"providers": {}}\n'
    base = f"{API}/repos/{OWNER}/{REPO}"
    responses.add(responses.GET, f"{base}/git/ref/heads/main",
                  json={"object": {"sha": "basesha"}}, status=200)
    responses.add(responses.GET, f"{base}/git/commits/basesha",
                  json={"tree": {"sha": "basetree"}}, status=200)
    responses.add(responses.GET, f"{base}/git/trees/basetree",
                  json={"tree": [{"path": "llmproxy/providers.json", "type": "blob",
                                  "sha": _git_blob_sha(providers)}],
                        "truncated": False}, status=200)

    url = create_or_update_pr(
        token="t", owner=OWNER, repo=REPO, base="main", branch="auto/providers",
        files={"llmproxy/providers.json": providers,
               "config.example.json": '{"example": "changed"}\n'},
        title="x", body="y",
        decisive_paths=["llmproxy/providers.json"],
    )
    assert url is None
    assert all(c.request.method == "GET" for c in responses.calls)


@responses.activate
def test_decisive_file_change_still_opens_pr():
    """A real providers.json change opens the PR even with decisive_paths set."""
    _register_happy_path()
    url = create_or_update_pr(
        token="t", owner=OWNER, repo=REPO, base="main", branch="auto/providers",
        files={"llmproxy/providers.json": '{"providers": {"x": 1}}\n',
               "config.example.json": '{"example": true}\n'},
        title="x", body="y",
        decisive_paths=["llmproxy/providers.json"],
    )
    assert url == f"https://github.com/{OWNER}/{REPO}/pull/99"


@responses.activate
def test_existing_branch_is_force_updated_and_pr_reused():
    base = f"{API}/repos/{OWNER}/{REPO}"
    responses.add(responses.GET, f"{base}/git/ref/heads/main",
                  json={"object": {"sha": "basesha"}}, status=200)
    responses.add(responses.GET, f"{base}/git/commits/basesha",
                  json={"tree": {"sha": "basetree"}}, status=200)
    responses.add(responses.GET, f"{base}/git/trees/basetree",
                  json={"tree": [], "truncated": False}, status=200)
    responses.add(responses.POST, f"{base}/git/blobs", json={"sha": "blob1"}, status=201)
    responses.add(responses.POST, f"{base}/git/trees", json={"sha": "newtree"}, status=201)
    responses.add(responses.POST, f"{base}/git/commits", json={"sha": "newcommit"}, status=201)
    # Branch already exists → 422 on create, then PATCH succeeds.
    responses.add(responses.POST, f"{base}/git/refs", json={"message": "exists"}, status=422)
    responses.add(responses.PATCH, f"{base}/git/refs/heads/auto/providers", json={}, status=200)
    # PR already exists → 422, then GET lists it.
    responses.add(responses.POST, f"{base}/pulls", json={"message": "exists"}, status=422)
    responses.add(responses.GET, f"{base}/pulls",
                  json=[{"head": {"ref": "auto/providers"},
                         "html_url": f"https://github.com/{OWNER}/{REPO}/pull/7"}], status=200)

    url = create_or_update_pr(
        token="t", owner=OWNER, repo=REPO, base="main", branch="auto/providers",
        files={"llmproxy/providers.json": "{}"}, title="x", body="y",
    )
    assert url == f"https://github.com/{OWNER}/{REPO}/pull/7"
