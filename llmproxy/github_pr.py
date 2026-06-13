"""github_pr.py — open a pull request from file contents via the GitHub API.

Used by the optional ``pr_providers_list`` feature: when the startup updater
rewrites ``providers.json`` locally, the server can push those changes to a
branch and open a PR against the default branch — entirely through the GitHub
REST/Git-Data API, so it never mutates the local git checkout (or even requires
one). It only needs a token, an ``owner/repo`` slug, and the file contents.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger("llmproxy.github_pr")

_API = "https://api.github.com"
_TIMEOUT = (5, 20)


def parse_github_slug(remote_url: str) -> tuple[str, str] | None:
    """Return (owner, repo) from a GitHub remote URL, or None if not GitHub.

    Handles both ``https://github.com/owner/repo(.git)`` and
    ``git@github.com:owner/repo(.git)`` forms.
    """
    if not remote_url:
        return None
    url = remote_url.strip()
    if url.startswith("git@github.com:"):
        path = url[len("git@github.com:"):]
    elif "github.com/" in url:
        path = url.split("github.com/", 1)[1]
    else:
        return None
    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _gh(method: str, url: str, token: str, *, json: dict | None = None) -> requests.Response:
    return requests.request(
        method, url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json=json,
        timeout=_TIMEOUT,
    )


def create_or_update_pr(
    *,
    token: str,
    owner: str,
    repo: str,
    base: str,
    branch: str,
    files: dict[str, str],
    title: str,
    body: str,
) -> str | None:
    """Commit *files* onto *branch* and open a PR into *base*.

    *files* maps repo-relative paths to their new text contents. Returns the PR
    URL (or the branch's existing-PR note) on success, or raises on a hard API
    error. The branch ref is force-updated if it already exists, and an existing
    open PR for the branch is reused (so repeated runs just refresh it).
    """
    # 1. Resolve the base branch head + its tree.
    ref = _gh("GET", f"{_API}/repos/{owner}/{repo}/git/ref/heads/{base}", token)
    ref.raise_for_status()
    base_sha = ref.json()["object"]["sha"]
    commit = _gh("GET", f"{_API}/repos/{owner}/{repo}/git/commits/{base_sha}", token)
    commit.raise_for_status()
    base_tree = commit.json()["tree"]["sha"]

    # 2. Create blobs + a tree layered on the base tree.
    tree_entries = []
    for path, content in files.items():
        blob = _gh("POST", f"{_API}/repos/{owner}/{repo}/git/blobs", token,
                   json={"content": content, "encoding": "utf-8"})
        blob.raise_for_status()
        tree_entries.append({
            "path": path, "mode": "100644", "type": "blob", "sha": blob.json()["sha"],
        })
    tree = _gh("POST", f"{_API}/repos/{owner}/{repo}/git/trees", token,
               json={"base_tree": base_tree, "tree": tree_entries})
    tree.raise_for_status()

    # 3. Create the commit.
    new_commit = _gh("POST", f"{_API}/repos/{owner}/{repo}/git/commits", token,
                     json={"message": title, "tree": tree.json()["sha"], "parents": [base_sha]})
    new_commit.raise_for_status()
    new_sha = new_commit.json()["sha"]

    # 4. Create or force-update the branch ref.
    create_ref = _gh("POST", f"{_API}/repos/{owner}/{repo}/git/refs", token,
                      json={"ref": f"refs/heads/{branch}", "sha": new_sha})
    if create_ref.status_code == 422:  # already exists → fast-forward/force
        upd = _gh("PATCH", f"{_API}/repos/{owner}/{repo}/git/refs/heads/{branch}", token,
                  json={"sha": new_sha, "force": True})
        upd.raise_for_status()
    else:
        create_ref.raise_for_status()

    # 5. Open the PR (reuse an existing one for this branch).
    pr = _gh("POST", f"{_API}/repos/{owner}/{repo}/pulls", token,
             json={"title": title, "head": branch, "base": base, "body": body})
    if pr.status_code in (200, 201):
        return pr.json().get("html_url")
    if pr.status_code == 422:
        # A PR for this head already exists; the ref update above refreshed it.
        existing = _gh("GET", f"{_API}/repos/{owner}/{repo}/pulls",
                       token, json=None)
        if existing.ok:
            for p in existing.json():
                if p.get("head", {}).get("ref") == branch:
                    return p.get("html_url")
        return f"(updated existing PR for {branch})"
    pr.raise_for_status()
    return None
