"""github_pr.py — open a pull request from file contents via the GitHub API.

Used by the optional ``pr_providers_list`` feature: when the startup updater
rewrites ``providers.json`` locally, the server can push those changes to a
branch and open a PR against the default branch — entirely through the GitHub
REST/Git-Data API, so it never mutates the local git checkout (or even requires
one). It only needs a token, an ``owner/repo`` slug, and the file contents.
"""

from __future__ import annotations

import hashlib
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


def _git_blob_sha(content: str) -> str:
    """Git's object id for a blob holding *content* (sha1 of 'blob <len>\\0<bytes>').

    Matches the sha GitHub stores in a tree entry, so identical content yields an
    identical sha — letting us tell whether a file actually differs from the base
    branch without a per-file content fetch (and without the contents API's inline
    size limit).
    """
    data = content.encode("utf-8")
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 — git uses sha1, not security-sensitive


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


def _base_tree_blobs(owner: str, repo: str, tree_sha: str, token: str) -> dict[str, str] | None:
    """Map repo-relative path -> blob sha for every file in *tree_sha* (recursive).

    Returns None if the tree can't be read (so callers fall back to opening the
    PR rather than wrongly assuming "no changes"). A truncated tree is also
    treated as unknown, since a missing path would otherwise read as a change.
    """
    resp = _gh("GET", f"{_API}/repos/{owner}/{repo}/git/trees/{tree_sha}?recursive=1", token)
    if not resp.ok:
        return None
    data = resp.json()
    if data.get("truncated"):
        return None
    return {
        entry["path"]: entry["sha"]
        for entry in data.get("tree", [])
        if entry.get("type") == "blob"
    }


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
    decisive_paths: list[str] | None = None,
) -> str | None:
    """Commit *files* onto *branch* and open a PR into *base*.

    *files* maps repo-relative paths to their new text contents. Returns the PR
    URL (or the branch's existing-PR note) on success, or raises on a hard API
    error. The branch ref is force-updated if it already exists, and an existing
    open PR for the branch is reused (so repeated runs just refresh it).

    *decisive_paths*, when given, restricts the "did anything change?" check to
    those paths. Other files (e.g. the derived ``config.example.json``) are still
    committed for consistency but never, on their own, cause a PR to be opened or
    refreshed — so a regenerated-but-equivalent derived file can't churn a PR on
    every run. Defaults to all files.
    """
    # 1. Resolve the base branch head + its tree.
    ref = _gh("GET", f"{_API}/repos/{owner}/{repo}/git/ref/heads/{base}", token)
    ref.raise_for_status()
    base_sha = ref.json()["object"]["sha"]
    commit = _gh("GET", f"{_API}/repos/{owner}/{repo}/git/commits/{base_sha}", token)
    commit.raise_for_status()
    base_tree = commit.json()["tree"]["sha"]

    # 1b. Skip entirely if the decisive files already match the base branch. The
    # caller only knows the local sidecar changed within the running container —
    # not whether it differs from `base` (a previous PR may already have merged
    # the same content). Comparing git blob shas against the base tree avoids
    # opening (or force-refreshing) a PR whose meaningful diff is empty. Only the
    # decisive paths count here, so a derived file that merely regenerated to an
    # equivalent (or version-skewed) form never triggers a PR by itself.
    decisive = [p for p in (decisive_paths or list(files)) if p in files]
    base_blobs = _base_tree_blobs(owner, repo, base_tree, token)
    if base_blobs is not None and all(
        base_blobs.get(path) == _git_blob_sha(files[path])
        for path in decisive
    ):
        logger.info(
            "[providers-pr] no changes vs %s — skipping PR (decisive files already current).", base
        )
        return None

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
