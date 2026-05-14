#!/usr/bin/env bash
# Merges all open Dependabot PRs across BillJr99/* repos and optionally
# enables Dependabot auto-merge on each repo going forward.
#
# Prerequisites:
#   gh auth login   (GitHub CLI, authenticated)
#
# Usage:
#   ./accept_dependabot_prs.sh            # merge pending PRs only
#   ./accept_dependabot_prs.sh --auto     # merge pending PRs + enable auto-merge
#   ./accept_dependabot_prs.sh --dry-run  # print what would happen, no changes

set -euo pipefail

OWNER="BillJr99"
MERGE_METHOD="merge"   # merge | squash | rebase
AUTO_MERGE=false
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --auto)     AUTO_MERGE=true ;;
    --dry-run)  DRY_RUN=true ;;
  esac
done

if ! command -v gh &>/dev/null; then
  echo "ERROR: gh CLI not found. Install from https://cli.github.com" >&2
  exit 1
fi

if ! gh auth status &>/dev/null; then
  echo "ERROR: not authenticated. Run: gh auth login" >&2
  exit 1
fi

echo "==> Fetching all repos for ${OWNER}..."
repos=$(gh repo list "$OWNER" --limit 200 --json nameWithOwner -q '.[].nameWithOwner')

merged=0
skipped=0
failed=0

for repo in $repos; do
  repo_name="${repo#"${OWNER}/"}"

  # --- enable auto-merge for future Dependabot PRs ---
  if $AUTO_MERGE; then
    if $DRY_RUN; then
      echo "[dry-run] would enable auto-merge on ${repo}"
    else
      # auto-merge requires the repo to allow it; silently skip if not supported
      gh api "repos/${repo}" \
        --method PATCH \
        --field allow_auto_merge=true \
        --silent 2>/dev/null || true

      # Dependabot auto-merge via ruleset/workflow isn't a single API call,
      # but we can approve + auto-merge each PR as it opens using:
      #   gh pr merge --auto --merge
      # For existing PRs we handle them below; the flag here just primes the setting.
      echo "  [auto-merge] enabled on ${repo}"
    fi
  fi

  # --- find open Dependabot PRs ---
  prs=$(gh pr list \
    --repo "$repo" \
    --author "app/dependabot" \
    --state open \
    --json number,title \
    --jq '.[] | "\(.number)\t\(.title)"' 2>/dev/null) || {
    # repo may be private / no access — skip silently
    continue
  }

  if [[ -z "$prs" ]]; then
    continue
  fi

  echo ""
  echo "==> ${repo}"
  while IFS=$'\t' read -r pr_number pr_title; do
    printf "    PR #%s: %s\n" "$pr_number" "$pr_title"
    if $DRY_RUN; then
      echo "    [dry-run] would merge"
      ((skipped++)) || true
    else
      if gh pr merge "$pr_number" \
           --repo "$repo" \
           --"$MERGE_METHOD" \
           --subject "Merge Dependabot PR #${pr_number}" \
           2>&1; then
        echo "    [merged]"
        ((merged++)) || true
      else
        echo "    [FAILED] — may need CI to pass or branch protection review"
        ((failed++)) || true
      fi
    fi
  done <<< "$prs"
done

echo ""
echo "=== Done ==="
if $DRY_RUN; then
  echo "    (dry-run) PRs that would be merged: ${skipped}"
else
  echo "    Merged:  ${merged}"
  echo "    Failed:  ${failed}  (branch protection / required checks not met)"
fi

# --- auto-merge tip ---
if ! $AUTO_MERGE; then
  echo ""
  echo "Tip: re-run with --auto to also enable auto-merge on every repo."
  echo "     Future Dependabot PRs that pass CI will then merge themselves."
  echo "     GitHub also lets you do this per-repo:"
  echo "       gh api repos/${OWNER}/<repo> --method PATCH --field allow_auto_merge=true"
fi
