#!/usr/bin/env python3
"""update_free_models.py — Scrape provider sources, diff against the sidecar,
and write add/remove updates back into llmproxy/providers.json.

Usage
-----
  python scripts/update_free_models.py --dry-run
  python scripts/update_free_models.py --provider google
  python scripts/update_free_models.py --source openrouter,docs
  python scripts/update_free_models.py --regen-config-only
  python scripts/update_free_models.py --config ~/.config/llmproxy/config.json

Behavior
--------
* All sources run independently. A source that fails (network error, parse
  error, rate-limit) emits no evidence — it never causes a removal.
* Aggregation: a model is added to believed_free if any high-confidence
  source says is_free=True and no high-confidence source contradicts. A
  model is removed if any high-confidence source says is_free=False, or if
  a successful /v1/models fetch for that provider is missing it AND the
  provider's API key was present (so the fetch is trusted).
* model_reasoning is preserved across runs; new models get reasoning
  inferred via llmproxy.providers.infer_reasoning_level.
* free_limits are merged: scraped limits override stored ones; existing
  limits are kept when no source reports new ones.

Manual review is expected — run with --dry-run first.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

# Allow `python scripts/update_free_models.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llmproxy.config import (  # noqa: E402
    load_config as load_user_config,
)
from llmproxy.config import (  # noqa: E402
    load_probe_state,
    save_probe_state,
)
from llmproxy.providers import (  # noqa: E402
    DATA_PATH,
    FREE_LIMIT_KEYS,
    infer_reasoning_level,
    load_data,
)
from scripts.sources import ALL_SOURCES, OPT_IN_SOURCES, Evidence  # noqa: E402
from scripts.sources.litellm_cost_map import fetch_pricing_map  # noqa: E402
from scripts.sources.probe import ProbeSource  # noqa: E402

CONFIG_EXAMPLE_PATH = REPO_ROOT / "config.example.json"

# Placeholder API keys used when regenerating config.example.json. Keeping
# the pattern matches the existing example so a `diff` post-refactor is
# byte-clean.
_PLACEHOLDER_KEYS: dict[str, str] = {
    "openai": "sk-...",
    "google": "AIza...",
    "groq": "gsk_...",
    "cerebras": "csk-...",
    "github": "ghp_...",
    "huggingface": "hf_...",
    "nvidia": "nvapi-...",
    "xai": "xai-...",
    "vercel": "vercel-...",
    "openrouter": "sk-or-...",
    "ollama": "ollama",
    "deepseek": "sk-...",
    "opencode-zen": "opencode-...",
    "together": "tgp_...",
    "fireworks": "fw_...",
}


# ---------------------------------------------------------------------------
# ANSI helpers (small subset; avoids importing from setup_wizard for layering)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _ok(s: str) -> str: return f"{_GREEN}{s}{_RESET}"
def _err(s: str) -> str: return f"{_RED}{s}{_RESET}"
def _warn(s: str) -> str: return f"{_YELLOW}{s}{_RESET}"
def _h(s: str) -> str: return f"{_BOLD}{_CYAN}{s}{_RESET}"
def _dim(s: str) -> str: return f"{_DIM}{s}{_RESET}"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    evidence: Iterable[Evidence],
    current_sidecar: dict,
    api_succeeded: set[str],
) -> dict:
    """Compute proposed sidecar updates from a flat list of Evidence records.

    Returns a dict per provider:
      {provider_key: {"add": [...], "remove": [...], "limits": {model: dict}}}

    Parameters
    ----------
    evidence
        All evidence records from all sources.
    current_sidecar
        The current providers.json contents.
    api_succeeded
        Set of provider_keys for which a /v1/models source ran cleanly. Only
        these providers can produce inference-based removals (otherwise a
        blocked host would cause silent deletions).
    """
    # provider_key -> model_id -> list[Evidence]
    by_model: dict[str, dict[str, list[Evidence]]] = defaultdict(lambda: defaultdict(list))
    # provider_key -> set[model_id] observed by any /v1/models call
    api_seen: dict[str, set[str]] = defaultdict(set)

    for ev in evidence:
        by_model[ev.provider][ev.model_id].append(ev)
        if ev.source == "api":
            api_seen[ev.provider].add(ev.model_id)

    out: dict[str, dict] = {}
    providers = current_sidecar["providers"]

    for provider_key, prov_cfg in providers.items():
        current_free: list[str] = list(prov_cfg.get("believed_free", []))
        adds: list[str] = []
        removes: list[str] = []
        limits: dict[str, dict] = {}
        capabilities: dict[str, list[str]] = {}

        # Adds — high-confidence positive, no high-confidence negative.
        candidates = set(by_model[provider_key].keys())
        for model_id in candidates:
            evs = by_model[provider_key][model_id]
            high_pos = any(e.confidence == "high" and e.is_free is True for e in evs)
            high_neg = any(e.confidence == "high" and e.is_free is False for e in evs)
            if high_pos and not high_neg and model_id not in current_free:
                adds.append(model_id)
            # Merge limits — prefer the highest-confidence non-empty record.
            for e in sorted(evs, key=lambda x: 0 if x.confidence == "high" else 1):
                if e.limits:
                    limits[model_id] = _normalize_limits(e.limits)
                    break
            # Merge capabilities — prefer the highest-confidence record that has
            # an opinion (capabilities is not None).
            for e in sorted(evs, key=lambda x: 0 if x.confidence == "high" else 1):
                if e.capabilities is not None:
                    capabilities[model_id] = list(e.capabilities)
                    break

        # Removes — high-confidence negative OR absent from a successful
        # /v1/models fetch (only for providers where api_succeeded).
        for model_id in current_free:
            evs = by_model[provider_key].get(model_id, [])
            if any(e.confidence == "high" and e.is_free is False for e in evs):
                removes.append(model_id)
                continue
            if provider_key in api_succeeded:
                # We have a trusted source-of-truth for what exists. If the
                # qualified model id is absent, signal removal.
                if model_id not in api_seen[provider_key]:
                    removes.append(model_id)

        out[provider_key] = {
            "add": sorted(set(adds)),
            "remove": sorted(set(removes)),
            "limits": limits,
            "capabilities": capabilities,
        }
    return out


def _normalize_limits(raw: dict) -> dict:
    """Force the 4-key shape with int|None values."""
    out = {}
    for k in FREE_LIMIT_KEYS:
        v = raw.get(k)
        if isinstance(v, (int, float)) and v >= 0:
            out[k] = int(v)
        else:
            out[k] = None
    return out


# ---------------------------------------------------------------------------
# Apply updates to the sidecar dict
# ---------------------------------------------------------------------------

def apply_updates(sidecar: dict, updates: dict) -> bool:
    """Apply updates in-place. Returns True if anything changed."""
    changed = False
    for provider_key, change in updates.items():
        prov = sidecar["providers"].get(provider_key)
        if prov is None:
            continue
        bf: list[str] = list(prov.get("believed_free", []))
        for mid in change["add"]:
            if mid not in bf:
                bf.append(mid)
                changed = True
                # Set a reasoning level if absent.
                mr = prov.setdefault("model_reasoning", {})
                if mid not in mr:
                    # Strip "provider/" prefix before inferring on the bare model id.
                    bare = mid.split("/", 1)[1] if "/" in mid else mid
                    mr[mid] = infer_reasoning_level(bare)
        for mid in change["remove"]:
            if mid in bf:
                bf.remove(mid)
                changed = True
        if bf != prov.get("believed_free", []):
            prov["believed_free"] = sorted(bf, key=str.lower) if change["add"] or change["remove"] else bf

        # Limits — only update if the scraped value differs.
        fl: dict = prov.setdefault("free_limits", {})
        for mid, new in change["limits"].items():
            if mid not in bf:
                continue  # don't store limits for models that aren't free
            old = fl.get(mid)
            if old != new:
                fl[mid] = new
                changed = True

        # Drop limits for models we just removed.
        for mid in change["remove"]:
            if mid in fl:
                del fl[mid]
                changed = True

        # Capabilities — only stored for free models (parallel to free_limits).
        # Avoid creating the key unless we actually store something, so a
        # no-op update never mutates the sidecar.
        mc: dict | None = prov.get("model_capabilities")
        for mid, caps in change.get("capabilities", {}).items():
            if mid not in bf:
                continue
            if caps:
                if mc is None:
                    mc = prov.setdefault("model_capabilities", {})
                if mc.get(mid) != caps:
                    mc[mid] = caps
                    changed = True
            elif mc is not None and mid in mc:
                del mc[mid]
                changed = True
        # Drop capabilities for models we just removed from the free set.
        for mid in change["remove"]:
            if mc is not None and mid in mc:
                del mc[mid]
                changed = True
    return changed


def _refresh_pricing(sidecar: dict) -> bool:
    """Fetch the litellm per-token pricing snapshot into sidecar['pricing'].

    Returns True if the block changed. Network/parse failures are non-fatal —
    a stale pricing block is better than aborting the whole run.
    """
    try:
        pricing = fetch_pricing_map()
    except Exception as exc:  # noqa: BLE001
        print(_warn(f"  pricing snapshot refresh failed: {exc}"))
        return False
    if not pricing:
        return False
    if sidecar.get("pricing") == pricing:
        return False
    sidecar["pricing"] = dict(sorted(pricing.items()))
    print(_ok(f"  pricing snapshot: {len(pricing)} model(s)"))
    return True


# ---------------------------------------------------------------------------
# Diff printer
# ---------------------------------------------------------------------------

def print_diff(updates: dict, source_status: dict[str, bool]) -> None:
    print(_h("\n=== Source status ==="))
    for s, ok in source_status.items():
        marker = _ok("OK ") if ok else _err("FAIL")
        print(f"  [{marker}] {s}")
    if not all(source_status.values()):
        print(_warn("  ⚠  Some sources failed. Removals from failed sources have been suppressed."))

    print(_h("\n=== Proposed changes ==="))
    any_change = False
    for provider_key, change in sorted(updates.items()):
        if not (change["add"] or change["remove"] or change["limits"]):
            continue
        any_change = True
        print(f"\n  {_h(provider_key)}:")
        for mid in change["add"]:
            print(_ok(f"    + {mid}"))
        for mid in change["remove"]:
            print(_err(f"    - {mid}"))
        for mid, lim in change["limits"].items():
            print(_dim(f"    ~ {mid}  limits: {lim}"))
    if not any_change:
        print(_dim("  (no changes)"))


# ---------------------------------------------------------------------------
# config.example.json regenerator
# ---------------------------------------------------------------------------

def regenerate_config_example(sidecar: dict, server_block: dict | None = None,
                              top_note: str | None = None) -> dict:
    """Build the config.example.json contents from the sidecar.

    Provider order, model order, and limit ordering all follow the sidecar's
    provider_order. This is deterministic so the regen is round-tripable.
    """
    order = sidecar.get("provider_order") or list(sidecar["providers"].keys())

    providers_block: dict = {
        # openai is in config.example.json but not in our provider_order;
        # carry it forward as a static example provider with no free entries.
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": _PLACEHOLDER_KEYS["openai"],
            "model_filter": None,
        }
    }
    # Insert each sidecar provider with its placeholder key.
    for key in order:
        prov = sidecar["providers"][key]
        block: dict = {
            "base_url": prov["base_url"],
            "api_key": _PLACEHOLDER_KEYS.get(key, "..."),
            # Providers that cannot serve a discovery endpoint advertise models
            # from an example_model_filter; everyone else allows all (null).
            "model_filter": list(prov["example_model_filter"])
            if prov.get("example_model_filter") else None,
        }
        # Carry forward optional model-discovery overrides (non-standard /models
        # path, id field, or task filter) so the example documents the working
        # config for providers like GitHub Models and Cloudflare Workers AI.
        for field in ("models_url", "models_id_field", "models_keep_task", "protocol"):
            if prov.get(field):
                block[field] = prov[field]
        # A per-provider note (e.g. auth/credential gotchas) surfaces in the
        # example as a leading "_note" key, mirroring the top-level convention.
        if prov.get("_note"):
            block = {"_note": prov["_note"], **block}
        providers_block[key] = block
    # ollama is local-only — also a static example.
    providers_block["ollama"] = {
        "base_url": "http://localhost:11434/v1",
        "api_key": _PLACEHOLDER_KEYS["ollama"],
        "model_filter": None,
    }

    believed_free: list[str] = []
    model_reasoning: dict[str, str] = {}
    model_capabilities: dict[str, list[str]] = {}
    free_limits: dict[str, dict] = {}
    for key in order:
        prov = sidecar["providers"][key]
        believed_free.extend(prov.get("believed_free", []))
        model_reasoning.update(prov.get("model_reasoning", {}))
        model_capabilities.update(prov.get("model_capabilities", {}))
        free_limits.update(prov.get("free_limits", {}))

    note = top_note or (
        "All providers must expose an OpenAI-compatible API "
        "(/models, /chat/completions, Bearer auth). "
        "The provider name 'llmproxy' is reserved and must not be used. "
        "Optional per-provider flag: set \"expose_to_virtual_models\": false to "
        "exclude a provider from all virtual endpoints (llmproxy__free, "
        "llmproxy__deep, llmproxy__tools, etc.) while still allowing direct "
        "calls to its models."
    )

    free_limits_with_note: dict = {
        "_note": (
            "Rate limits for capacity-aware load balancing on llmproxy/free "
            "and llmproxy/*/free endpoints. Tracked per-worker-process; "
            "resets on restart. Both request limits (rpm/rpd) and token limits "
            "(tpm/tpd) are enforced. Check provider docs — limits change "
            "frequently."
        ),
        **free_limits,
    }

    return {
        "_note": note,
        "providers": providers_block,
        "believed_free": believed_free,
        "model_reasoning": model_reasoning,
        "model_capabilities": model_capabilities,
        "free_limits": free_limits_with_note,

        # Opt-in maintenance flags (all default false). See the README:
        # probe_cost / autoremove_believed_free → "Verifying free models are
        # actually free"; update_believed_free_on_startup → "Running the updater
        # on startup"; pr_providers_list → "Proposing providers.json changes as a
        # PR from a running deployment". probe_frequency_days throttles the probe
        # to at most once every N days (0 = every run); the last-run timestamp is
        # cached in probe_state.json next to config.json.
        "probe_cost": False,
        "autoremove_believed_free": False,
        "probe_frequency_days": 0,
        "update_believed_free_on_startup": False,
        "pr_providers_list": False,

        "server": server_block or {
            "host": "0.0.0.0",
            "port": 8080,
            "log_level": "INFO",
            "request_timeout": 120,
            "stream_timeout": 300,
        },
    }


def write_config_example(out: Path = CONFIG_EXAMPLE_PATH) -> None:
    sidecar = load_data()
    cfg = regenerate_config_example(sidecar)
    out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def sidecar_fallback_paths(config_path: str | None) -> tuple[Path, Path] | None:
    """Return (providers.json, config.example.json) paths in the user-config dir.

    These mirror the computed artifacts when the bundled, image-layer copies are
    read-only — a writable location (e.g. the container's /config bind mount) a
    read-only deployment can still review and open a providers PR from. Returns
    None when no user config path is known (nowhere writable to fall back to).
    """
    if not config_path:
        return None
    cfg_dir = Path(config_path).parent
    return cfg_dir / "providers.json", cfg_dir / "config.example.json"


def _write_sidecar_fallback(sidecar: dict, config_path: str | None) -> None:
    """Mirror the computed sidecar + config.example to the user-config dir."""
    paths = sidecar_fallback_paths(config_path)
    if paths is None:
        return
    providers_path, example_path = paths
    try:
        providers_path.parent.mkdir(parents=True, exist_ok=True)
        providers_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        example_path.write_text(
            json.dumps(regenerate_config_example(sidecar), indent=2) + "\n", encoding="utf-8"
        )
        print(_warn(
            f"Mirrored computed providers.json + config.example.json to "
            f"{providers_path.parent} (review there / open a providers PR)."
        ))
    except OSError as e:
        print(_warn(f"Could not mirror computed artifacts to the config dir: {e}"))


# ---------------------------------------------------------------------------
# User config.json sync  (--config)
# ---------------------------------------------------------------------------

def _provider_of(model_id: str) -> str:
    """Provider key for a qualified model id ('github/openai/gpt-4o' -> 'github')."""
    return model_id.split("/", 1)[0]


def reconcile_user_config(sidecar: dict, user_cfg: dict) -> dict:
    """Sync a user config's free-tier sections from *sidecar*, in place.

    Scope is limited to providers that appear in both the user config's
    ``providers`` block and the sidecar. For those providers:

      * ``believed_free`` and ``free_limits`` are reconciled — newly-free models
        are added and models that are no longer free are removed.
      * ``model_reasoning`` and ``model_capabilities`` are add-only; existing
        tags are never pruned (a model can carry a reasoning level or capability
        tags without being free, and users may hand-tag their own models).

    Entries for any other provider (custom providers, or sidecar providers the
    user has not configured) are left untouched, as are non-model keys in
    ``free_limits`` (e.g. the ``_note`` string).

    Returns a changes summary:
      {"believed_free": {"add": [...], "remove": [...]},
       "free_limits":   {"set": [...], "remove": [...]},
       "model_reasoning": {"add": [...]},
       "model_capabilities": {"add": [...]}}
    """
    providers: dict = sidecar.get("providers", {})
    configured = set(user_cfg.get("providers", {})) & set(providers)

    # Sidecar aggregates restricted to configured providers.
    sc_believed: set[str] = set()
    sc_limits: dict[str, dict] = {}
    sc_reasoning: dict[str, str] = {}
    sc_capabilities: dict[str, list[str]] = {}
    for pkey in configured:
        prov = providers[pkey]
        sc_believed.update(prov.get("believed_free", []))
        sc_limits.update(prov.get("free_limits", {}))
        sc_reasoning.update(prov.get("model_reasoning", {}))
        sc_capabilities.update(prov.get("model_capabilities", {}))

    changes = {
        "believed_free": {"add": [], "remove": []},
        "free_limits": {"set": [], "remove": []},
        "model_reasoning": {"add": []},
        "model_capabilities": {"add": []},
    }

    # ── believed_free ──────────────────────────────────────────────────────
    existing_bf = user_cfg.get("believed_free")
    if not isinstance(existing_bf, list):
        existing_bf = []
    new_bf: list[str] = []
    for mid in existing_bf:
        if not isinstance(mid, str):
            new_bf.append(mid)  # leave anything unexpected alone
            continue
        if _provider_of(mid) in configured and mid not in sc_believed:
            changes["believed_free"]["remove"].append(mid)  # no longer free
        else:
            new_bf.append(mid)  # untouched provider, or still free
    present = set(new_bf)
    for mid in sorted(sc_believed):
        if mid not in present:
            new_bf.append(mid)
            changes["believed_free"]["add"].append(mid)
    user_cfg["believed_free"] = new_bf
    new_bf_set = set(new_bf)

    # ── free_limits ────────────────────────────────────────────────────────
    existing_fl = user_cfg.get("free_limits")
    if not isinstance(existing_fl, dict):
        existing_fl = {}
    for key in list(existing_fl.keys()):
        # Preserve non-model keys (e.g. "_note") and unconfigured providers.
        if "/" not in key or _provider_of(key) not in configured:
            continue
        if key not in new_bf_set:
            del existing_fl[key]
            changes["free_limits"]["remove"].append(key)
    for mid, lim in sc_limits.items():
        if mid in new_bf_set and existing_fl.get(mid) != lim:
            existing_fl[mid] = copy.deepcopy(lim)
            changes["free_limits"]["set"].append(mid)
    user_cfg["free_limits"] = existing_fl

    # ── model_reasoning (add-only) ─────────────────────────────────────────
    existing_mr = user_cfg.get("model_reasoning")
    if not isinstance(existing_mr, dict):
        existing_mr = {}
    for mid, level in sc_reasoning.items():
        if mid not in existing_mr:
            existing_mr[mid] = level
            changes["model_reasoning"]["add"].append(mid)
    user_cfg["model_reasoning"] = existing_mr

    # ── model_capabilities (add-only) ──────────────────────────────────────
    existing_mc = user_cfg.get("model_capabilities")
    if not isinstance(existing_mc, dict):
        existing_mc = {}
    for mid, caps in sc_capabilities.items():
        if mid not in existing_mc:
            existing_mc[mid] = list(caps)
            changes["model_capabilities"]["add"].append(mid)
    user_cfg["model_capabilities"] = existing_mc

    return changes


def _config_changed(changes: dict) -> bool:
    return any(
        bucket.get(k) for bucket in changes.values() for k in bucket
    )


def print_config_diff(path: Path, changes: dict) -> None:
    print(_h(f"\n=== Proposed config changes ({path}) ==="))
    if not _config_changed(changes):
        print(_dim("  (no changes — config already in sync)"))
        return
    labels = {
        "believed_free": "believed_free",
        "free_limits": "free_limits",
        "model_reasoning": "model_reasoning",
    }
    for section in ("believed_free", "free_limits", "model_reasoning"):
        bucket = changes[section]
        if not any(bucket.values()):
            continue
        print(f"\n  {_h(labels[section])}:")
        for mid in bucket.get("add", []):
            print(_ok(f"    + {mid}"))
        for mid in bucket.get("set", []):
            print(_dim(f"    ~ {mid}"))
        for mid in bucket.get("remove", []):
            print(_err(f"    - {mid}"))


def _sync_user_config(sidecar: dict, config_path: str, *, dry_run: bool) -> int:
    """Reconcile the user config at *config_path* from *sidecar*. Returns an
    exit code (0 ok, non-zero on a read error)."""
    path = Path(config_path).expanduser()
    try:
        user_cfg = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(_err(f"\n--config: file not found: {path}"))
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        print(_err(f"\n--config: could not read {path}: {exc}"))
        return 2
    if not isinstance(user_cfg, dict):
        print(_err(f"\n--config: {path} is not a JSON object"))
        return 2

    changes = reconcile_user_config(sidecar, user_cfg)
    print_config_diff(path, changes)

    if not _config_changed(changes):
        return 0
    if dry_run:
        print(_dim("\n(dry run — config not written)"))
        return 0
    path.write_text(json.dumps(user_cfg, indent=2), encoding="utf-8")
    print(_ok(f"Synced free-tier sections into {path}"))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_source(
    source_name: str,
    *,
    config_path: str | None = None,
    probe_max: int | None = None,
    probe_provider: str | None = None,
) -> tuple[str, bool, list[Evidence], str | None]:
    try:
        if source_name == "probe":
            src = ProbeSource(config_path=config_path, max_models=probe_max,
                              provider_filter=probe_provider)
        else:
            cls = ALL_SOURCES[source_name]
            src = cls()
        evidence = src.fetch()
    except Exception as exc:  # noqa: BLE001
        return source_name, False, [], str(exc)
    return source_name, True, evidence, None


def _probe_due(last_probe_at: str | None, frequency_days, now: datetime | None = None
               ) -> tuple[bool, float | None]:
    """Decide whether the cost probe is due to run.

    Returns ``(due, days_since_last)``. The probe is due when:
      * ``frequency_days`` is missing or <= 0 (no throttle — run every time), or
      * there is no usable ``last_probe_at`` timestamp, or
      * at least ``frequency_days`` have elapsed since the last probe.

    ``days_since_last`` is ``None`` when there is no usable prior timestamp.
    """
    try:
        freq = float(frequency_days or 0)
    except (TypeError, ValueError):
        freq = 0.0
    if freq <= 0:
        return True, None
    if not last_probe_at:
        return True, None
    try:
        last = datetime.fromisoformat(last_probe_at)
    except (TypeError, ValueError):
        return True, None
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    now = now or datetime.now(UTC)
    days_since = (now - last).total_seconds() / 86400.0
    return days_since >= freq, days_since


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_sources = [s for s in ALL_SOURCES if s not in OPT_IN_SOURCES]
    ap.add_argument("--dry-run", action="store_true",
                    help="Print proposed changes without writing anything.")
    ap.add_argument("--provider",
                    help="Limit updates to a single provider (e.g. 'google').")
    ap.add_argument("--source", default=",".join(default_sources),
                    help="Comma-separated source names (default: all except opt-in probes).")
    ap.add_argument("--probe", action="store_true",
                    help="Actively probe believed_free models for cost (sends real "
                         "requests; requires configured API keys). Also enabled by "
                         "setting probe_cost: true in config.json.")
    ap.add_argument("--probe-max", type=int, metavar="N",
                    help="Probe at most N models (bounds spend).")
    ap.add_argument("--probe-provider", metavar="NAME",
                    help="Only probe models from this provider.")
    ap.add_argument("--ignore-throttle", action="store_true",
                    help="Probe even if probe_frequency_days says it is too soon "
                         "since the last probe (bypasses the throttle).")
    ap.add_argument("--regen-config-only", action="store_true",
                    help="Skip scraping; regenerate config.example.json from the current sidecar.")
    ap.add_argument("--config", metavar="PATH",
                    help="Also sync a user config.json's believed_free / free_limits "
                         "(add new + remove no-longer-free) and model_reasoning (add-only) "
                         "from the updated sidecar. Limited to providers configured in that file.")
    args = ap.parse_args(argv)

    # Read opt-in flags from the user config (probe_cost / autoremove_believed_free).
    try:
        user_cfg = load_user_config(args.config, force_reload=True)
    except Exception:  # noqa: BLE001 — a missing/broken config must not break scraping
        user_cfg = {}
    probe_cost = bool(user_cfg.get("probe_cost", False)) or args.probe
    autoremove = bool(user_cfg.get("autoremove_believed_free", False))

    # Throttle the probe to at most once every probe_frequency_days. The last-run
    # timestamp lives in a sibling cache file (probe_state.json), not config.json.
    # --ignore-throttle bypasses this; frequency 0 means "probe every time".
    if probe_cost and not args.ignore_throttle:
        state = load_probe_state(args.config)
        due, days_since = _probe_due(
            state.get("last_probe_at"), user_cfg.get("probe_frequency_days", 0)
        )
        if not due:
            freq = user_cfg.get("probe_frequency_days", 0)
            since = f"{days_since:.1f}" if days_since is not None else "?"
            print(_warn(
                f"  ⚠  probe throttled — last run was {since} day(s) ago, "
                f"probe_frequency_days={freq}. Use --ignore-throttle to override."
            ))
            probe_cost = False

    if args.regen_config_only:
        write_config_example()
        print(_ok(f"Regenerated {CONFIG_EXAMPLE_PATH}"))
        if args.config:
            return _sync_user_config(load_data(), args.config, dry_run=args.dry_run)
        return 0

    requested = [s.strip() for s in args.source.split(",") if s.strip()]
    if probe_cost and "probe" not in requested:
        requested.append("probe")
    unknown = [s for s in requested if s not in ALL_SOURCES]
    if unknown:
        print(_err(f"Unknown source(s): {unknown}. Known: {sorted(ALL_SOURCES.keys())}"))
        return 2

    print(_h(f"\nFetching evidence from sources: {requested}"))
    if "probe" in requested:
        print(_warn("  ⚠  probe enabled — sending real requests to believed_free models "
                    "(uses configured API keys / quota)."))
    all_evidence: list[Evidence] = []
    source_status: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=min(5, len(requested))) as ex:
        futures = {
            ex.submit(_run_source, s, config_path=args.config,
                      probe_max=args.probe_max, probe_provider=args.probe_provider): s
            for s in requested
        }
        for fut in as_completed(futures):
            name, ok, evs, err = fut.result()
            source_status[name] = ok
            if ok:
                print(_dim(f"  {name}: {len(evs)} evidence record(s)"))
                all_evidence.extend(evs)
            else:
                print(_err(f"  {name}: FAILED — {err}"))

    # Probe-confirmed paid models. When autoremove_believed_free is off (the
    # default), we report these but do NOT remove them from believed_free.
    probe_paid = {
        (ev.provider, ev.model_id)
        for ev in all_evidence
        if ev.source == "probe" and ev.is_free is False
    }
    if probe_paid:
        print(_h("\n=== Probe flagged believed_free models reporting a cost ==="))
        for _provider_name, model_id in sorted(probe_paid):
            print(f"  {_warn('⚠')} {model_id}")
        if autoremove:
            print(_warn("  autoremove_believed_free=true → these will be removed."))
        else:
            print(_dim("  autoremove_believed_free=false → flagged only (not removed). "
                       "Set it true in config.json to auto-remove."))
        if not autoremove:
            all_evidence = [
                ev for ev in all_evidence
                if not (ev.source == "probe" and ev.is_free is False)
            ]

    # If only "api" succeeded for a provider, we trust /models presence as
    # ground truth for that provider.
    api_succeeded = {
        ev.provider for ev in all_evidence if ev.source == "api"
    }

    sidecar = load_data()
    updates = aggregate(all_evidence, sidecar, api_succeeded)

    # Provider filter
    if args.provider:
        updates = {k: v for k, v in updates.items() if k == args.provider}
        if not updates:
            print(_err(f"Unknown provider: {args.provider}"))
            return 2

    print_diff(updates, source_status)

    # Refresh the per-token pricing snapshot (used by the proxy to cost tokens
    # offline) whenever the litellm cost map ran cleanly.
    pricing_changed = False
    if source_status.get("litellm_cost_map") and not args.provider:
        pricing_changed = _refresh_pricing(sidecar)

    if args.dry_run:
        # Reflect the would-be sidecar state in the user-config diff without
        # writing anything to the sidecar.
        if args.config:
            target = copy.deepcopy(sidecar)
            apply_updates(target, updates)
            _sync_user_config(target, args.config, dry_run=True)
        print(_dim("\n(dry run — no files written)"))
        return 0

    changed = apply_updates(sidecar, updates) or pricing_changed
    if changed:
        try:
            DATA_PATH.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
            print(_ok(f"\nUpdated {DATA_PATH}"))
            write_config_example()
            print(_ok(f"Regenerated {CONFIG_EXAMPLE_PATH}"))
        except OSError as e:
            # In a container the bundled sidecar (and config.example.json) live on
            # a read-only image layer, so the scrape result can't be persisted
            # there — that's expected and ephemeral. Don't abort: the user-config
            # sync and the probe-throttle timestamp below still need to run.
            print(_warn(
                f"\nCould not persist {DATA_PATH} ({e}); continuing without "
                "writing the sidecar (expected on a read-only image)."
            ))
            # Mirror the computed artifacts to the (writable) user-config dir so a
            # read-only deployment can still review them and open a providers PR
            # (pr_providers_list) from the running container.
            _write_sidecar_fallback(sidecar, args.config)
    else:
        print(_dim("\nNo changes to apply."))

    # Record when the probe last ran so probe_frequency_days can throttle the
    # next invocation. Only on a real run where the probe was actually included.
    # Done after (and independent of) the sidecar write above so a read-only
    # providers.json can't prevent the throttle from advancing.
    if "probe" in requested:
        save_probe_state(
            {"last_probe_at": datetime.now(UTC).isoformat()}, args.config
        )

    # Sync the user config even when the sidecar was unchanged — a stale config
    # should still be reconciled against the current sidecar.
    if args.config:
        return _sync_user_config(sidecar, args.config, dry_run=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
