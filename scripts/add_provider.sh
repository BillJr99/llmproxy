#!/usr/bin/env bash
#
# add_provider.sh — add a provider to an llmproxy config without running the
# full `llmproxy --setup` wizard.
#
# Prompts for the provider, its API key and the config file path, then merges
# providers.<key> into config.json — the same file and the same shape the wizard
# writes. If the chosen provider is missing from this checkout's
# llmproxy/providers.json, it also offers to add it there (and to provider_order)
# and to regenerate config.example.json, so the script works on older checkouts.
#
# Requires: python3 (already required to run llmproxy) and curl. No jq.
#
# Usage: scripts/add_provider.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SIDECAR="$REPO_ROOT/llmproxy/providers.json"
DEFAULT_CONFIG="${LLMPROXY_CONFIG:-$HOME/.config/llmproxy/config.json}"

command -v python3 >/dev/null 2>&1 || { echo "error: python3 is required but not on PATH." >&2; exit 1; }
command -v curl    >/dev/null 2>&1 || { echo "error: curl is required but not on PATH." >&2; exit 1; }

TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT
HELPER="$TMPDIR_RUN/helper.py"

cat > "$HELPER" <<'PY_EOF'
"""Helper for add_provider.sh — all JSON reading and writing lives here so the
shell side only has to handle prompting."""
import json
import os
import sys
from collections import OrderedDict

# Providers this script knows about even when the checkout's providers.json
# predates them. Merged under whatever the sidecar already defines, so on a
# current checkout the sidecar wins and this table is inert.
BUILTIN = OrderedDict([
    ("xkiro", {
        "display": "xKiro (multi-vendor gateway)",
        "base_url": "https://api.xkiro.com/v1",
        "key_required": True,
        "key_hint": "Create an API key at xkiro.com — keys look like sk-xt-... (docs: docs.xkiro.com)",
    }),
    ("teamorouter", {
        "display": "TeamoRouter (LLM routing gateway)",
        "base_url": "https://api.teamorouter.com/v1",
        "key_required": True,
        "key_hint": "Create an API key at teamorouter.com — keys look like sk-teamo-... (docs: teamorouter.com/docs/api-integration)",
    }),
    ("gmi", {
        "display": "GMI Cloud",
        "base_url": "https://api.gmi-serving.com/v1",
        "key_required": True,
        "key_hint": "Create an API key at console.gmicloud.ai → Organization Settings → API Keys (docs: docs.gmicloud.ai/inference-engine). Inference is served from api.gmi-serving.com; the console host is GMI's control plane and serves no /chat/completions.",
    }),
])

# Fields the wizard copies from a template onto the config entry.
PASSTHROUGH = ("models_url", "models_id_field", "models_keep_task", "protocol", "_note")


def load_sidecar(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh, object_pairs_hook=OrderedDict)
    except Exception:
        return None


def menu(path):
    """Sidecar templates in provider_order, plus any BUILTIN entry missing from it."""
    data = load_sidecar(path)
    out = OrderedDict()
    in_file = []
    if data:
        in_file = list(data.get("providers", {}).keys())
        for key in data.get("provider_order") or in_file:
            if key in data.get("providers", {}):
                out[key] = data["providers"][key]
    for key, tpl in BUILTIN.items():
        if key not in out:
            out[key] = tpl
    return {"templates": out, "parsed": data is not None, "in_file": in_file}


def write_atomic(path, text, mode=0o600):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def main():
    cmd, args = sys.argv[1], sys.argv[2:]

    if cmd == "menu":
        info = menu(args[0])
        json.dump(info, sys.stdout, ensure_ascii=False)

    elif cmd == "field":
        # field <sidecar> <key> <field>
        tpl = menu(args[0])["templates"].get(args[1], {})
        val = tpl.get(args[2])
        if val is None:
            sys.stdout.write("")
        elif isinstance(val, bool):
            sys.stdout.write("true" if val else "false")
        elif isinstance(val, (dict, list)):
            sys.stdout.write(json.dumps(val, ensure_ascii=False))
        else:
            sys.stdout.write(str(val))

    elif cmd == "merge":
        # merge <config> <sidecar> <provider_key> <template_key> <base_url> <api_key>
        cfg_path, sidecar, pkey, tkey, base_url, api_key = args
        tpl = menu(sidecar)["templates"].get(tkey, {})

        config = OrderedDict()
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as fh:
                text = fh.read().strip()
            if text:
                try:
                    config = json.loads(text, object_pairs_hook=OrderedDict)
                except ValueError as exc:
                    sys.stderr.write(
                        "error: %s is not valid JSON (%s).\n"
                        "Refusing to overwrite it — fix or move the file and re-run.\n"
                        % (cfg_path, exc))
                    return 1
            # Keep a copy before we touch it.
            with open(cfg_path, encoding="utf-8") as fh:
                original = fh.read()
            write_atomic(cfg_path + ".bak", original)
        else:
            parent = os.path.dirname(cfg_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        if not isinstance(config, dict):
            sys.stderr.write("error: %s does not contain a JSON object.\n" % cfg_path)
            return 1
        providers = config.setdefault("providers", OrderedDict())
        if not isinstance(providers, dict):
            sys.stderr.write("error: %s has a non-object 'providers' key.\n" % cfg_path)
            return 1

        entry = OrderedDict()
        if tpl.get("_note"):
            entry["_note"] = tpl["_note"]
        entry["base_url"] = base_url
        entry["api_key"] = api_key
        # Providers with no OpenAI-shaped catalog advertise models from
        # model_filter instead; the template carries the starting list.
        entry["model_filter"] = tpl.get("example_model_filter")
        for field in PASSTHROUGH:
            if field == "_note":
                continue
            if tpl.get(field):
                value = tpl[field]
                # The URL placeholders were already resolved in base_url.
                if field == "models_url":
                    for ph in ("{account_id}", "{gateway_id}"):
                        if ph in value:
                            value = value.replace(ph, os.environ.get(
                                "SUBST_" + ph.strip("{}").upper(), ph))
                entry[field] = value

        existed = pkey in providers
        providers[pkey] = entry
        write_atomic(cfg_path, json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        sys.stdout.write("updated" if existed else "added")

    elif cmd == "patch-sidecar":
        # patch-sidecar <sidecar> <key>  — add a BUILTIN entry the checkout lacks
        sidecar, key = args
        data = load_sidecar(sidecar)
        if data is None:
            sys.stdout.write("unreadable")
            return 0
        if key in data.get("providers", {}):
            sys.stdout.write("present")
            return 0
        tpl = BUILTIN.get(key)
        if tpl is None:
            sys.stdout.write("unknown")
            return 0
        entry = OrderedDict(tpl)
        # The metadata blocks every provider entry must carry; the scraper fills them.
        entry["believed_free"] = []
        entry["model_reasoning"] = OrderedDict()
        entry["free_limits"] = OrderedDict()
        entry["fallback_models"] = []
        data["providers"][key] = entry
        data.setdefault("provider_order", []).append(key)
        write_atomic(sidecar, json.dumps(data, indent=2, ensure_ascii=False) + "\n", 0o644)
        sys.stdout.write("patched")

    elif cmd == "count-models":
        # count-models <body-file>
        with open(args[0], encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("data") or payload.get("result") or []
        else:
            rows = []
        sys.stdout.write(str(len(rows)))

    else:
        sys.stderr.write("unknown subcommand: %s\n" % cmd)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
PY_EOF

helper() { python3 "$HELPER" "$@"; }

[ -f "$SIDECAR" ] || { echo "error: llmproxy/providers.json not found at $SIDECAR." >&2; exit 1; }

MENU_JSON="$(helper menu "$SIDECAR")"
mapfile -t KEYS < <(printf '%s' "$MENU_JSON" | python3 -c '
import json, sys
for key in json.load(sys.stdin)["templates"]:
    print(key)')

echo
echo "llmproxy — add a provider"
echo "========================="
echo
i=1
for k in "${KEYS[@]}"; do
  printf '%3d) %-42s %s\n' "$i" "$(helper field "$SIDECAR" "$k" display)" \
                                "$(helper field "$SIDECAR" "$k" base_url)"
  i=$((i + 1))
done
echo

while :; do
  read -r -p "Select a provider [1-${#KEYS[@]}]: " choice
  case "$choice" in
    ''|*[!0-9]*) echo "  Enter a number." ;;
    *) if [ "$choice" -ge 1 ] && [ "$choice" -le "${#KEYS[@]}" ]; then break; fi
       echo "  Out of range." ;;
  esac
done
TPL_KEY="${KEYS[$((choice - 1))]}"

DISPLAY="$(helper field "$SIDECAR" "$TPL_KEY" display)"
BASE_URL="$(helper field "$SIDECAR" "$TPL_KEY" base_url)"
KEY_REQUIRED="$(helper field "$SIDECAR" "$TPL_KEY" key_required)"
KEY_HINT="$(helper field "$SIDECAR" "$TPL_KEY" key_hint)"

echo
echo "Selected: $DISPLAY"

# The provider key is the prefix in model IDs (e.g. xkiro/openai/gpt-5.6-sol).
read -r -p "Provider name (used as prefix in model IDs) [$TPL_KEY]: " PROVIDER_KEY
PROVIDER_KEY="${PROVIDER_KEY:-$TPL_KEY}"
if [ "$PROVIDER_KEY" = "llmproxy" ]; then
  echo "error: 'llmproxy' is reserved by the proxy itself." >&2
  exit 1
fi

# ── {account_id} / {gateway_id} substitution, as the wizard does ─────────────
if [ "$(helper field "$SIDECAR" "$TPL_KEY" account_id_required)" = "true" ]; then
  label="$(helper field "$SIDECAR" "$TPL_KEY" account_id_label)"
  hint="$(helper field "$SIDECAR" "$TPL_KEY" account_id_hint)"
  [ -n "$hint" ] && echo "  $hint"
  read -r -p "${label:-Account ID}: " acct
  [ -n "$acct" ] || { echo "error: an account ID is required for this provider." >&2; exit 1; }
  BASE_URL="${BASE_URL//\{account_id\}/$acct}"
  export SUBST_ACCOUNT_ID="$acct"
fi
if [ "$(helper field "$SIDECAR" "$TPL_KEY" gateway_id_required)" = "true" ]; then
  label="$(helper field "$SIDECAR" "$TPL_KEY" gateway_id_label)"
  hint="$(helper field "$SIDECAR" "$TPL_KEY" gateway_id_hint)"
  [ -n "$hint" ] && echo "  $hint"
  read -r -p "${label:-Gateway ID}: " gw
  [ -n "$gw" ] || { echo "error: a gateway ID is required for this provider." >&2; exit 1; }
  BASE_URL="${BASE_URL//\{gateway_id\}/$gw}"
  export SUBST_GATEWAY_ID="$gw"
fi

read -r -p "Base URL [$BASE_URL]: " in_url
BASE_URL="${in_url:-$BASE_URL}"
BASE_URL="${BASE_URL%/}"
[ -n "$BASE_URL" ] || { echo "error: a base URL is required." >&2; exit 1; }

# ── API key ──────────────────────────────────────────────────────────────────
[ -n "$KEY_HINT" ] && echo "  $KEY_HINT"
echo "  (a \${VAR} reference is stored as-is and resolved from the environment at request time)"
if [ "$KEY_REQUIRED" = "false" ]; then
  read -r -s -p "API key (optional): " API_KEY; echo
else
  read -r -s -p "API key: " API_KEY; echo
  [ -n "$API_KEY" ] || { echo "error: this provider requires an API key." >&2; exit 1; }
fi

# ── Config file path ─────────────────────────────────────────────────────────
echo
read -r -p "Config file [$DEFAULT_CONFIG]: " CONFIG_PATH
CONFIG_PATH="${CONFIG_PATH:-$DEFAULT_CONFIG}"
case "$CONFIG_PATH" in "~"/*) CONFIG_PATH="$HOME/${CONFIG_PATH#\~/}" ;; esac

if [ ! -f "$CONFIG_PATH" ]; then
  read -r -p "$CONFIG_PATH does not exist. Create it? [Y/n]: " yn
  case "${yn:-Y}" in [Nn]*) echo "Aborted."; exit 1 ;; esac
fi

# ── Merge ────────────────────────────────────────────────────────────────────
RESULT="$(helper merge "$CONFIG_PATH" "$SIDECAR" "$PROVIDER_KEY" "$TPL_KEY" "$BASE_URL" "$API_KEY")"
echo
if [ "$RESULT" = "updated" ]; then
  echo "Updated provider '$PROVIDER_KEY' in $CONFIG_PATH (previous version saved to $CONFIG_PATH.bak)."
else
  echo "Added provider '$PROVIDER_KEY' to $CONFIG_PATH."
fi

# ── Reachability check ───────────────────────────────────────────────────────
MODELS_URL="$(helper field "$SIDECAR" "$TPL_KEY" models_url)"
MODELS_URL="${MODELS_URL//\{account_id\}/${SUBST_ACCOUNT_ID:-}}"
MODELS_URL="${MODELS_URL//\{gateway_id\}/${SUBST_GATEWAY_ID:-}}"
FETCH_URL="${MODELS_URL:-$BASE_URL/models}"
echo
echo "Checking $FETCH_URL …"
BODY="$TMPDIR_RUN/models.json"
HTTP_CODE="$(curl -sS -o "$BODY" -w '%{http_code}' --max-time 45 \
  -H 'Accept: application/json' \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  "$FETCH_URL" || echo 000)"
if [ "$HTTP_CODE" = "200" ]; then
  echo "  OK — $(helper count-models "$BODY" 2>/dev/null || echo '?') model(s) advertised."
else
  echo "  HTTP $HTTP_CODE — the provider was still saved; llmproxy will retry at startup."
  [ -s "$BODY" ] && { head -c 300 "$BODY"; echo; }
fi

# ── Offer to add the template to this checkout if it is missing ──────────────
IN_FILE="$(printf '%s' "$MENU_JSON" | KEY="$TPL_KEY" python3 -c '
import json, os, sys
print("1" if os.environ["KEY"] in json.load(sys.stdin)["in_file"] else "0")')"

if [ "$IN_FILE" != "1" ]; then
  echo
  echo "'$TPL_KEY' is not in this checkout's llmproxy/providers.json."
  read -r -p "Add it there (and to provider_order) too? [y/N]: " yn
  case "${yn:-N}" in
    [Yy]*)
      echo "  providers.json: $(helper patch-sidecar "$SIDECAR" "$TPL_KEY")"
      if [ -f "$REPO_ROOT/scripts/update_free_models.py" ]; then
        (cd "$REPO_ROOT" && python3 scripts/update_free_models.py --regen-config-only) \
          && echo "  config.example.json: regenerated"
      fi
      ;;
    *) echo "  Skipped — the config entry above works regardless." ;;
  esac
fi

echo
echo "Done. Restart llmproxy to pick up the new provider."
