#!/usr/bin/env bash
#
# add_provider.sh — add a provider to an llmproxy config without running the
# full `llmproxy --setup` wizard.
#
# Prompts for the server (local by default, or any ssh destination), the
# provider (pick by number or type to filter), its API key and the config file
# path, backs the existing config up to /tmp, then merges providers.<key> into
# config.json — the same file and the same shape the wizard writes. For a remote
# server the prompts run here and the backup and merge run there over ssh. If the chosen provider is missing from this checkout's
# llmproxy/providers.json, it also offers to add it there (and to provider_order)
# and to regenerate config.example.json, so the script works on older checkouts.
#
# Requires: python3 (already required to run llmproxy) and curl; ssh, and
# python3 on the remote host, for a remote server. No jq.
#
# Usage: scripts/add_provider.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SIDECAR="$REPO_ROOT/llmproxy/providers.json"

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
    ("aihubmix", {
        "display": "AIHubMix (multi-vendor gateway)",
        "base_url": "https://aihubmix.com/v1",
        "key_required": True,
        "key_hint": "Create an API key at console.aihubmix.com (docs: docs.aihubmix.com). Chat is served from aihubmix.com/v1; aihubmix.com/api/v1 is a catalog-only API with no /chat/completions.",
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
    ("tokenharbor", {
        "display": "Token Harbor (multi-vendor gateway)",
        "base_url": "https://tokenharbor.ai/v1",
        "key_required": True,
        "key_hint": "Create an API key at tokenharbor.ai/dashboard/api-keys (Universal Key, format thk_live_...; docs: tokenharbor.ai/docs). Free models carry a ':free' id suffix and never bill the wallet.",
    }),
    ("tokenrouter", {
        "display": "TokenRouter (multi-vendor gateway)",
        "base_url": "https://api.tokenrouter.com/v1",
        "key_required": True,
        "key_hint": "Create an API key in the TokenRouter console at tokenrouter.com (model prices: tokenrouter.com/models). Name the entry 'tokenrouter' so the bundled per-model pricing applies.",
    }),
    ("atria-asi", {
        "display": "Atria ASI",
        "base_url": "https://api.atria-asi.ai/v1",
        "key_required": True,
        "key_hint": "Create an API key at api.atria-asi.ai/console/keys (Google sign-in). Keys look like atr_... (docs: api.atria-asi.ai/docs)",
    }),
    ("unbiased-ai", {
        "display": "Unbiased AI",
        "base_url": "https://api.unbiased.ai/v1",
        "key_required": True,
        "key_hint": "Create an API key at platform.unbiased.ai (signup is reviewed by hand). Keys look like sk_...",
    }),
    ("kilo", {
        "display": "Kilo AI (multi-vendor gateway)",
        "base_url": "https://api.kilo.ai/api/gateway",
        "key_required": True,
        "key_hint": "Create an API key at app.kilo.ai → Your Profile on your personal account (not an organization), then scroll to the bottom of the page. Requires account credits (docs: kilo.ai/docs/gateway). The base path is /api/gateway, not /v1.",
    }),
    ("modelscope", {
        "display": "ModelScope (Alibaba API-Inference)",
        "base_url": "https://api-inference.modelscope.cn/v1",
        "key_required": True,
        "key_hint": "Create an SDK token at modelscope.cn/my/myaccesstoken. Tokens look like ms-... (docs: modelscope.cn/docs/model-service/API-Inference/intro). Free tier is an account-wide 2,000 requests/day.",
    }),
    ("aion-labs", {
        "display": "Aion Labs",
        "base_url": "https://api.aionlabs.ai/v1",
        "key_required": True,
        "key_hint": "Create an API key at aionlabs.ai/app/api-keys (docs: aionlabs.ai/docs/api-reference)",
    }),
    ("agnes-ai", {
        "display": "Agnes AI (multimodal gateway)",
        "base_url": "https://apihub.agnes-ai.com/v1",
        "key_required": True,
        "key_hint": "Create an API key in the Agnes console at platform.agnes-ai.com → API Key management. Keys look like sk-... (docs: wiki.agnes-ai.com). The catalog also lists image and video models; set model_filter to the text ones.",
    }),
    ("xiaomi-mimo", {
        "display": "Xiaomi MiMo",
        "base_url": "https://api.xiaomimimo.com/v1",
        "key_required": True,
        "key_hint": "Create an API key in the Xiaomi MiMo API Open Platform console at platform.xiaomimimo.com → API Keys (docs: mimo.mi.com/docs)",
    }),
    ("openmathmodel", {
        "display": "Open Math Model (Open Math OS)",
        "base_url": "https://os.openmathmodel.org/v1",
        "key_required": True,
        "key_hint": "Use a team API key from Open Math OS at os.openmathmodel.org; keys look like nuc_ak_... and are issued per team (teams by application at openmathmodel.org).",
    }),
    ("demonroute", {
        "display": "Demon Route",
        "base_url": "https://api.demonroute.com/v1",
        "key_required": True,
        "key_hint": "Create an API key at demonroute.com (Console → Keys)",
    }),
    ("meta-muse", {
        "display": "Meta Muse (Meta Model API)",
        "base_url": "https://api.meta.ai/v1",
        "key_required": True,
        "key_hint": "Create an API key in the Meta developer console at dev.meta.ai (Bearer auth; docs: dev.meta.ai/docs/models)",
        "protocol_choices": ["responses", "openai"],
        "example_model_filter": ["muse-wonder-1.0", "muse-spark-1.3", "muse-spark-1.2", "muse-spark-1.1", "muse-ember-1.0"],
    }),
])

# Fields the wizard copies from a template onto the config entry.
PASSTHROUGH = ("models_url", "models_id_field", "models_keep_task", "protocol",
               "free_allowance", "_note")


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

    elif cmd == "template":
        # template <sidecar> <key>  — the template as JSON, for merge
        json.dump(menu(args[0])["templates"].get(args[1], {}), sys.stdout,
                  ensure_ascii=False)

    elif cmd == "merge":
        # merge <config> <provider_key> <base_url>
        # The template JSON and the API key arrive in the environment
        # (ADD_PROVIDER_TEMPLATE, ADD_PROVIDER_API_KEY), never on the command
        # line, where any local user could read the key from `ps`. Taking the
        # template as JSON rather than a sidecar path also lets merge run on a
        # remote host that has no llmproxy checkout.
        cfg_path, pkey, base_url = args
        api_key = os.environ.get("ADD_PROVIDER_API_KEY", "")
        tpl = json.loads(os.environ.get("ADD_PROVIDER_TEMPLATE") or "{}",
                         object_pairs_hook=OrderedDict)

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
        # A template offering several protocols defaults to its first choice,
        # as the wizard does on Enter and as config.example.json shows.
        choices = tpl.get("protocol_choices") or []
        if not tpl.get("protocol") and choices and choices[0] != "openai":
            entry["protocol"] = choices[0]
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
# key <TAB> display <TAB> base_url, one row per template, in menu order.
mapfile -t ROWS < <(printf '%s' "$MENU_JSON" | python3 -c '
import json, sys
for key, tpl in json.load(sys.stdin)["templates"].items():
    disp = str(tpl.get("display") or key).replace("\t", " ")
    print("\t".join([key, disp, str(tpl.get("base_url") or "")]))')
KEYS=(); DISPLAYS=(); URLS=()
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r k d u <<<"$row"
  KEYS+=("$k"); DISPLAYS+=("$d"); URLS+=("${u:-}")
done

echo
echo "llmproxy — add a provider"
echo "========================="

# ── Target server ────────────────────────────────────────────────────────────
# "local" edits this machine's config. Anything else is an ssh destination
# (host, user@host or an ~/.ssh/config alias): the prompts all run here, and the
# backup and merge run on that host over ssh. The API key travels inside the
# ssh session's stdin, never on a command line.
echo
read -r -p "Server to add it to (local, or an ssh host such as user@host) [local]: " TARGET
TARGET="${TARGET:-local}"
case "$TARGET" in local|localhost|127.0.0.1|::1) TARGET=local ;; esac
if [ "$TARGET" != "local" ]; then
  command -v ssh >/dev/null 2>&1 || { echo "error: ssh is required for a remote server." >&2; exit 1; }
  echo "  Checking ssh access to $TARGET …"
  ssh -n -o ConnectTimeout=15 "$TARGET" 'command -v python3 >/dev/null 2>&1' \
    || { echo "error: cannot reach $TARGET over ssh, or python3 is not installed there." >&2; exit 1; }
fi

# Run a bash script read from stdin on the target (this machine or $TARGET).
run_on_target() {
  if [ "$TARGET" = "local" ]; then bash -s; else ssh -T -o ConnectTimeout=15 "$TARGET" bash -s; fi
}

# ── Provider menu (pick by number, or type text to filter) ──────────────────
print_rows() {
  local idx
  for idx in "$@"; do
    printf '%3d) %-42s %s\n' "$((idx + 1))" "${DISPLAYS[idx]}" "${URLS[idx]}"
  done
}
ALL_IDX=("${!KEYS[@]}")
echo
print_rows "${ALL_IDX[@]}"
echo
echo "Enter a number to choose, text to filter by key, name or URL (e.g. 'hub'),"
echo "or press Enter to show the full list again."
SEL=""
while [ -z "$SEL" ]; do
  read -r -p "Select a provider [1-${#KEYS[@]} or filter text]: " choice
  case "$choice" in
    '') print_rows "${ALL_IDX[@]}" ;;
    *[!0-9]*)
      q="${choice,,}"
      matches=()
      for i in "${ALL_IDX[@]}"; do
        hay="${KEYS[i]} ${DISPLAYS[i]} ${URLS[i]}"
        [[ "${hay,,}" == *"$q"* ]] && matches+=("$i")
      done
      if [ "${#matches[@]}" -eq 0 ]; then
        echo "  No provider matches '$choice'."
      elif [ "${#matches[@]}" -eq 1 ]; then
        read -r -p "  Use ${DISPLAYS[matches[0]]}? [Y/n]: " yn
        case "${yn:-Y}" in [Nn]*) ;; *) SEL="${matches[0]}" ;; esac
      else
        print_rows "${matches[@]}"
      fi
      ;;
    *) if [ "$choice" -ge 1 ] && [ "$choice" -le "${#KEYS[@]}" ]; then
         SEL="$((choice - 1))"
       else
         echo "  Out of range."
       fi ;;
  esac
done
TPL_KEY="${KEYS[SEL]}"

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

# ── Config file path (resolved on the target, so ~ means that host's home) ──
if [ "$TARGET" = "local" ]; then
  DEFAULT_CONFIG="${LLMPROXY_CONFIG:-~/.config/llmproxy/config.json}"
else
  DEFAULT_CONFIG="~/.config/llmproxy/config.json"
fi
echo
read -r -p "Config file on $TARGET [$DEFAULT_CONFIG]: " CONFIG_PATH
CONFIG_PATH="${CONFIG_PATH:-$DEFAULT_CONFIG}"

# Prints "yes <path>" or "no <path>" with ~ expanded on the target.
PROBE="$({
  printf 'CONFIG_PATH=%q\n' "$CONFIG_PATH"
  cat <<'SH_EOF'
case "$CONFIG_PATH" in "~") CONFIG_PATH="$HOME" ;; "~"/*) CONFIG_PATH="$HOME/${CONFIG_PATH#\~/}" ;; esac
if [ -f "$CONFIG_PATH" ]; then echo "yes $CONFIG_PATH"; else echo "no $CONFIG_PATH"; fi
SH_EOF
} | run_on_target)"
CONFIG_PATH="${PROBE#* }"
if [ "${PROBE%% *}" != "yes" ]; then
  read -r -p "$CONFIG_PATH does not exist on $TARGET. Create it? [Y/n]: " yn
  case "${yn:-Y}" in [Nn]*) echo "Aborted."; exit 1 ;; esac
fi

# ── Back up to /tmp, then merge (on the target) ─────────────────────────────
TPL_JSON="$(helper template "$SIDECAR" "$TPL_KEY")"
OUT="$({
  echo 'set -euo pipefail'
  printf 'CONFIG_PATH=%q\nPROVIDER_KEY=%q\nBASE_URL=%q\n' "$CONFIG_PATH" "$PROVIDER_KEY" "$BASE_URL"
  printf 'export ADD_PROVIDER_API_KEY=%q\n' "$API_KEY"
  printf 'export ADD_PROVIDER_TEMPLATE=%q\n' "$TPL_JSON"
  printf 'export SUBST_ACCOUNT_ID=%q SUBST_GATEWAY_ID=%q\n' "${SUBST_ACCOUNT_ID:-}" "${SUBST_GATEWAY_ID:-}"
  echo 'T="$(mktemp -d)"; trap '"'"'rm -rf "$T"'"'"' EXIT'
  echo "cat > \"\$T/helper.py\" <<'PY_EOF'"
  cat "$HELPER"
  echo 'PY_EOF'
  cat <<'SH_EOF'
if [ -f "$CONFIG_PATH" ]; then
  # mktemp creates the file 0600; cp keeps that mode, since the copy holds keys.
  BK="$(mktemp "/tmp/llmproxy-config.$(date +%Y%m%d-%H%M%S).XXXXXX")"
  cp "$CONFIG_PATH" "$BK"
  echo "backup $BK"
fi
echo "result $(python3 "$T/helper.py" merge "$CONFIG_PATH" "$PROVIDER_KEY" "$BASE_URL")"
SH_EOF
} | run_on_target)"
BACKUP="$(printf '%s\n' "$OUT" | sed -n 's/^backup //p')"
RESULT="$(printf '%s\n' "$OUT" | sed -n 's/^result //p')"
[ -n "$RESULT" ] || { echo "error: the merge on $TARGET did not complete." >&2; exit 1; }
echo
[ -n "$BACKUP" ] && echo "Backed up the original config on $TARGET to $BACKUP (and $CONFIG_PATH.bak)."
if [ "$RESULT" = "updated" ]; then
  echo "Updated provider '$PROVIDER_KEY' in $CONFIG_PATH on $TARGET."
else
  echo "Added provider '$PROVIDER_KEY' to $CONFIG_PATH on $TARGET."
fi

# ── Reachability check ───────────────────────────────────────────────────────
MODELS_URL="$(helper field "$SIDECAR" "$TPL_KEY" models_url)"
MODELS_URL="${MODELS_URL//\{account_id\}/${SUBST_ACCOUNT_ID:-}}"
MODELS_URL="${MODELS_URL//\{gateway_id\}/${SUBST_GATEWAY_ID:-}}"
FETCH_URL="${MODELS_URL:-$BASE_URL/models}"
echo
echo "Checking $FETCH_URL from this machine …"
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
echo "Done. Restart llmproxy on $TARGET to pick up the new provider."
