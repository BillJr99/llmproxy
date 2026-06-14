#!/usr/bin/env bash
#
# test.sh — smoke-test a running llmproxy server with curl.
#
# Exercises the OpenAI-compatible endpoints and the llmproxy virtual models,
# with extra coverage for capability-aware routing (tools / vision) added in
# the model_capabilities feature.
#
# Usage:
#   ./test.sh                              # against http://localhost:8080
#   ./test.sh http://localhost:9000        # custom base (no /v1 suffix)
#   BASE_URL=http://host:8080 ./test.sh
#   LLMPROXY_API_KEY=sk-... ./test.sh      # send Authorization: Bearer ...
#
# Requirements: curl. jq is optional but gives richer assertions.
#
# Exit code is non-zero if any hard check FAILs. WARN/SKIP do not fail the run
# (they cover behavior that depends on which providers/models you configured).

set -u

BASE_URL="${1:-${BASE_URL:-http://localhost:8080}}"
BASE_URL="${BASE_URL%/}"           # strip trailing slash
API_KEY="${LLMPROXY_API_KEY:-}"

# ── output helpers ──────────────────────────────────────────────────────────
if [ -t 1 ]; then
  RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; CYN=$'\033[36m'; DIM=$'\033[2m'; RST=$'\033[0m'
else
  RED=''; GRN=''; YEL=''; CYN=''; DIM=''; RST=''
fi
PASS=0; FAIL=0; WARN=0; SKIP=0
pass() { PASS=$((PASS+1)); echo "  ${GRN}PASS${RST} $1"; }
fail() { FAIL=$((FAIL+1)); echo "  ${RED}FAIL${RST} $1"; }
warn() { WARN=$((WARN+1)); echo "  ${YEL}WARN${RST} $1"; }
skip() { SKIP=$((SKIP+1)); echo "  ${DIM}SKIP $1${RST}"; }
hdr()  { echo; echo "${CYN}== $1 ==${RST}"; }

HAVE_JQ=0; command -v jq >/dev/null 2>&1 && HAVE_JQ=1
command -v curl >/dev/null 2>&1 || { echo "curl is required"; exit 2; }
[ "$HAVE_JQ" -eq 0 ] && echo "${YEL}note:${RST} jq not found — assertions fall back to grep."

BODY="$(mktemp)"; trap 'rm -f "$BODY"' EXIT
AUTH=(); [ -n "$API_KEY" ] && AUTH=(-H "Authorization: Bearer ${API_KEY}")

# req METHOD PATH [JSON]  -> echoes HTTP status; full body written to $BODY
req() {
  local method="$1" path="$2" data="${3:-}"
  if [ -n "$data" ]; then
    curl -sS -o "$BODY" -w '%{http_code}' -X "$method" "${BASE_URL}${path}" \
      -H 'Content-Type: application/json' "${AUTH[@]}" -d "$data"
  else
    curl -sS -o "$BODY" -w '%{http_code}' -X "$method" "${BASE_URL}${path}" "${AUTH[@]}"
  fi
}

# jq query against $BODY, or empty string if jq missing / query fails
jqr() { [ "$HAVE_JQ" -eq 1 ] && jq -r "$1" "$BODY" 2>/dev/null || echo ""; }

ok2xx() { [ "$1" -ge 200 ] && [ "$1" -lt 300 ]; }

echo "${CYN}llmproxy smoke test${RST} → ${BASE_URL}   ${DIM}(api_key: $([ -n "$API_KEY" ] && echo set || echo none))${RST}"

# ── reachability ────────────────────────────────────────────────────────────
hdr "Reachability"
code=$(req GET /health) || true
if [ -z "${code:-}" ] || [ "$code" = "000" ]; then
  fail "server not reachable at ${BASE_URL} (is it running? 'python run.py')"
  echo; echo "Aborting: ${FAIL} failure(s)."; exit 1
fi
ok2xx "$code" && pass "GET /health → $code" || fail "GET /health → $code"

code=$(req GET /version)
if ok2xx "$code"; then
  name=$(jqr '.name'); pass "GET /version → $code ${DIM}(${name:-?})${RST}"
else fail "GET /version → $code"; fi

# ── models listing + virtual model discovery ───────────────────────────────
hdr "Models"
code=$(req GET /v1/models)
if ok2xx "$code"; then
  n=$(jqr '.data | length'); pass "GET /v1/models → $code ${DIM}(${n:-?} models)${RST}"
else
  fail "GET /v1/models → $code"
fi

# Discover which virtual models are advertised and a real fallback model id.
ALL_IDS=""; FALLBACK_MODEL=""
if [ "$HAVE_JQ" -eq 1 ]; then
  ALL_IDS=$(jq -r '.data[].id' "$BODY" 2>/dev/null)
  FALLBACK_MODEL=$(echo "$ALL_IDS" | grep -v '^llmproxy__' | head -n1)
fi
has_model() { echo "$ALL_IDS" | grep -qx "$1"; }

for vm in llmproxy__free llmproxy__local llmproxy__tools llmproxy__tools/free llmproxy__vision llmproxy__vision/free llmproxy__fusion llmproxy__fusion/free; do
  if has_model "$vm"; then pass "advertised: $vm"; else skip "not advertised: $vm (no matching tagged models)"; fi
done

# Inspect candidate list for a capability endpoint when present.
if has_model llmproxy__tools; then
  code=$(req GET "/v1/models/llmproxy__tools")
  if ok2xx "$code"; then
    c=$(jqr '._candidates | length'); pass "GET /v1/models/llmproxy__tools → $code ${DIM}(${c:-?} candidates)${RST}"
  else warn "GET /v1/models/llmproxy__tools → $code"; fi
fi

# Choose a model for the chat tests: prefer llmproxy__free, else a real model.
CHAT_MODEL=""
if has_model llmproxy__free; then CHAT_MODEL="llmproxy__free"
elif [ -n "$FALLBACK_MODEL" ]; then CHAT_MODEL="$FALLBACK_MODEL"; fi

# ── basic chat completion ───────────────────────────────────────────────────
hdr "Chat completion"
if [ -z "$CHAT_MODEL" ]; then
  skip "no usable model found in /v1/models — configure a provider first"
else
  echo "  ${DIM}model: ${CHAT_MODEL}${RST}"
  code=$(req POST /v1/chat/completions "{\"model\":\"${CHAT_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":16}")
  content=$(jqr '.choices[0].message.content')
  if ok2xx "$code" && { [ "$HAVE_JQ" -eq 0 ] || [ -n "$content" ]; }; then
    pass "chat → $code ${DIM}($(echo "${content:-ok}" | head -c 40))${RST}"
  else
    fail "chat → $code $(head -c 200 "$BODY")"
  fi
fi

# ── streaming (SSE) ─────────────────────────────────────────────────────────
hdr "Streaming"
if [ -z "$CHAT_MODEL" ]; then
  skip "no usable model for streaming"
else
  out=$(curl -sS -N -X POST "${BASE_URL}/v1/chat/completions" \
        -H 'Content-Type: application/json' "${AUTH[@]}" \
        -d "{\"model\":\"${CHAT_MODEL}\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"count: 1 2 3\"}],\"max_tokens\":16}" \
        2>/dev/null | head -c 4000)
  if echo "$out" | grep -q '^data:'; then pass "stream emitted SSE 'data:' chunks"
  else fail "stream produced no SSE chunks"; fi
fi

# ── tool calling (capability-aware routing + failover) ──────────────────────
hdr "Tool calling"
TOOL_MODEL=""
if has_model llmproxy__tools; then TOOL_MODEL="llmproxy__tools"
elif [ -n "$CHAT_MODEL" ]; then TOOL_MODEL="$CHAT_MODEL"; fi
if [ -z "$TOOL_MODEL" ]; then
  skip "no model available for tool test"
else
  echo "  ${DIM}model: ${TOOL_MODEL} (tool_choice: required)${RST}"
  read -r -d '' TOOL_REQ <<JSON
{"model":"${TOOL_MODEL}","tool_choice":"required",
 "messages":[{"role":"user","content":"What is the weather in Paris?"}],
 "tools":[{"type":"function","function":{"name":"get_weather",
   "description":"Get current weather for a city",
   "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}
JSON
  code=$(req POST /v1/chat/completions "$TOOL_REQ")
  tc_count=$(jqr '[.choices[0].message.tool_calls // []] | flatten | length')
  fn=$(jqr '.choices[0].message.tool_calls[0].function.name')
  if ! ok2xx "$code"; then
    fail "tool call → $code $(head -c 200 "$BODY")"
  elif [ "$HAVE_JQ" -eq 0 ]; then
    grep -q 'tool_calls' "$BODY" && pass "tool call → $code (tool_calls present)" || warn "tool call → $code (install jq to verify)"
  elif [ "${tc_count:-0}" -ge 1 ]; then
    pass "tool call → $code ${DIM}(called ${fn:-?})${RST}"
  else
    warn "tool call → $code but no tool_calls returned (no tool-capable model honored it — check model_capabilities)"
  fi
fi

# ── vision (image input routing) ────────────────────────────────────────────
hdr "Vision"
if has_model llmproxy__vision; then
  # 1x1 transparent PNG, inlined as a data URL.
  PNG="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
  read -r -d '' VIS_REQ <<JSON
{"model":"llmproxy__vision","max_tokens":16,
 "messages":[{"role":"user","content":[
   {"type":"text","text":"Reply with the single word: ok"},
   {"type":"image_url","image_url":{"url":"${PNG}"}}]}]}
JSON
  code=$(req POST /v1/chat/completions "$VIS_REQ")
  if ok2xx "$code"; then pass "vision request → $code"
  else warn "vision request → $code $(head -c 160 "$BODY")"; fi
else
  skip "llmproxy__vision not advertised (no vision-tagged models configured)"
fi

# ── OpenAI legacy completions ───────────────────────────────────────────────
hdr "OpenAI /v1/completions (legacy)"
if [ -z "$CHAT_MODEL" ]; then
  skip "no usable model for /v1/completions"
else
  code=$(req POST /v1/completions "{\"model\":\"${CHAT_MODEL}\",\"prompt\":\"Reply with exactly: pong\",\"max_tokens\":16}")
  if ok2xx "$code"; then pass "completions → $code"
  else warn "completions → $code (not all providers support the legacy endpoint)"; fi
fi

# ── Anthropic Messages family ───────────────────────────────────────────────
# llmproxy also speaks the Anthropic dialect inbound: /v1/messages (+ streaming)
# and /v1/messages/count_tokens. These route to the same models as the OpenAI
# endpoints, so the same CHAT_MODEL is reused.
hdr "Anthropic /v1/messages"
if [ -z "$CHAT_MODEL" ]; then
  skip "no usable model for /v1/messages"
else
  echo "  ${DIM}model: ${CHAT_MODEL}${RST}"
  code=$(req POST /v1/messages "{\"model\":\"${CHAT_MODEL}\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}]}")
  text=$(jqr '.content[0].text')
  typ=$(jqr '.type')
  if ok2xx "$code" && { [ "$HAVE_JQ" -eq 0 ] || [ "$typ" = "message" ]; }; then
    pass "messages → $code ${DIM}($(echo "${text:-ok}" | head -c 40))${RST}"
  else
    fail "messages → $code $(head -c 200 "$BODY")"
  fi

  # count_tokens utility
  code=$(req POST /v1/messages/count_tokens "{\"model\":\"${CHAT_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"how many tokens is this\"}]}")
  n=$(jqr '.input_tokens')
  if ok2xx "$code" && { [ "$HAVE_JQ" -eq 0 ] || [ -n "$n" ]; }; then
    pass "messages/count_tokens → $code ${DIM}(${n:-?} tokens)${RST}"
  else
    warn "messages/count_tokens → $code"
  fi
fi

hdr "Anthropic /v1/messages streaming"
if [ -z "$CHAT_MODEL" ]; then
  skip "no usable model for /v1/messages streaming"
else
  out=$(curl -sS -N -X POST "${BASE_URL}/v1/messages" \
        -H 'Content-Type: application/json' "${AUTH[@]}" \
        -d "{\"model\":\"${CHAT_MODEL}\",\"max_tokens\":16,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"count: 1 2 3\"}]}" \
        2>/dev/null | head -c 4000)
  if echo "$out" | grep -q '^event: message_start' && echo "$out" | grep -q '^event: message_stop'; then
    pass "stream emitted Anthropic events (message_start … message_stop)"
  elif echo "$out" | grep -q '^event:'; then
    warn "stream emitted some Anthropic events but not the full envelope"
  else
    fail "stream produced no Anthropic SSE events"
  fi
fi

# ── Gemini generateContent family ───────────────────────────────────────────
# llmproxy also speaks the Gemini dialect inbound: the model rides in the URL
# path and streaming uses :streamGenerateContent. CHAT_MODEL (a /v1/models id
# such as provider__model or llmproxy__free) carries no slash, so it slots
# straight into the path.
hdr "Gemini /v1beta/models/{model}:generateContent"
if [ -z "$CHAT_MODEL" ]; then
  skip "no usable model for Gemini generateContent"
else
  echo "  ${DIM}model: ${CHAT_MODEL}${RST}"
  code=$(req POST "/v1beta/models/${CHAT_MODEL}:generateContent" "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"Reply with exactly: pong\"}]}],\"generationConfig\":{\"maxOutputTokens\":16}}")
  text=$(jqr '.candidates[0].content.parts[0].text')
  if ok2xx "$code" && { [ "$HAVE_JQ" -eq 0 ] || grep -q '"candidates"' "$BODY"; }; then
    pass "generateContent → $code ${DIM}($(echo "${text:-ok}" | head -c 40))${RST}"
  else
    fail "generateContent → $code $(head -c 200 "$BODY")"
  fi

  code=$(req POST "/v1beta/models/${CHAT_MODEL}:countTokens" "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"how many tokens is this\"}]}]}")
  n=$(jqr '.totalTokens')
  if ok2xx "$code" && { [ "$HAVE_JQ" -eq 0 ] || [ -n "$n" ]; }; then
    pass "countTokens → $code ${DIM}(${n:-?} tokens)${RST}"
  else
    warn "countTokens → $code"
  fi

  out=$(curl -sS -N -X POST "${BASE_URL}/v1beta/models/${CHAT_MODEL}:streamGenerateContent?alt=sse" \
        -H 'Content-Type: application/json' "${AUTH[@]}" \
        -d "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"count: 1 2 3\"}]}],\"generationConfig\":{\"maxOutputTokens\":16}}" \
        2>/dev/null | head -c 4000)
  if echo "$out" | grep -q '"candidates"'; then pass "streamGenerateContent emitted Gemini SSE chunks"
  else fail "streamGenerateContent produced no Gemini SSE chunks"; fi
fi

# ── error handling ──────────────────────────────────────────────────────────
hdr "Error handling"
code=$(req POST /v1/chat/completions '{"messages":[{"role":"user","content":"hi"}]}')
[ "$code" = "400" ] && pass "missing model → 400" || warn "missing model → $code (expected 400)"

code=$(req POST /v1/chat/completions '{"model":"definitely/not-a-real-model","messages":[{"role":"user","content":"hi"}]}')
if [ "$code" -ge 400 ]; then pass "unknown model → $code (rejected)"; else fail "unknown model → $code (expected >=400)"; fi

# ── fusion (multi-model deliberation) ───────────────────────────────────────
hdr "Fusion"
# Prefer the general llmproxy__fusion pool (draws from all configured providers,
# typically paid keys that are reliably reachable) and fall back to the free pool
# only when the general one isn't advertised. The free-tier panel is the flakier
# target — its members rate-limit and blip — so it makes a poorer smoke signal.
FUSION_MODEL=""
if has_model llmproxy__fusion; then FUSION_MODEL="llmproxy__fusion"
elif has_model "llmproxy__fusion/free"; then FUSION_MODEL="llmproxy__fusion/free"; fi
if [ -z "$FUSION_MODEL" ]; then
  skip "no fusion model advertised (need >=2 eligible models)"
else
  echo "  ${DIM}model: ${FUSION_MODEL}${RST}"
  # Capture headers (-D) so we can assert the additive provenance channel.
  HDRS="$(mktemp)"
  code=$(curl -sS -o "$BODY" -D "$HDRS" -w '%{http_code}' -X POST \
        "${BASE_URL}/v1/chat/completions" -H 'Content-Type: application/json' "${AUTH[@]}" \
        -d "{\"model\":\"${FUSION_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, name a tradeoff between REST and gRPC.\"}],\"max_tokens\":128}")
  content=$(jqr '.choices[0].message.content')
  if ok2xx "$code" && { [ "$HAVE_JQ" -eq 0 ] || [ -n "$content" ]; }; then
    pass "fusion chat → $code ${DIM}($(echo "${content:-ok}" | head -c 40))${RST}"
  else
    fail "fusion chat → $code $(head -c 200 "$BODY")"
  fi
  # Provenance: X-LLMProxy-Fusion header (always) and llmproxy_fusion body field.
  if grep -qi '^x-llmproxy-fusion:' "$HDRS"; then pass "X-LLMProxy-Fusion header present"
  else warn "no X-LLMProxy-Fusion header (older build or upstream error)"; fi
  if [ "$HAVE_JQ" -eq 1 ]; then
    panel=$(jq -r 'if has("llmproxy_fusion") then (.llmproxy_fusion.panel | length) else "absent" end' "$BODY" 2>/dev/null)
    if [ -n "$panel" ] && [ "$panel" != "null" ] && [ "$panel" != "absent" ] && [ "$panel" != "0" ]; then
      pass "llmproxy_fusion body field present ${DIM}(${panel} panel models)${RST}"
    else warn "no llmproxy_fusion body field (synth may have fallen back / error / non-OpenAI render)"; fi
  fi
  rm -f "$HDRS"
fi

# ── input-aware first-pick on the general virtuals ──────────────────────────
hdr "Input-aware routing (llmproxy__free)"
if ! has_model llmproxy__free; then
  skip "llmproxy__free not advertised"
else
  # A tiny prompt and a 'thinking' request should both succeed; the proxy biases
  # the first model tried by input size/type, but failover still guarantees a
  # reply, so this asserts functionality rather than which tier was chosen.
  code=$(req POST /v1/chat/completions "{\"model\":\"llmproxy__free\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}")
  ok2xx "$code" && pass "small prompt routed → $code" || warn "small prompt → $code"
  code=$(req POST /v1/chat/completions "{\"model\":\"llmproxy__free\",\"reasoning_effort\":\"high\",\"messages\":[{\"role\":\"user\",\"content\":\"Think carefully, then answer: 2+2?\"}],\"max_tokens\":16}")
  ok2xx "$code" && pass "thinking request routed → $code" || warn "thinking request → $code"
fi

# ── summary ─────────────────────────────────────────────────────────────────
hdr "Summary"
echo "  ${GRN}${PASS} passed${RST}, ${RED}${FAIL} failed${RST}, ${YEL}${WARN} warned${RST}, ${DIM}${SKIP} skipped${RST}"
[ "$FAIL" -eq 0 ] || exit 1
