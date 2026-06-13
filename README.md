# llmproxy

An OpenAI-compatible HTTP proxy that aggregates multiple LLM providers behind a
single endpoint.  Clients that speak the OpenAI API (LangChain, LiteLLM, Open
WebUI, Cursor, etc.) connect to llmproxy without modification; llmproxy routes
each request to the correct upstream based on a provider-prefix embedded in the
model name.

---

## File overview

```
llmproxy/
├── run.py                   ← start the server (no install needed)
├── llmproxy_test_client.py  ← live integration test client (talks to a running proxy)
├── test_tui.py              ← interactive chat TUI (despite the name — not a test suite)
├── llmproxy/                ← the package
│   ├── __main__.py
│   ├── config.py
│   ├── server.py
│   ├── usage.py             ← token/cost accounting primitives (GET /v1/usage)
│   ├── setup_wizard.py
│   ├── admin.py             ← web admin UI + config API (/admin, /admin/api/*)
│   ├── static/admin/        ← self-contained single-page admin frontend
│   ├── providers.py         ← loader for the JSON sidecar
│   └── providers.json       ← single source of truth for ALL provider templates
│                                (+ believed_free / model_reasoning / model_capabilities / free_limits / pricing)
├── scripts/
│   └── update_free_models.py ← scraper that keeps providers.json's free-tier fields current
│       └── sources/         ← per-source plugins (openrouter, community, /models, docs, litellm, probe)
├── tests/                   ← pytest unit/integration suite
├── requirements.txt
├── requirements-dev.txt     ← pytest, ruff, responses (test-only deps)
├── pyproject.toml           ← pytest + ruff config
├── setup.py                 ← only needed for pip install
├── Dockerfile
├── docker-compose.yml
├── config.example.json      ← auto-generated from llmproxy/providers.json
└── .github/workflows/
    ├── ci.yml               ← pytest, ruff, config-example-up-to-date guard
    └── docker-publish.yml   ← GHCR image publish
```

---

## Model naming convention

All models exposed by llmproxy follow this pattern:

```
<provider_name>/<upstream_model_id>
```

The `upstream_model_id` may itself contain slashes.  Examples:

| Proxy model string                         | Provider   | Upstream model                |
|--------------------------------------------|------------|-------------------------------|
| `openrouter/openrouter/free`               | openrouter | `openrouter/free`             |
| `openrouter/anthropic/claude-3.5-sonnet`   | openrouter | `anthropic/claude-3.5-sonnet` |
| `openai/gpt-4o`                            | openai     | `gpt-4o`                      |
| `deepseek/deepseek-chat`                   | deepseek   | `deepseek-chat`               |
| `ollama/llama3`                            | ollama     | `llama3`                      |

The proxy strips the leading `<provider_name>/` before forwarding the request to
the upstream provider's base URL.

### Display format returned by `GET /v1/models`

While the slash form above is the canonical **input** form (accepted by every
endpoint), `GET /v1/models` advertises ids in a different **display** form:

```
<provider_name>__<upstream_model_id>
```

For example, an Ollama model with upstream id `qwen2.5vl:3b` is listed as
`ollama__qwen2.5vl:3b`. The double-underscore separator avoids two real-world
client bugs:

- Spaces and parentheses break strict client validators (e.g. Hermes rejects
  any model name containing whitespace).
- A `/` separator causes some clients to silently truncate the id at the first
  `/`, hiding the provider suffix in their menus.

Putting the provider on the left mirrors the canonical `provider/model` slash
form, so the two ids read consistently across logs, configs, and menus.

**Single-slash flattening.** Some upstream ids contain *multiple* slashes
(e.g. OpenRouter's `meta-llama/llama-3/instruct`). In the display form, all but
the **last** slash are collapsed into `_`, so a display id always carries at most
one `/`:

| Upstream model (under `openrouter`)   | Display id                               |
|---------------------------------------|------------------------------------------|
| `gpt-4o`                              | `openrouter__gpt-4o`                     |
| `anthropic/claude-3.5-sonnet`         | `openrouter__anthropic/claude-3.5-sonnet`|
| `meta-llama/llama-3/instruct`         | `openrouter__meta-llama_llama-3/instruct`|

This keeps the proxy grammar unambiguous — `__` always separates the provider
and there is never more than a single `/` — which also matters for the
per-provider virtual models below. Routing always forwards to the upstream under
the **original** (un-flattened) id, and the **input** forms are unchanged: you can
still send the canonical `provider/upstream` slash form with a multi-slash
upstream (e.g. `openrouter/meta-llama/llama-3/instruct`).

Clients may submit any of these four forms in `"model"` on chat/completions
requests:

- `provider/model` — canonical slash form
- `provider__model` — current display form
- `model__provider` — legacy display form from PR #27
- `model (provider)` — pre-PR #27 legacy display form

All four resolve identically.

### The `free` virtual model

llmproxy advertises a special synthetic model named `llmproxy__free`.  When a request
arrives with `"model": "llmproxy__free"` (or the legacy `"llmproxy/free"`), the proxy:

1. Collects every model across all providers whose upstream ID contains the
   word `free` (case-insensitive) **or** whose upstream ID (or full
   `provider/upstream` ID) appears in the top-level `believed_free` config list
   — see [Configuration](#configuration).
2. Picks a **random starting position** in that list, then tries each
   candidate in order, wrapping around.
3. Returns the first response with an HTTP status below 400.  If a candidate
   is rate-limited, overloaded, or otherwise unhealthy, it is skipped silently
   and the next one is tried.

This spreads load across free-tier endpoints and provides automatic failover —
useful when any individual free model is rate-limited.

```bash
# Use the free virtual model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy__free", "messages": [{"role": "user", "content": "Hello!"}]}'

# Inspect which backends are currently eligible
curl http://localhost:8080/v1/models/llmproxy__free | jq '._candidates'
```

The `llmproxy__free` model appears at the top of `GET /v1/models` whenever at least one
eligible backend is available.

### The `local` virtual model

llmproxy also advertises a synthetic model named `llmproxy__local`.  When a request
arrives with `"model": "llmproxy__local"` (or the legacy `"llmproxy/local"`), the proxy:

1. Collects every model across all providers whose `base_url` hostname matches
   a loopback address (`localhost`, `127.x.x.x`, `::1`, `0.0.0.0`), an mDNS
   name (`*.local`), or a Docker host-gateway alias (`host.docker.internal`,
   `gateway.docker.internal`).
2. Picks a **random starting position** in that list, then tries each
   candidate in order, wrapping around.
3. Returns the first response with an HTTP status below 400.

This is useful for clients that want to use whichever local model (Ollama,
LM Studio, llama.cpp, etc.) happens to be running without hard-coding a
specific model name.

```bash
# Use the local virtual model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy__local", "messages": [{"role": "user", "content": "Hello!"}]}'

# Inspect which backends are currently eligible
curl http://localhost:8080/v1/models/llmproxy__local | jq '._candidates'
```

The `llmproxy__local` model appears in `GET /v1/models` only when at least one model
from a localhost-backed provider is present in the route cache — meaning the
provider must be reachable and its `/models` listing must have been fetched
successfully.

> **Local models are not added to `believed_free`.** Local-provider models
> (Ollama, LM Studio, OpenWebUI, etc.) live entirely under the `__local` family
> — `llmproxy__local`, `llmproxy__standard/local`, and so on. When the setup
> wizard auto-registers a local provider, it tags each discovered model in
> `model_reasoning` only; `believed_free` is reserved for cloud free-tier
> offerings. If you want a local model to also appear under `llmproxy__free`,
> add it to `believed_free` by hand.

### Reasoning-level virtual models

You can optionally tag individual models in the config with a **reasoning
level** — `exploratory`, `standard`, or `deep` — to group them by how much
thinking effort they are expected to apply.  When at least one model is tagged
with a given level, llmproxy exposes corresponding virtual endpoints:

| Virtual model name           | Selects                                                       |
|------------------------------|---------------------------------------------------------------|
| `llmproxy__exploratory`      | All models tagged `exploratory`                               |
| `llmproxy__standard`         | All models tagged `standard`                                  |
| `llmproxy__deep`             | All models tagged `deep`                                      |
| `llmproxy__exploratory/free` | Models tagged `exploratory` **and** qualifying as free-tier   |
| `llmproxy__exploratory/local`| Models tagged `exploratory` **and** served on localhost       |
| `llmproxy__standard/free`    | Models tagged `standard` **and** qualifying as free-tier      |
| `llmproxy__standard/local`   | Models tagged `standard` **and** served on localhost          |
| `llmproxy__deep/free`        | Models tagged `deep` **and** qualifying as free-tier          |
| `llmproxy__deep/local`       | Models tagged `deep` **and** served on localhost              |

Each virtual endpoint uses the same random-start round-robin with automatic
failover as `llmproxy__free` and `llmproxy__local`.  The legacy `llmproxy/...`
form (e.g. `llmproxy/deep/free`) is still accepted on input for backward
compatibility with pinned client configs.

```bash
# Use the deep reasoning virtual model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy__deep", "messages": [{"role": "user", "content": "Prove P≠NP"}]}'

# Inspect which backends are eligible for llmproxy__standard/free
curl http://localhost:8080/v1/models/llmproxy__standard/free | jq '._candidates'
```

Tags are configured via the `model_reasoning` field — see
[Configuration → model_reasoning](#model_reasoning) below.

### Capability-aware routing & failover

Free models vary wildly in what they support: some handle tool/function calls,
some accept images, some emit reasoning, some honor JSON-mode. llmproxy can tag
each model with the capabilities it supports (via `model_capabilities`) and use
that to route requests on **any** virtual model:

- **Proactive ordering** — when a request needs a capability, candidates that
  support it are tried **first**. This is a stable reordering: models with
  unknown capability are kept as fallbacks, so incomplete metadata never turns a
  request into a hard failure.
- **Reactive failover** — when a capability was *mandatory* but the upstream
  returned a 200 that didn't deliver it, llmproxy fails over to the next
  candidate, exactly like it does on an HTTP error. Today this covers:
  - **tools** — `tool_choice` forced a call (`"required"` or a specific
    function) but the response contained no `tool_calls`.
  - **json** — `response_format` requested JSON but the body wasn't valid JSON.

  (Reactive 200-body detection runs on **non-streaming** requests only; streaming
  responses still benefit from proactive ordering. Capabilities without a
  reliable 200 signal — **vision**, **reasoning** — rely on the upstream
  returning an HTTP error, which already triggers failover.)

The `tool_choice: "auto"` case is never treated as a failure — a model may
legitimately answer without calling a tool.

Detected capabilities: `tools`, `vision`, `reasoning`, `json`.

When at least one model is tagged, dedicated capability virtual endpoints appear:

| Virtual model name        | Selects                                                  |
|---------------------------|----------------------------------------------------------|
| `llmproxy__tools`         | All models tagged `tools`                                 |
| `llmproxy__tools/free`    | Models tagged `tools` **and** qualifying as free-tier     |
| `llmproxy__vision`        | All models tagged `vision`                                |
| `llmproxy__vision/free`   | Models tagged `vision` **and** qualifying as free-tier    |

```bash
# Route a tool-calling request only to tool-capable free models, failing
# over automatically if one returns no tool call:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy__tools/free",
       "tool_choice": "required",
       "tools": [{"type": "function", "function": {"name": "get_weather"}}],
       "messages": [{"role": "user", "content": "Weather in Paris?"}]}'

# llmproxy__free also benefits — it now orders/fails over by capability when
# the request carries tools or images.
```

Tags are configured via the `model_capabilities` field, which **auto-populates**
from the scraper (OpenRouter's `supported_parameters` / image modality) and the
setup wizard's *Manage model tags → Tag model capabilities* menu — see
[Configuration → model_capabilities](#model_capabilities) below. The legacy
`llmproxy/...` input form (e.g. `llmproxy/tools/free`) is also accepted.

### Per-provider virtual models

The reasoning, capability, and `free` families above aggregate across **all**
providers. To scope failover to a **single** provider, llmproxy also advertises
per-provider virtual models of the form:

```
llmproxy__<provider>            # cycles through ALL of that provider's models
llmproxy__<provider>/<dimension>
```

where `<dimension>` is one of `exploratory`, `standard`, `deep` (reasoning
levels), `tools`, `vision` (capabilities), or `free`. For example, with a Google
provider:

| Virtual model name              | Selects                                                |
|---------------------------------|--------------------------------------------------------|
| `llmproxy__google`              | All of Google's models                                 |
| `llmproxy__google/deep`         | Google models tagged `deep`                            |
| `llmproxy__google/standard`     | Google models tagged `standard`                        |
| `llmproxy__google/exploratory`  | Google models tagged `exploratory`                     |
| `llmproxy__google/tools`        | Google models tagged `tools`                           |
| `llmproxy__google/vision`       | Google models tagged `vision`                          |
| `llmproxy__google/free`         | Google's free-tier models (capacity-aware, like `llmproxy__free`) |

```bash
# Deep reasoning, but only ever route to Google:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy__google/deep", "messages": [{"role": "user", "content": "Prove P≠NP"}]}'

# Inspect which of Google's models back a per-provider virtual:
curl http://localhost:8080/v1/models/llmproxy__google/free | jq '._candidates'
```

Eligibility: per-provider virtuals are advertised only for providers that are
**enabled**, **non-local** (not `localhost` / `host.docker.internal` / `*.local`),
and **not** opted out via `expose_to_virtual_models: false`. Each variant appears
in `GET /v1/models` only when the provider actually has a backing model for that
dimension. `llmproxy__<provider>/free` uses the same capacity-aware load
balancing and usage tracking as `llmproxy__free`. The legacy `llmproxy/...` input
form is also accepted.

> **Precedence / naming note:** existing global virtual names always take
> precedence. If you name a provider exactly `free`, `local`, `deep`, `standard`,
> `exploratory`, `tools`, or `vision`, then that one colliding name (e.g.
> `llmproxy__standard` or `llmproxy__standard/free`) resolves to the **global**
> virtual; the provider's other per-provider variants (e.g.
> `llmproxy__standard/deep`) still work.

---

## API dialects — OpenAI **and** Anthropic, in and out

llmproxy speaks more than one API dialect on both edges. Internally everything is
normalized to the OpenAI chat/completions schema, so all routing, virtual models,
capability ordering, caching, and usage accounting work identically regardless of
which dialect a client or upstream uses.

### Inbound — what clients can speak

| Surface | Endpoints | Notes |
| --- | --- | --- |
| **OpenAI** | `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings` | The original surface. Streaming via SSE. |
| **Anthropic** | `POST /v1/messages`, `POST /v1/messages/count_tokens` | Point an Anthropic SDK at llmproxy. Streaming emits the Anthropic event format (`message_start`, `content_block_delta`, …). |
| **Gemini** | `POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent`, `:countTokens` | Point the Google GenAI SDK at llmproxy. The model id rides in the URL path; streaming emits Gemini `GenerateContentResponse` SSE chunks. |

All three surfaces accept any model id llmproxy knows — direct (`provider__model`) **and**
the virtual models (`llmproxy__free`, `llmproxy__deep`, …). So an Anthropic SDK call
with `model="llmproxy__free"` is routed and load-balanced exactly like the OpenAI path.
(xAI/Grok, Mistral, Groq, DeepSeek, etc. are OpenAI- and/or Anthropic-compatible, so they
need no separate inbound surface — use the OpenAI or Anthropic endpoints for them.)

```python
# Anthropic SDK pointed at llmproxy — works with streaming and tools
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8080", api_key="unused")
client.messages.create(model="llmproxy__free", max_tokens=256,
                       messages=[{"role": "user", "content": "hi"}])
```

### Outbound — what upstreams can speak (`protocol`)

A provider's optional `"protocol"` field selects how llmproxy talks to it:

| `protocol` | Upstream call | Auth |
| --- | --- | --- |
| `openai` (default) | `{base_url}/chat/completions` | `Authorization: Bearer` |
| `anthropic` | `{base_url}/messages` (native Messages API) | `x-api-key` + `anthropic-version` |
| `gemini` | `{base_url}/models/{model}:generateContent` (+ `:streamGenerateContent`) | `x-goog-api-key` |

This means the big providers can be added with just an API key — Anthropic (Claude) and
Google Gemini over their **native** protocols, and OpenAI plus dozens of
OpenAI-compatible gateways over the default. The `anthropic` and `gemini` provider
templates ship in the setup wizard. Translation covers text, tool definitions/calls/
results, and token usage, non-streaming and streaming, for **any inbound × upstream
combination** (e.g. an Anthropic-SDK client can stream from a Gemini upstream).

> Non-OpenAI upstreams advertise their models from `model_filter` (there is no
> OpenAI-shaped `/v1/models` to discover). Best-effort: provider-specific extras
> (Anthropic thinking/prompt-caching, Gemini safety settings) are not yet mapped.

## Configuration

Config is stored at `~/.config/llmproxy/config.json` (or the path in
`$LLMPROXY_CONFIG`, or the `--config` flag).

### Schema

```json
{
  "providers": {
    "<name>": {
      "base_url": "https://...",
      "api_key": "sk-...",
      "model_filter": ["model-a", "model-b"],

      "protocol": "openai",

      "models_url": "https://.../catalog/models",
      "models_id_field": "name",
      "models_keep_task": "Text Generation"
    }
  },
  "believed_free": [
    "openrouter/qwen/qwen3-coder:free",
    "gpt-oss-20b",
    "nvidia/meta/llama-3.1-70b-instruct"
  ],
  "model_reasoning": {
    "anthropic/claude-3.5-haiku": "exploratory",
    "anthropic/claude-sonnet-4-5": "standard",
    "anthropic/claude-opus-4": "deep",
    "openrouter/deepseek/deepseek-r1": "deep",
    "nvidia/meta/llama-3.1-70b-instruct": "standard"
  },
  "model_capabilities": {
    "openrouter/qwen/qwen3-coder:free": ["tools", "reasoning"],
    "google/gemini-2.5-flash": ["tools", "vision", "json"]
  },
  "free_limits": {
    "groq/llama-3.1-8b-instant": {
      "requests_per_minute": 30,
      "requests_per_day": 14400,
      "tokens_per_minute": 6000,
      "tokens_per_day": 500000
    }
  },

  "probe_cost": false,
  "autoremove_believed_free": false,
  "probe_frequency_days": 0,
  "update_believed_free_on_startup": false,
  "pr_providers_list": false,

  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "log_level": "INFO",
    "request_timeout": 120,
    "stream_timeout": 300,
    "response_cache_ttl": 120,
    "stream_include_usage": true
  }
}
```

`model_filter` is an optional list of upstream model IDs to allow (without the
provider prefix).  It is not set by default in `config.example.json`.  Set it to
`null` or omit it to permit all models from that provider.  It can be used as a
manual allowlist, or as a fallback model list for providers whose `/v1/models`
endpoint does not work (e.g. Cloudflare AI Gateway).

The three `models_*` keys are **optional per-provider model-discovery
overrides**, for providers that don't expose a standard OpenAI
`GET <base_url>/models`:

- **`models_url`** — fetch the model list from this exact URL instead of
  `<base_url>/models`. Use it when the catalog lives at a different path than
  the chat endpoint. For example, GitHub Models serves chat at
  `https://models.github.ai/inference/chat/completions` but its catalog at
  `https://models.github.ai/catalog/models`, and Cloudflare Workers AI has no
  `GET /v1/models` — its catalog is at
  `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search`.
- **`models_id_field`** — the field on each returned model object that holds the
  usable upstream id (default `"id"`). Cloudflare's `models/search` puts the
  `@cf/...` id in `"name"` and reserves `"id"` for an internal UUID, so set this
  to `"name"`.
- **`models_keep_task`** — when set, keep only models whose `task.name` matches
  (case-insensitive). Cloudflare's catalog mixes Text Generation, embeddings,
  and image tasks in one list; set this to `"Text Generation"` to keep only
  chat-capable models.

These overrides are part of the provider templates in
[`llmproxy/providers.json`](llmproxy/providers.json), so the setup wizard writes
them automatically when you add GitHub Models or Cloudflare Workers AI (with the
`{account_id}` placeholder substituted into `models_url`).

`believed_free` is an **optional** top-level array of model names that the
`free` virtual model should include even when their ID doesn't contain the
word `free`.  Omit the field entirely (or set it to `[]`) to keep the
default behaviour — only IDs that literally contain `free` are pulled in.
Each entry is matched (case-insensitively) against either the upstream
model ID (e.g. `gpt-oss-20b`) or the full proxy ID (e.g.
`openrouter/qwen/qwen3-coder:free`).  The setup wizard manages this field
via its "Manage model tags" menu and via the per-provider auto-populate step
when you add a templated provider; the merged defaults come from
[`llmproxy/providers.json`](llmproxy/providers.json).

> **Free-tier accuracy:** The `believed_free` entries in `config.example.json` and in
> `llmproxy/providers.json` are best-effort estimates based on publicly-stated provider
> free tiers. Provider offerings change without notice — no guarantee is made as to accuracy.
> Verify directly with each provider before relying on free availability in production. The
> [`scripts/update_free_models.py`](#keeping-the-free-models-list-current) scraper exists to
> keep these entries current.

<a name="model_reasoning"></a>
`model_reasoning` is an **optional** top-level object that tags individual
models with a reasoning level.  Valid levels are `exploratory`, `standard`,
and `deep`.  Each key is matched (case-insensitively) against either the
upstream model ID (e.g. `anthropic/claude-opus-4`) or the full
`provider/upstream_model` proxy ID (e.g.
`openrouter/anthropic/claude-opus-4`).  When a level has at least one tagged
model in the route cache, the corresponding virtual endpoint is advertised in
`GET /v1/models`.  Omit the field entirely (or set it to `{}`) to disable
reasoning-level routing.  The setup wizard manages this field via its
"Manage model tags" menu (merged defaults come from `llmproxy/providers.json`).

<a name="model_capabilities"></a>
`model_capabilities` is an **optional** top-level object that tags individual
models with the capabilities they support.  Valid values are `tools`, `vision`,
`reasoning`, and `json` (a list per model).  Keys are matched
(case-insensitively) against either the upstream model ID or the full
`provider/upstream_model` proxy ID, like `model_reasoning`.  It drives
[capability-aware routing & failover](#capability-aware-routing--failover) on
all virtual models and powers the `llmproxy__tools` / `llmproxy__vision`
endpoints (advertised when at least one model carries the tag).  Omit it (or set
it to `{}`) to disable capability-aware behavior — the proxy then behaves exactly
as before.  The field **auto-populates** from the scraper (OpenRouter's
`supported_parameters` and image input modality) and from the setup wizard's
"Manage model tags → Tag model capabilities" menu.

See `config.example.json` for a complete annotated example.

<a name="free_limits"></a>
### `free_limits` — capacity-aware free-tier load balancing

`free_limits` is an **optional** top-level object mapping a
`provider/upstream_model` key (lowercased) to that model's free-tier quota:

```json
"free_limits": {
  "groq/llama-3.1-8b-instant": {
    "requests_per_minute": 30,
    "requests_per_day": 14400,
    "tokens_per_minute": 6000,
    "tokens_per_day": 500000
  }
}
```

When a `…/free` virtual model (`llmproxy__free`, `llmproxy__<level>/free`,
`llmproxy__<provider>/free`, …) picks which backend to use, it scores each
candidate by how much of its quota is still unused and prefers the one with the
most headroom (weighted random, so load is still spread). **Both** request
limits (`requests_per_minute` / `requests_per_day`) **and** token limits
(`tokens_per_minute` / `tokens_per_day`) are now enforced — a model that has
burned through its per-minute token budget is scored down and skipped just like
one that hit its request cap, which keeps traffic inside the free tier for
providers that meter by tokens. Any field set to `null` is ignored. Counters are
in-memory and **per worker process** (see the note on multi-worker below).

<a name="usage-accounting"></a>
### Token + cost accounting — `GET /v1/usage`

The proxy tracks tokens and dollar cost for every request it serves and exposes
them on a read-only endpoint:

```bash
curl http://localhost:8080/v1/usage | jq
```

```json
{
  "object": "usage.report",
  "since": "2026-06-13T10:00:00+00:00",
  "models": [
    {
      "model": "groq/llama-3.1-8b-instant",
      "requests": 412,
      "prompt_tokens": 50231, "completion_tokens": 18044, "total_tokens": 68275,
      "tokens_last_60s": 1203, "tokens_today": 68275,
      "cost": 0.0, "cost_currency": "USD",
      "cost_sources": {"provider": 0, "computed": 412, "unknown": 0},
      "believed_free": true,
      "unexpected_cost": false
    }
  ],
  "totals": {"requests": 412, "prompt_tokens": 50231, "completion_tokens": 18044,
             "total_tokens": 68275, "cost": 0.0},
  "flagged_paid_free_models": []
}
```

- **Token counts** come from the upstream `usage` block of each response
  (streaming included — the proxy asks for a final usage chunk via
  `stream_options.include_usage`; disable with `server.stream_include_usage:
  false` if an upstream rejects it).
- **Cost** is *hybrid*: the provider's own `usage.cost` is used when present
  (e.g. OpenRouter, Vercel AI Gateway); otherwise it is computed from a
  per-token `pricing` snapshot bundled into `llmproxy/providers.json` by the
  scraper (`cost_sources` tells you which was used for how many requests).
- **`flagged_paid_free_models`** lists any model in `believed_free` that served
  a request reporting a **non-zero** cost. These are surfaced for review only —
  the proxy never edits `believed_free` at runtime. Use this to spot a model
  that has quietly left its free tier.

`POST /v1/usage/reset` clears the counters for the current worker; it is gated by
the same auth policy as the [admin API](#security--localhost-only-by-default)
(loopback-only unless an admin token is set).

> **Per-worker accounting.** Like the load-balancer counters, usage/cost is
> in-memory and per worker process. Under a multi-worker gunicorn deployment each
> worker reports only the requests it served, and the totals reset on restart.
> Run a single worker if you need one consolidated view.

<a name="cost-flags"></a>
### Verifying free models are actually free

Two opt-in, top-level config flags (both default `false`) let you keep
`believed_free` honest:

```json
{
  "probe_cost": false,
  "autoremove_believed_free": false,
  "probe_frequency_days": 0
}
```

- **`probe_cost`** — when `true`, [`scripts/update_free_models.py`](#keeping-the-free-models-list-current)
  actively probes each `believed_free` model with a tiny real chat request
  (`max_tokens: 1`) using your configured API keys, inspects the returned
  `usage`/cost, and flags any model that reports a cost. This spends a small
  amount of quota, so it is off by default. (You can also trigger it for a single
  run with the `--probe` flag.)
- **`autoremove_believed_free`** — when `true`, the scraper **removes** any
  model that probes (or prices) as non-free from `believed_free` (and therefore
  from the `/free` virtual model). When `false` (default), such models are
  reported in the scraper output and in `flagged_paid_free_models`, but left in
  place for you to review.
- **`probe_frequency_days`** — throttles the probe so it runs at most once every
  _N_ days, which matters when `probe_cost` is combined with
  `update_believed_free_on_startup` (otherwise every server boot would spend
  quota). `0` (default) probes on every run; `1` is at most once a day, `7` once a
  week, etc. The last-run timestamp is cached in `probe_state.json` next to your
  `config.json` (not in `config.json` itself). The throttle applies to both
  `probe_cost: true` and the `--probe` flag; pass `--ignore-throttle` to
  `update_free_models.py` to force a probe regardless of how recently one ran.

<a name="update-on-startup"></a>
### Running the updater on startup — `update_believed_free_on_startup`

Set the top-level flag to refresh free-tier data automatically when the server
boots:

```json
{ "update_believed_free_on_startup": true }
```

When `true`, the server runs `scripts/update_free_models.py` once per worker in a
background thread at startup (it never blocks request handling). It:

- **rewrites `llmproxy/providers.json`** (the sidecar) with any `believed_free` /
  `free_limits` / `pricing` changes, and regenerates `config.example.json`, and
- **syncs your `config.json`** (`believed_free` / `free_limits` / `model_reasoning`),
  which the proxy picks up via the normal config hot-reload.

Every line the updater prints — including `Updated …/providers.json`, each
`believed_free` add/remove, and `Synced free-tier sections into …` — is re-emitted
through the server log with a `[startup-update]` prefix (at `INFO` level), so set
`server.log_level` to `INFO` to watch it work. If `probe_cost` is also `true`, the
startup run includes the active cost probe (and, with `autoremove_believed_free`,
removes any model it finds is no longer free). Defaults to `false`.

> The scraper lives in the repo-root `scripts/` package. The Docker image ships
> it, and the server adds its parent directory to `sys.path` so the import works
> under gunicorn. If a slimmed-down deployment omits `scripts/`, the server logs
> `[startup-update] updater unavailable …` and skips the update. The sidecar
> rewrite is **ephemeral in a container** (it lives in the image layer) — the
> durable effect is the `config.json` sync on your mounted volume. To land sidecar
> changes back in the repo, use the [CI auto-update workflow](#automated-providersjson-updates-ci).

<a name="pr-providers-list"></a>
### Proposing `providers.json` changes as a PR — `pr_providers_list`

When `update_believed_free_on_startup` refreshes the sidecar (optionally with
probing, if `probe_cost` / `autoremove_believed_free` are on), set this flag to
have the running deployment open a **pull request** with the result instead of
only keeping the change in its ephemeral local copy:

```json
{
  "update_believed_free_on_startup": true,
  "pr_providers_list": true,
  "pr_providers_repo": "BillJr99/llmproxy"
}
```

When `true` **and** the startup run actually changed `providers.json`, the server
pushes `llmproxy/providers.json` + `config.example.json` to a branch and opens (or
refreshes) a PR against the base branch — using the GitHub API directly, so it
**never touches a local git checkout** and works even in a container with no
`.git`. It logs `[providers-pr] opening PR …` and the resulting PR URL.

Required / optional settings (all top-level):

| Key | Required | Default | Meaning |
|-----|----------|---------|---------|
| `pr_providers_list` | — | `false` | Master switch. |
| `pr_providers_repo` | **yes** | — | Target repo as `"owner/repo"`. |
| `pr_providers_token` | yes¹ | — | GitHub token; may be a `${VAR}` ref. ¹Falls back to the `GITHUB_TOKEN` / `GH_TOKEN` environment variables. Needs `contents:write` + `pull_requests:write`. |
| `pr_providers_base` | — | `"main"` | Base branch for the PR. |
| `pr_providers_branch` | — | `"llmproxy-auto/providers"` | Head branch (force-updated each run; an open PR for it is reused). |

If the token or `pr_providers_repo` is missing, the server logs a `[providers-pr]`
warning and skips — it never fails the startup update. This is the **deployment**
counterpart to the repo-level
[CI auto-update workflow](#automated-providersjson-updates-ci): the workflow
proposes PRs from a scheduled scrape, while `pr_providers_list` proposes them from
a live deployment (which can additionally probe real model costs).

### Provider templates

Provider templates and free-tier metadata both live in
[`llmproxy/providers.json`](llmproxy/providers.json) — the single source of
truth. The setup wizard reads from this file at startup; `config.example.json`
is regenerated from the same file. To add or update a provider, edit
`providers.json` directly (or run the scraper — see
[Keeping the free-models list current](#keeping-the-free-models-list-current)).

The wizard currently offers ready-made templates for these providers:

| Provider                          | Default key      | Base URL                                                   |
|-----------------------------------|------------------|-----------------------------------------------------------|
| Nous Research (Hermes)            | `nous`           | `https://inference-api.nousresearch.com/v1`                |
| Nvidia NIM                        | `nvidia`         | `https://integrate.api.nvidia.com/v1`                      |
| Google Gemini (OpenAI-compat)     | `google`         | `https://generativelanguage.googleapis.com/v1beta/openai`  |
| Cerebras                          | `cerebras`       | `https://api.cerebras.ai/v1`                               |
| GitHub Models                     | `github`         | `https://models.github.ai/inference`                       |
| SambaNova Cloud                   | `sambanova`      | `https://api.sambanova.ai/v1`                              |
| Mistral AI                        | `mistral`        | `https://api.mistral.ai/v1`                                |
| Groq                              | `groq`           | `https://api.groq.com/openai/v1`                           |
| Together AI                       | `together`       | `https://api.together.xyz/v1`                              |
| Fireworks AI                      | `fireworks`      | `https://api.fireworks.ai/inference/v1`                    |
| Cloudflare Workers AI             | `cloudflare-workers` | `https://api.cloudflare.com/client/v4/accounts/.../ai/v1`  |
| Zhipu AI (BigModel)               | `zhipu`          | `https://open.bigmodel.cn/api/paas/v4`                     |
| Z.AI                              | `z-ai`           | `https://api.z.ai/api/paas/v4`                             |
| Cohere                            | `cohere`         | `https://api.cohere.com/compatibility/v1`                  |
| DeepSeek                          | `deepseek`       | `https://api.deepseek.com/v1`                              |
| OpenRouter                        | `openrouter`     | `https://openrouter.ai/api/v1`                             |
| Ollama Cloud                      | `ollama-cloud`   | `https://ollama.com/v1`                                    |
| Moonshot AI (Kimi)                | `moonshot`       | `https://api.moonshot.ai/v1`                               |
| MiniMax                           | `minimax`        | `https://api.minimax.io/v1`                                |
| Hugging Face Inference            | `huggingface`    | `https://router.huggingface.co/v1`                         |
| xAI (Grok)                        | `xai`            | `https://api.x.ai/v1`                                      |
| Cloudflare AI Gateway             | `cloudflare-ai-gateway` | `https://gateway.ai.cloudflare.com/v1/{account}/{gw}/workers-ai/v1` |
| Vercel AI Gateway                 | `vercel`         | `https://ai-gateway.vercel.sh/v1`                          |
| Venice AI                         | `venice`         | `https://api.venice.ai/api/v1`                             |
| OpenCode Zen (free gateway)       | `opencode-zen`   | `https://opencode.ai/zen/v1`                               |

> **API key required.** Every provider in this table requires an API key. The setup wizard displays a hint showing where to obtain each key. For keyless local access (e.g. a local Ollama instance), use the manual "Add / edit a provider" option in the wizard.

Any OpenAI-compatible provider can also be added manually via the "Add / edit a
provider (manual)" menu option.

> **Providers that do not support a standard `GET <base_url>/models` (as of June 2026)**
> Some providers return an error or non-JSON response for the default `/models`
> path. There are two ways to handle these:
>
> 1. **Point discovery at the real catalog** with the `models_url` /
>    `models_id_field` / `models_keep_task` overrides (see
>    [Schema](#schema)). The bundled templates for **GitHub Models** and
>    **Cloudflare Workers AI** already do this, so their models are discovered
>    live.
> 2. **Synthesize from `model_filter`** — when no working catalog endpoint
>    exists, set `model_filter` to the upstream ids you want and llmproxy
>    advertises those when the `/models` fetch fails.
>
> | Provider | Default `/models` symptom | Handling |
> |---|---|---|
> | **GitHub Models** | HTTP 404 — catalog is at `/catalog/models`, not `/inference/models` | `models_url` → `https://models.github.ai/catalog/models` |
> | **Cloudflare Workers AI** | HTTP 405 — no `GET /v1/models` | `models_url` → `…/ai/models/search`, `models_id_field: "name"`, `models_keep_task: "Text Generation"` |
> | **Cloudflare AI Gateway** | HTTP 401 — gateway proxies inference only, no catalog | `model_filter` (synthesized); a 401 also means the API token is missing/under-scoped for Workers AI |
> | **Hugging Face Inference** | Returns HTML rather than JSON for `/v1/models` | `model_filter` (synthesized) |
> | **Open WebUI** (self-hosted, e.g. behind a custom domain) | HTTP 200 but HTML — the OpenAI API lives under `/api` | set `base_url` to `https://<host>/api` |

---

## Web admin UI

Everything the setup wizard configures — server settings, providers (add / edit /
delete, add-from-template, live model discovery), the model categorizations that
drive the virtual endpoints (`believed_free`, `model_reasoning`,
`model_capabilities`, `free_limits`), and a derived preview of the virtual
endpoints — can also be managed from a web frontend served by the running proxy
at **`/admin`** (same host and port as the API):

```
http://localhost:8080/admin
```

The UI is a self-contained single page (no build step, no external assets) and
writes straight to `config.json` via a JSON API under `/admin/api/*`. Changes
take effect without a restart (host/port changes excepted), because every worker
re-reads the config file when it changes.

### Security — localhost-only by default

The admin **API** edits secrets, so it is locked down by default:

* **No token configured (default):** `/admin/api/*` answers **only loopback
  requests** (`127.0.0.1` / `::1`). The UI shell at `/admin` is still served (it
  carries no secrets), but the data API refuses non-local callers.
* **Token configured:** any origin that presents the token is allowed. Set it via
  the `LLMPROXY_ADMIN_TOKEN` environment variable or `config["admin"]["token"]`
  (which may itself be a `${VAR}` reference). The UI prompts for the token and
  sends it as `Authorization: Bearer <token>` (or `X-Admin-Token`).

API responses never return plaintext keys — literal secrets are masked
(`sk-…1234`) while `${VAR}` references are shown verbatim (they are not secret).
Submitting a blank API-key field leaves the stored key unchanged.

Disable the UI entirely with `--no-admin` (or `config["admin"]["enabled"]: false`);
force-enable with `--admin`. When the server binds a non-loopback host with no
token set, startup logs a warning that remote admin access will be refused.

```jsonc
"admin": {
  "enabled": true,
  "token": "${LLMPROXY_ADMIN_TOKEN}"   // optional; unset ⇒ loopback-only
}
```

## Environment-variable references

The provider `api_key` and `base_url` fields (and the admin `token`) may contain
`${VAR}` references that are resolved from the process environment **at request
time** — so secrets never need to be written literally into `config.json`:

```jsonc
"providers": {
  "openai": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "${OPENAI_API_KEY}"
  },
  "ollama": {
    "base_url": "http://${OLLAMA_HOST}:11434/v1"
  }
}
```

An unset variable resolves to the empty string. This is ideal for Docker / cloud
deployments: pass `-e OPENAI_API_KEY=…` to the container and keep the bind-mounted
`config.json` free of credentials. The stored config keeps the raw `${VAR}` text
(the admin UI and setup wizard show and edit the reference, not the resolved
value); only outbound upstream requests see the resolved secret.

---

## Keeping the free-models list current

Provider free tiers change without notice. The free-tier fields in
[`llmproxy/providers.json`](llmproxy/providers.json) hold the
project's best-effort view of *which* models are currently free and *what*
their rate limits are — used by the `llmproxy__free` virtual endpoint and by the
setup wizard's "auto-populate" step.

A scraper at `scripts/update_free_models.py` polls multiple sources, diffs the
result against the sidecar, and prints proposed adds / removes / limit changes
for human review.

### Sources

| Source       | Confidence | What it does |
|--------------|------------|--------------|
| `openrouter` | high       | Hits `https://openrouter.ai/api/v1/models` and flags any model with `pricing.prompt == 0` as free. |
| `docs`       | high       | Per-provider HTML scrapers for published rate-limit / free-tier pages (Google, Groq, Cerebras, Mistral, Cohere). Add more under `scripts/sources/docs/`. |
| `api`        | medium     | Calls each provider's OpenAI-compatible `/v1/models` endpoint when `<PROVIDER>_API_KEY` is set in your environment. Used to detect *removals* (a believed-free model that's no longer listed). |
| `litellm_cost_map` | medium | Reads the public [litellm](https://github.com/BerriAI/litellm) pricing map: flags zero-priced models as free **and** snapshots per-token prices for paid ones into the sidecar `pricing` block (used by the proxy to cost tokens offline — see [Token + cost accounting](#usage-accounting)). |
| `community`  | low        | Pulls the [tashfeenahmed/freellmapi](https://github.com/tashfeenahmed/freellmapi) community list as a sanity signal. |
| `probe`      | high · **opt-in** | Sends a tiny real chat request to each `believed_free` model and flags any that report a cost. Off by default; enable with `probe_cost: true` in `config.json` or the `--probe` flag. Spends a little quota. |

### Usage

```bash
# Preview proposed changes (no files written)
python scripts/update_free_models.py --dry-run

# Apply the changes to llmproxy/providers.json and regenerate config.example.json
python scripts/update_free_models.py

# Restrict to one provider
python scripts/update_free_models.py --provider google --dry-run

# Restrict to specific sources
python scripts/update_free_models.py --source openrouter,docs --dry-run

# Just regenerate config.example.json from the current sidecar (no scraping)
python scripts/update_free_models.py --regen-config-only

# Also sync your live config.json's free-tier sections from the sidecar
python scripts/update_free_models.py --config ~/.config/llmproxy/config.json --dry-run
python scripts/update_free_models.py --config ~/.config/llmproxy/config.json

# Sync the config from the current sidecar without scraping
python scripts/update_free_models.py --regen-config-only --config ~/.config/llmproxy/config.json

# Actively probe believed_free models for cost (real requests; needs API keys).
# Equivalent to setting "probe_cost": true in config.json.
python scripts/update_free_models.py --probe --config ~/.config/llmproxy/config.json --dry-run
python scripts/update_free_models.py --probe --probe-max 20 --probe-provider groq
```

### Verifying free tiers and auto-removal (`probe_cost` / `autoremove_believed_free`)

By default the scraper only *adds* high-confidence free models and *removes* ones
that a trusted source contradicts. Two `config.json` flags extend this to
empirical cost checks (see [Verifying free models are actually free](#cost-flags)):

- **`probe_cost: true`** (or `--probe`) runs the `probe` source — a real
  `max_tokens: 1` request to every `believed_free` model that has a configured
  API key — and flags any that report a non-zero cost.
- **`autoremove_believed_free: true`** lets those probe-flagged (and otherwise
  non-free) models be removed from `believed_free` automatically. When `false`
  (default), the run prints the flagged models but makes no removal.

You can also have the server run this updater on boot — see
[`update_believed_free_on_startup`](#update-on-startup).

### Syncing your live config (`--config PATH`)

The proxy reads `believed_free` / `model_reasoning` / `model_capabilities` /
`free_limits` at runtime from *your* `config.json`, not from the sidecar. Pass
`--config PATH` to also reconcile a live config in the same run (honors
`--dry-run`):

- **Scope is limited to providers configured in that file.** Entries for custom
  providers, or sidecar providers you haven't configured, are left untouched —
  as are non-model keys like the `_note` in `free_limits`.
- **`believed_free` and `free_limits` are synced** — newly-free models are added
  and models that are no longer free are removed.
- **`model_reasoning` and `model_capabilities` are add-only.** Existing tags are
  never pruned or overwritten, so a model keeps its reasoning level / capability
  tags (including any you set by hand) even after it leaves the free tier.
- Your `providers`, `server`, and any other config sections are preserved; only
  the free-tier sections change.

### Safety properties

- **A failed source never causes a removal.** Sources run independently; any
  source that errors out (network failure, parse error, 5xx) emits no
  evidence rather than "every model is absent". The scraper prints which
  sources succeeded so you can judge how much to trust the diff.
- **`/v1/models` presence ≠ free.** The `api` source only contributes
  *existence* evidence; it can flag removals but cannot decide that a model
  is free.
- **Reasoning levels are preserved.** Existing `model_reasoning` entries are
  never overwritten. New models are tagged via
  `infer_reasoning_level()` (deep keywords → deep; size in B → standard /
  exploratory) so you can hand-tune later.

### Optional environment variables

When set, each `<PROVIDER>_API_KEY` enables the `api` source for that provider:

```
GROQ_API_KEY=gsk-...        GOOGLE_API_KEY=AIza-...
CEREBRAS_API_KEY=csk-...    MISTRAL_API_KEY=...
COHERE_API_KEY=...          SAMBANOVA_API_KEY=...
```
(and so on — uppercase the provider key, replace `-` with `_`, append `_API_KEY`).

<a name="automated-providersjson-updates-ci"></a>
### Automated `providers.json` updates (CI → PR)

A scheduled GitHub Actions workflow,
[`.github/workflows/update-providers.yml`](.github/workflows/update-providers.yml),
keeps the sidecar current **in the repository** without anyone running the
scraper by hand. Once a week (and on demand via the Actions tab) it:

1. runs `python scripts/update_free_models.py` with the default, read-only
   sources — provider docs, `/models` catalogs, OpenRouter, the litellm cost
   map, and the community list. It **does not** run the opt-in `probe` source,
   so **no real model requests / quota are spent**;
2. regenerates `config.example.json`; and
3. if `llmproxy/providers.json` or `config.example.json` changed, opens (or
   updates) a pull request against `main` on the `chore/update-providers`
   branch — using [`peter-evans/create-pull-request`](https://github.com/peter-evans/create-pull-request).
   When nothing changed, no PR is created. The run logs the `git status` diff
   and the action logs whether a PR was opened.

**Enabling / disabling.** A GitHub Action can't read your deployment's private
`config.json`, so the on/off switch is a **repository variable** rather than a
config flag: set `PROVIDERS_AUTOUPDATE` to `false` under *Settings → Secrets and
variables → Actions → Variables* to disable the scheduled run (it is treated as
enabled unless explicitly `false`). Manual `workflow_dispatch` runs always
execute. The workflow needs `contents: write` and `pull-requests: write`
permissions (already declared in the file); if your org disables PR creation by
`GITHUB_TOKEN`, enable it under *Settings → Actions → General → Workflow
permissions*.

> This repo-level workflow and the server-side
> [`update_believed_free_on_startup`](#update-on-startup) flag are complementary:
> the workflow lands durable updates in the repo via reviewable PRs, while the
> startup flag refreshes a running deployment's live config.

---

## Quick start — local, no install

This is the recommended path for local use.  You only need `flask` and
`requests`; no `pip install .` or `pip install -e .` is required.

### 1. Install dependencies

```bash
pip install flask requests
```

`gunicorn` is optional.  If installed, the server uses it automatically for
better concurrency; otherwise it falls back to the Flask development server,
which is fine for local use.

```bash
pip install gunicorn   # optional
```

### 2. Configure providers

Run the interactive setup wizard.  It creates `~/.config/llmproxy/config.json`
and prompts you for each provider's name, base URL, API key, and optional model
filter.

```bash
python run.py --setup
```

You can re-run `--setup` at any time to add, edit, or remove providers.

### 3. Start the server

```bash
python run.py
```

The server binds to `0.0.0.0:8080` by default.  Override host or port without
editing the config:

```bash
python run.py --port 9000 --log-level DEBUG
```

`run.py` resolves its own location via `os.path.abspath(__file__)`, so it works
correctly regardless of which directory you invoke it from:

```bash
python /path/to/llmproxy/run.py --setup
python /path/to/llmproxy/run.py
```

### 4. Reconfigure at any time

```bash
python run.py --setup
```

The server hot-reloads config on each request (modification-time cache), so
provider changes take effect immediately without a restart.  Only `host` or
`port` changes require a restart.

---

## Tests, dev tooling, and CI

The repo has three distinct things named "test"-ish — each does something
different:

| File                       | What it is                                                                     |
|----------------------------|--------------------------------------------------------------------------------|
| `tests/`                   | The pytest unit/integration suite (run with `pytest`). New as of this release. |
| `llmproxy_test_client.py`  | Live integration test client. Talks to a running llmproxy over HTTP.           |
| `test_tui.py`              | Interactive chat TUI for hand-driving the proxy (despite the misleading name). |

### Running the unit suite

```bash
pip install -r requirements-dev.txt
pytest                                  # run everything
pytest --cov=llmproxy --cov=scripts     # with coverage
pytest tests/test_scraper                # just the scraper tests
ruff check llmproxy scripts tests        # lint
```

CI runs the same checks on every push and pull request — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). It runs:
- `pytest` across Python 3.11 and 3.12,
- `ruff` lint,
- a guard that fails the build if `config.example.json` has drifted from
  `llmproxy/providers.json` (regenerate locally with
  `python scripts/update_free_models.py --regen-config-only`).

---

## Live integration test client

`llmproxy_test_client.py` is a standalone script with no dependencies beyond
`requests`.  It connects to a running llmproxy instance and exercises all
endpoints, printing a pass/fail/skip report.

### Basic usage

```bash
# Run all test suites against the default localhost:8080
python llmproxy_test_client.py

# Target a different host or port
python llmproxy_test_client.py --base-url http://localhost:9000/v1

# Force a specific model for chat/embedding/streaming tests
python llmproxy_test_client.py --model openrouter/openrouter/free

# Run only the structural tests (no live LLM calls required)
python llmproxy_test_client.py --suite health --suite errors

# Skip streaming (useful in environments that buffer SSE)
python llmproxy_test_client.py --no-stream

# Include OpenAI SDK compatibility test (requires: pip install openai)
python llmproxy_test_client.py --use-sdk
```

### Test suites

| Suite       | What it checks                                                        | Needs provider?  |
|-------------|-----------------------------------------------------------------------|------------------|
| `health`    | `GET /health` returns 200 and lists active providers                  | No               |
| `errors`    | Missing model field, bad prefix, unknown provider, non-JSON body      | No               |
| `models`    | `GET /v1/models` aggregates all providers; naming convention          | Yes              |
| `free`      | Sends several prompts to `model="llmproxy__free"`; tests cycling + streaming   | Yes (free tier)  |
| `local`     | Sends several prompts to `model="llmproxy__local"`; skipped if none configured | Yes (localhost)  |
| `chat`      | Non-streaming chat completion; checks response content                | Yes              |
| `streaming` | Streaming SSE chat; prints tokens live as they arrive                 | Yes              |
| `embeddings`| Embedding request; accepts graceful 400/404 if unsupported            | Yes              |
| `sdk`       | Same chat + stream tests via the `openai` Python package              | Yes              |

When no `--model` flag is given, the client auto-selects a model from the
proxy's `/v1/models` list, preferring names that suggest a free or small model
(`free`, `mini`, `flash`, `haiku`, `small`, `8b`, etc.).

### Example output (no providers configured)

```
llmproxy test client
Target: http://localhost:8080/v1
───────────────────────────────────────────────────────

══ Health Check ══
  ✓ GET /health returns 200  providers=[]
  No providers configured yet. Run: python run.py --setup

══ Error Handling ══
  ✓ Missing 'model' field → 400
  ✓ Non-prefixed model string → 400
  ✓ Unknown provider → 404
  ✓ Non-JSON body → 400
  ✓ GET /health JSON schema contains 'status'

───────────────────────────────────────────────────────
Results:  6 passed  0 failed  1 skipped  / 7 total
```

---

## Installation via pip (optional)

If you prefer a system-wide `llmproxy` command, install the package:

```bash
pip install -e .        # editable install (recommended for development)
# or
pip install .
```

After installation, `run.py` is no longer needed; use the `llmproxy` command
directly:

```bash
llmproxy --setup
llmproxy
llmproxy --port 9000 --log-level DEBUG
llmproxy --list-providers
llmproxy --version
```

---

## Docker

### Build the image

```bash
docker build -t llmproxy .
```

Or pull from GHCR (see [GHCR — hosting and pulling](#ghcr--hosting-and-pulling)):

```bash
docker pull ghcr.io/billjr99/llmproxy:latest
```

### First-time setup

Config is bind-mounted from `~/.config/llmproxy` on the host.  The image runs
as a non-root user by default (no `--user` required); passing
`--user $(id -u):$(id -g)` makes files created inside the container owned by
you on the host.

```bash
mkdir -p ~/.config/llmproxy

docker run -it --rm \
  --user $(id -u):$(id -g) \
  -v ~/.config/llmproxy:/config \
  -e LLMPROXY_CONFIG=/config/config.json \
  llmproxy --setup
```

### Start the server

```bash
docker run -d \
  -p 8080:8080 \
  --user $(id -u):$(id -g) \
  -v ~/.config/llmproxy:/config \
  -e LLMPROXY_CONFIG=/config/config.json \
  --name llmproxy \
  llmproxy
```

The [web admin UI](#web-admin-ui) is available on the same published port at
`http://localhost:8080/admin`. Because the container binds `0.0.0.0`, set an
admin token to allow access (the API otherwise serves loopback only), and use
`${VAR}` references in `config.json` to keep credentials in the environment
rather than the bind-mounted file:

```bash
docker run -d \
  -p 8080:8080 \
  --user $(id -u):$(id -g) \
  -v ~/.config/llmproxy:/config \
  -e LLMPROXY_CONFIG=/config/config.json \
  -e LLMPROXY_ADMIN_TOKEN=choose-a-strong-token \
  -e OPENAI_API_KEY=sk-… \
  --name llmproxy \
  llmproxy
```

### Reconfigure without stopping the server

```bash
docker run -it --rm \
  --user $(id -u):$(id -g) \
  -v ~/.config/llmproxy:/config \
  -e LLMPROXY_CONFIG=/config/config.json \
  llmproxy --setup

# Restart only if host or port changed; hot-reload handles everything else
docker restart llmproxy
```

### Connecting to a local provider from inside the container

When llmproxy runs in Docker but you want it to talk to a local provider like
Ollama running on the host (or in a sibling container), `localhost` inside the
container points to the container itself — not to your host. You have three
options; pick whichever fits your setup.

**Option A — `host.docker.internal` (recommended for Docker Desktop)**

Change the provider's `base_url` from `http://localhost:11434/v1` to
`http://host.docker.internal:11434/v1`. llmproxy already treats
`host.docker.internal` and `gateway.docker.internal` as local for the purposes
of `llmproxy__local` routing, so the `__local` virtual model picks it up
automatically.

On plain Linux (no Docker Desktop), `host.docker.internal` doesn't resolve by
default — add it explicitly:

```bash
docker run --add-host=host.docker.internal:host-gateway ... llmproxy
```

…or in `docker-compose.yml`:

```yaml
services:
  llmproxy:
    # ...
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

**Option B — host networking**

Start llmproxy with `--network=host` and keep the original
`http://localhost:11434/v1` config. Simplest on Linux; not available on
Docker Desktop.

### Named-volume alternative

If you prefer to keep the config entirely inside Docker (useful for CI or
rootless environments where a host-path mount is inconvenient), mount the
named volume over the default config location under the non-root user's home
(`/home/llmproxy/.config/llmproxy`):

```bash
# Setup
docker run -it --rm \
  -v llmproxy_config:/home/llmproxy/.config/llmproxy \
  llmproxy --setup

# Server
docker run -d \
  -p 8080:8080 \
  -v llmproxy_config:/home/llmproxy/.config/llmproxy \
  --name llmproxy \
  llmproxy
```

---

## docker-compose

The `docker-compose.yml` uses a bind mount from `~/.config/llmproxy` on the
host and runs containers as the current user.  Create a `.env` file first so
Compose picks up your UID/GID:

```bash
printf "UID=%s\nGID=%s\n" "$(id -u)" "$(id -g)" > .env
mkdir -p ~/.config/llmproxy
```

```bash
# Build and start the server (detached)
docker-compose up -d

# First-time setup or reconfigure (interactive)
docker-compose run --rm setup

# Restart to apply host/port changes
docker-compose restart llmproxy

# View logs
docker-compose logs -f llmproxy

# Stop and remove containers (host config directory is preserved)
docker-compose down
```

---

## GHCR — hosting and pulling

### Publish your own image

The included GitHub Actions workflow (`.github/workflows/docker-publish.yml`)
automatically builds and pushes the image to
[GitHub Container Registry (GHCR)](https://ghcr.io) on every push to `main`
and on every version tag (`v*`).  It uses `GITHUB_TOKEN`, so no extra secrets
or personal access tokens are needed.

To enable it, fork or push the repo to GitHub — the workflow runs automatically.
Images are published to:

```
ghcr.io/<your-github-username>/llmproxy
```

For this repository: `ghcr.io/billjr99/llmproxy`.

**Tags produced:**

| Event | Tags |
|-------|------|
| Push to `main` | `main`, `latest` |
| Push tag `v1.2.3` | `1.2.3`, `1.2`, `latest` |

### Pull and run

```bash
docker pull ghcr.io/billjr99/llmproxy:latest

mkdir -p ~/.config/llmproxy

# First-time setup
docker run -it --rm \
  --user $(id -u):$(id -g) \
  -v ~/.config/llmproxy:/config \
  -e LLMPROXY_CONFIG=/config/config.json \
  ghcr.io/billjr99/llmproxy:latest --setup

# Start the server
docker run -d \
  -p 8080:8080 \
  --user $(id -u):$(id -g) \
  -v ~/.config/llmproxy:/config \
  -e LLMPROXY_CONFIG=/config/config.json \
  --name llmproxy \
  ghcr.io/billjr99/llmproxy:latest
```

### Use in docker-compose

To use the GHCR image instead of building locally, replace `build: .` in
`docker-compose.yml` with:

```yaml
image: ghcr.io/billjr99/llmproxy:latest
```

---

## API endpoints

All endpoints mirror the OpenAI API.

| Method | Path                    | Description                               |
|--------|-------------------------|-------------------------------------------|
| GET    | `/health`               | Health check; returns provider list       |
| GET    | `/version`              | Returns the running llmproxy version      |
| GET    | `/v1/models`            | Aggregate model list from all providers   |
| GET    | `/v1/models/<model_id>` | Single model lookup                       |
| POST   | `/v1/chat/completions`  | Chat completions (streaming supported)    |
| POST   | `/v1/completions`       | Legacy text completions                   |
| POST   | `/v1/embeddings`        | Embeddings                                |
| *      | `/v1/<anything>`        | Pass-through to upstream (see note below) |

For pass-through endpoints not listed above (e.g., `/v1/audio/transcriptions`),
the proxy routes based on the `model` field in the request body.  For
GET/DELETE requests without a model field, append `?provider=<name>` to the URL.

---

## Client configuration examples

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-used",           # llmproxy uses the upstream key from config
)

response = client.chat.completions.create(
    model="openrouter/anthropic/claude-3.5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### opencode

Add the following to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",

  "plugin": [
    "opencode-lmstudio"
  ],

  "provider": {
    "lmstudio": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llmproxy",
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "sk-local"
      }
    }
  }
}
```

The `opencode-lmstudio` plugin provides the `@ai-sdk/openai-compatible` adapter.
The `apiKey` value is not used by llmproxy but is required by the adapter; any
non-empty string works.

### pi.dev

Install the [pi-openai-compat](https://github.com/BillJr99/pi-openai-compat)
plugin and point it at `http://localhost:8080`.  No API key is required.

### curl

```bash
# List all available models
curl http://localhost:8080/v1/models | jq '.data[].id'

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openrouter/openrouter/free",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## CLI reference

All flags apply equally to `python run.py` and the installed `llmproxy` command.

```
usage: run.py [--setup] [--config PATH] [--host HOST] [--port PORT]
              [--log-level LEVEL] [--list-providers] [--version]

  (no flags)           Start the proxy server.
  --setup              Interactive configuration wizard.
  --config PATH        Override config file location.
  --host HOST          Bind host (overrides config).
  --port PORT          Bind port (overrides config).
  --log-level LEVEL    DEBUG | INFO | WARNING | ERROR.
  --list-providers     Print configured providers and exit.
  --version            Print version and exit.
```

---

## Environment variables

| Variable          | Purpose                                |
|-------------------|----------------------------------------|
| `LLMPROXY_CONFIG` | Override the default config file path. |

---

## Architecture notes

- The server is a thin Flask application backed by gunicorn (gthread workers)
  when gunicorn is installed, falling back to the Flask development server.
- `/v1/models` queries all providers concurrently via `ThreadPoolExecutor`.  A
  single unreachable provider is logged as a warning and omitted from the
  aggregate response rather than causing an overall failure.
- Config is hot-reloaded on each request via an mtime cache; provider changes
  take effect without a server restart.  Only `host` and `port` changes require
  one.
- Streaming responses are relayed as raw SSE byte streams via
  `stream_with_context`, preserving upstream chunk boundaries.
