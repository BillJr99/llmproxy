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
│   ├── usage.py             ← token/cost accounting + health windows (GET /v1/usage)
│   ├── signals.py           ← tool-result signals that size agentic requests by progress
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
├── LICENSE                  ← Apache License 2.0
├── THIRD_PARTY_NOTICES.md   ← attribution + licences for adapted third-party work
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

`GET /v1/models` advertises ids in a **display** form built from the provider and
the upstream model id:

```
<provider_name>__<upstream_model_id>
```

The `__` (double underscore) is the provider separator. A single `/` may still
appear *inside* the upstream model portion. For example, an Ollama model with
upstream id `qwen2.5vl:3b` is listed as `ollama__qwen2.5vl:3b`, and OpenRouter's
`deepseek/deepseek-chat-v3` is listed as `openrouter__deepseek/deepseek-chat-v3`.
This shape avoids two real-world client bugs:

- Spaces and parentheses break strict client validators (e.g. Hermes rejects
  any model name containing whitespace) — the display form has neither.
- Clients that group their model picker by the segment **before the first `/`**
  (e.g. opencode) would collapse every model under one provider if the id began
  `provider/…`. Keeping `__` as the provider separator means there is no leading
  `provider/` segment, so the full list is shown. (Such clients derive the
  *display label* from the `name` field, which llmproxy populates — including a
  human-readable name for each virtual model, see below.)

| Upstream model (under `openrouter`)   | Display id                                   |
|---------------------------------------|----------------------------------------------|
| `gpt-4o`                              | `openrouter__gpt-4o`                         |
| `anthropic/claude-3.5-sonnet`         | `openrouter__anthropic/claude-3.5-sonnet`   |
| `meta-llama/llama-3/instruct`         | `openrouter__meta-llama_llama-3/instruct`   |

(Upstream ids with *multiple* slashes — like the third row — keep only the last
slash; any earlier slashes collapse to a single `_`, so the display id carries at
most one `/` and the `__` provider separator stays unambiguous.)

Routing always forwards to the upstream under the **original** id; the display
form is purely cosmetic. Internally the proxy uses this same canonical
`provider__model` form (the route cache is also keyed on the `provider/model`
slash form, so an inbound slash id resolves losslessly even when an upstream id
itself contains `__`).

Clients may submit any of these forms in `"model"` on chat/completions requests —
they all resolve identically:

- `provider__model` — current display / canonical form
- `provider/model` — slash form (interior `/` written as `__`); also accepted
- `model__provider` — legacy display form from PR #27
- `model (provider)` — pre-PR #27 legacy display form

So nothing pinned in an existing client config breaks: a request for
`openrouter/gpt-4o` resolves exactly like the advertised `openrouter__gpt-4o`.

#### Classification fields in the model object

Beyond the OpenAI-standard `id` / `object` / `owned_by` / `created`, each entry in
`GET /v1/models` (and `GET /v1/models/<id>`) carries OpenRouter-style classification
fields so clients can infer a model's type without a separate probe:

- `architecture` — `{ "input_modalities": [...], "output_modalities": [...], "modality": "text+image->text" }`, derived from the upstream's modalities (text-only fallback when the upstream doesn't report them).
- `supported_parameters` — surfaces what llmproxy already tracks: `["tools", "tool_choice"]` for tool-capable models and `["reasoning"]` for models tagged in `model_reasoning`.
- `context_length` — normalized from the upstream when available.

These are additive — strict OpenAI clients ignore the extra keys, while clients that
read the OpenRouter schema (e.g. Hermes) can classify models from the listing alone.
The synthetic virtual models (`llmproxy/free`, `llmproxy/tools`, `llmproxy/vision`,
…) carry the same fields.

## Virtual models

Alongside the real `provider__model` ids, llmproxy advertises **synthetic** model
names under the reserved `llmproxy` namespace. A virtual model doesn't map to one
upstream — it stands for a *pool* of candidate models that share a property (free,
local, a reasoning level, a capability, a single provider, …). When you send a
request to a virtual model, llmproxy picks an ordered list of candidates from that
pool and **cycles** through them until one returns a usable answer. This gives you
automatic load-spreading and failover without pinning a specific upstream in your
client config.

Every virtual model is advertised in the `llmproxy/<name>` slash form — so
`llmproxy/free`, `llmproxy/tools`, and the sliced `llmproxy/deep__free`,
`llmproxy/<provider>__free`, etc. (any `/` inside the name is encoded as `__`, so
each advertised id carries exactly one `/`, right after `llmproxy`). This makes
client pickers that group the listing by the segment before the first `/` (e.g.
opencode) put every virtual under one `llmproxy` group with a *distinct* label per
entry, instead of collapsing them. Each virtual also carries a human-readable,
slash-free `name` (e.g. `[llmproxy] Deep — Free`) for UIs that display the `name`
field. Real model ids keep the canonical `provider__model` form so the same pickers
don't collapse every model from one provider.

On **input**, the proxy is liberal: the advertised slash form
(`llmproxy/deep__free`), the canonical internal form (`llmproxy__deep/free`), the
legacy three-part slash form (`llmproxy/deep/free`), and an all-`__` spelling
(`llmproxy__deep__free`) all resolve to the same virtual. A virtual model only
appears in the listing when at least one eligible backend currently exists for it.

The families are:

| Family                    | Examples                                              | Pool |
|---------------------------|------------------------------------------------------|------|
| Cost-tiered (default)     | `llmproxy/loadbalanced`                              | The whole pool, walked free → local → paid |
| General                   | `llmproxy/free`, `llmproxy/local`                    | All free / all localhost-served models |
| Reasoning level           | `llmproxy/exploratory`, `llmproxy/standard`, `llmproxy/deep` (+ `/free`, `/local`) | Models tagged at that reasoning tier |
| Flagship (computed)       | `llmproxy/flagship` (+ `/free`, `/local`)            | The models nearest the state of the art, [selected automatically](#flagship-tier) |
| Capability                | `llmproxy/tools`, `llmproxy/vision` (+ `/free`)     | Models tagged with that capability |
| Per-provider              | `llmproxy/<provider>` (+ `/<dimension>`)            | One provider's models, optionally sliced |
| Fusion (deliberation)     | `llmproxy/fusion`, `llmproxy/fusion__free`           | A panel of models, judged + synthesized |

All of these **except fusion** share the same cycling-and-failover machinery
described next; fusion fans out to a panel instead (see [Fusion](#fusion-virtual-models-multi-model-deliberation)).

The sub-variants are not a separate mechanism. `llmproxy/flagship__free` reaches
the same cycling engine as `llmproxy/flagship`, differing only in which
candidates its pool contains and in the ordering passes that run before the walk;
the failover rules themselves are driven by the upstream's response, never by the
virtual's name. So every family and every `*/free` and `*/local` slice fails over
across the models in its own pool, and the per-provider virtuals do too, within
the one provider they name.

### How cycling & failover works

When a request targets a (non-fusion) virtual model, llmproxy:

1. **Builds the candidate pool** for that virtual name.
2. **Orders** the pool. Free-tier pools are ordered by remaining capacity
   (capacity-aware weighted sampling — see [`free_limits`](#free_limits)); every
   other pool starts from a **random position** to spread load. Two stable
   reorderings may then run on top without ever dropping a candidate: the
   [request-fit triage](#request-fit-triage-every-free-and-local-virtual) for the
   `*/free` and `*/local` virtuals, and [capability ordering](#capability-aware-routing--failover)
   when the request needs a capability. Any models listed in
   [`favorite_free_models`](#favorite_free_models) that are present in the pool
   are then promoted to the front in ranked order, optionally followed by
   [free-tier cache affinity](#free_tier_cache_affinity) so one conversation
   sticks to one model. Last of all,
   [context-window fit](#context_aware_routing) sinks any candidate whose window
   is known to be too small for the request — a no-op, and literally the same
   list object, whenever nothing is known to overflow.

   Capability ordering is **three-valued**: a model tagged for the capability
   ranks first, a model with *no* capability metadata second, and one whose
   metadata says it lacks the capability last. Tag coverage in the shipped
   sidecar is partial and the untagged remainder includes some of the strongest
   tool-callers in the free pool, so treating "unknown" as "incapable" would bury
   exactly the models you want.
3. **Tries each candidate in order**, returning the first **usable** response.

A candidate is considered to have **failed** — so llmproxy moves on to the next
one — in any of these cases:

- **HTTP error** — the upstream returns a status ≥ 400.
- **Timeout / connection error** — the upstream is unreachable or exceeds the
  per-candidate timeout (60s; a slow upstream can't stall the whole failover
  chain).
- **200 with an unusable body** *(non-streaming)* — the body carries a top-level
  `error` object, has no `choices`, or isn't valid JSON. Some providers answer
  `200 OK` while really reporting an error; these now fail over instead of being
  handed to the client.
- **Forced capability not honored** *(non-streaming)* — a `tool_choice` that
  demanded a call came back with no `tool_calls`, or a `response_format` asked for
  JSON and the body wasn't valid JSON. See
  [capability failover](#capability-aware-routing--failover).
- **Stream that errors on arrival** *(streaming)* — llmproxy peeks the first SSE
  chunk before committing; if the stream opens with an `error` event the candidate
  fails over. The peeked chunk is replayed verbatim once a healthy stream is
  committed, so the first token is never dropped.
- **Stream that produces no output** *(streaming, opt-in)* — with
  [`server.stream_commit_on_content`](#stream_commit_on_content) the pre-commit
  check widens from the first non-empty chunk to the first chunk carrying real
  output, so a provider that accepts the request, emits a role preamble and then
  errors or closes fails over instead of corrupting the turn.

**Transient failures fail over immediately; only the last candidate retries.**
While alternatives remain, a *transient* failure (HTTP 429 / 5xx, a timeout, or a
connection error) moves straight to the next candidate — one attempt, no backoff —
so a rate-limited or flaky upstream never stalls the pipeline when another model
could answer now. The **last** candidate has nowhere to fail over to, so it alone
gets a same-candidate retry with a short backoff. Non-transient errors
(400/401/404 and the like) always fail straight over, since a retry wouldn't help.

**The whole walk can be time-boxed.** By default there is no overall deadline: the
per-candidate timeout is 60s, so a long pool of slow upstreams can keep a client
waiting for minutes. Set [`server.cycle_deadline_seconds`](#cycle_deadline_seconds)
to bound the candidate walk as a whole.

**A candidate that rejected a request as too large is remembered**, by the size
it refused rather than by a cooldown, so requests at least that big
[route around it](#oversized-requests) while smaller ones still prefer it.

**Each model appears in a pool exactly once.** The route cache is dual-keyed — every model
is stored under both its canonical `provider__model` id and the advertised
`provider/model` form, so an inbound id in either shape resolves without string surgery —
and until recently the code that *built* candidate pools walked those keys rather than the
routes behind them, so every pool was silently doubled. Failover therefore retried the
same model before moving to the next one, and a provider's shared `free_allowance` was
counted twice over, exhausting at half its real quota. Pools are now built from distinct
routing targets; lookups still use both keys.

**A slow candidate can be given a deadline of its own, and is remembered.** Set
[`server.virtual_timeout_seconds`](#virtual_timeout_seconds) to say how long any
virtual-pool candidate may go without producing bytes — including the gap between
chunks once a stream is running, which is the only bound that catches a reply
that starts normally and then stops. A timeout is then treated as a `429`: the
candidate is cooled and the next request rotates off it, rather than paying the
same timeout again.

**A stream that dies after it starts is not silent.** Once bytes have reached the
client there is nothing to fail over to, but the stream is still terminated
properly — an `error` frame, a final chunk carrying `finish_reason`, and `[DONE]` —
so a client accumulating a partial `tool_calls` argument string learns that no more
fragments are coming instead of hanging or parsing truncated JSON. A stream that
stops because the upstream went quiet also cools the candidate, so the next
request avoids it. The provider is
also **demoted** for it: a candidate is credited with a success when its stream
commits, and a later mid-stream death revokes that credit, so an upstream that
reliably dies four fifths of the way through a generation stops being ranked first.
Client disconnects are excluded, so pressing Ctrl-C never demotes a healthy model.
To fail over from a mid-generation death outright, see
[`server.stream_buffer_full`](#stream_buffer_full).

**Quota errors are remembered.** A 402/429 (or a quota-signalling error body)
doesn't just fail over within the request — it cools that account/model for a
short window so **later** requests skip it too. See
[quota-aware rotation](#quota-rotation); with [multiple accounts](#accounts) the
rotation is accounts-first (same model, fresh credential) before moving on.

When **every** candidate has failed, llmproxy returns the last upstream response
(so you still see the real diagnostic body and status) rather than a synthesized
error; if no candidate was even reachable it returns a `503`.

The order the walk follows is the pool's own: capacity headroom for the `/free`
virtuals, the cost waterfall for `loadbalanced`, a random rotation elsewhere, and
for [flagship](#flagship-ordering) a strict descending benchmark ranking, so a
flagship failover steps down to the next-strongest model rather than to an
arbitrary one.

Two boundaries are worth stating outright, because both are deliberate. The walk
**never widens past its own pool**: an exhausted `llmproxy/flagship` does not
spill into `llmproxy/deep`, and `loadbalanced` is the only virtual that crosses
tiers at all. And a **stream can only fail over before it commits**, because once
bytes have reached the client there is nothing to fail over to, so a
mid-generation death is terminated cleanly and the provider demoted rather than
retried, unless [`server.stream_buffer_full`](#stream_buffer_full) is set.

Which candidate actually answered is reported on every reply, as both a response
header and, for virtual models, a field in the body. See
[route provenance](#route-provenance).

You can inspect the live pool behind any virtual model without sending a chat
request:

```bash
curl http://localhost:8080/v1/models/llmproxy/free | jq '._candidates'
```

### The `free` virtual model

`llmproxy/free` (also accepted: the `llmproxy/free` slash form) pools every model across all providers
whose upstream ID contains the word `free` (case-insensitive) **or** whose upstream
ID (or full `provider/upstream` ID) appears in the resolved `believed_free` list
— see [where routing metadata lives](#routing-metadata). Its pool is
**capacity-aware**: among healthy candidates, models with more remaining
free-tier quota are preferred, while
load is still spread (see [`free_limits`](#free_limits)). Failover then follows the
[shared rules](#how-cycling--failover-works) above, which is exactly what you want
when an individual free endpoint is rate-limited.

```bash
# Use the free virtual model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy/free", "messages": [{"role": "user", "content": "Hello!"}]}'
```

The `llmproxy/free` model appears at the top of `GET /v1/models` whenever at least one
eligible backend is available.

### The `local` virtual model

`llmproxy/local` (also accepted: the `llmproxy/local` slash form) pools every model whose provider
`base_url` hostname is a loopback address (`localhost`, `127.x.x.x`, `::1`,
`0.0.0.0`), an mDNS name (`*.local`), or a Docker host-gateway alias
(`host.docker.internal`, `gateway.docker.internal`). It uses random-start cycling
with the [shared failover rules](#how-cycling--failover-works) — useful for clients
that want whichever local model (Ollama, LM Studio, llama.cpp, etc.) happens to be
running without hard-coding a name.

```bash
# Use the local virtual model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy/local", "messages": [{"role": "user", "content": "Hello!"}]}'
```

The `llmproxy/local` model appears in `GET /v1/models` only when at least one model
from a localhost-backed provider is present in the route cache — meaning the
provider must be reachable and its `/models` listing must have been fetched
successfully.

> **Local models are not added to `believed_free`.** Local-provider models
> (Ollama, LM Studio, OpenWebUI, etc.) live entirely under the `__local` family
> — `llmproxy/local`, `llmproxy/standard__local`, and so on. When the setup
> wizard auto-registers a local provider, it tags each discovered model in
> `model_reasoning` only; `believed_free` is reserved for cloud free-tier
> offerings. The same sync also prunes any `believed_free` and `free_limits`
> entry a local provider had accumulated. It writes to the curated layer of
> `routing_metadata.json`, not to `config.json`. If you want a local model to
> also appear under `llmproxy/free`, add it to `believed_free` by hand, through
> the admin UI.

### Request-fit triage (every `*/free` and `*/local` virtual)

Every free and local virtual — the general `llmproxy/free` / `llmproxy/local`, the
reasoning families (`llmproxy/deep__free`, `llmproxy/exploratory__local`, …), the
capability ones (`llmproxy/tools__free`, `llmproxy/vision__free`), and the
per-provider `<provider>/free` — triages each request to the most appropriately
**sized** model in its pool, before the usual capacity/random cycling. This is the
same "best model for the job" idea as [`loadbalanced`](#the-loadbalanced-virtual-model),
but applied **strictly within a single tier** (see the containment note below).

The proxy estimates the prompt size and detects an explicit "thinking" intent
(`reasoning_effort` of `medium`/`high`, or a truthy `reasoning` field), then orders
candidates by two axes:

- **Reasoning-tier fit** — a short prompt prefers a fast (`exploratory`) model, a
  long prompt or a thinking request prefers a `deep` model, and mid-size prompts
  prefer `standard` (per each model's `model_reasoning` tier).
- **Size fit within a tier** — among models of the same tier, a light request
  prefers the **smaller** model and a deep/thinking request prefers the **larger**
  one (inferred from the model's parameter-count hint, e.g. `70b`). This is what
  lets even a constrained sub-virtual like `llmproxy/deep__free` pick the
  right-sized deep model from whatever is available — a small deep model for a
  quick prompt, the biggest one for heavy reasoning.

This is a *stable reordering* layered below the capability ordering (forced
tools/JSON still win) that never drops a candidate, so failover behavior is
unchanged. It needs no configuration — thresholds live in `server.py`
(`_TIER_SMALL_MAX_TOKENS`, `_TIER_MEDIUM_MAX_TOKENS`).

#### Tool-result signals — sizing agentic requests by how they're *going*

Prompt size is a good proxy for difficulty in one-shot chat and a poor one for
coding agents: a two-line request can carry a task that has been failing for
twelve turns. So when a request's message list contains tool results, llmproxy
reads them and nudges the size-derived tier one step.

It looks at what the agent *did* (assistant `tool_calls`, with shell commands
classified as read/write/edit by their command string) and *how it went*
(`role: "tool"` content, matched against an error table tiered soft/hard/
critical). Those become four axes — error severity, spinning, exploring, and
production intensity — scored into a single number:

- **Escalate** when the agent is erroring or stuck (no writes or edits deep into
  a conversation). Hard overrides escalate immediately on a critical error
  (OOM, connection refused) or on a **context-compaction summary** — compaction
  erases the very evidence this pass reads, so without the override a hard task
  would snap back to the weakest tier at the worst moment.
- **De-escalate** on a clean finish: tests passing, recent writes, no errors.

The scoring is deliberately calibrated so **no single signal can flip the tier
on its own** — one maxed axis scores ~0.46 against a 0.5 threshold, so a switch
needs corroboration. The adjustment is bounded to one step in either direction,
which keeps it a correction to the size heuristic rather than a replacement for
it, and it costs nothing: no extra model call, no network, no configuration.
Explicit `reasoning_effort` still wins outright.

Set `server.tool_signal_routing` to `false` for strictly size-based triage. The
logic lives in `llmproxy/signals.py`, is free of Flask and config, and is tested
against canned agent transcripts in `tests/test_signals.py`.

> **Tier containment.** A `*/free` virtual *only ever* serves models from the
> free list, and a `*/local` virtual *only ever* serves localhost-backed models.
> The triage just **reorders** the already tier-scoped candidate pool — it never
> adds, substitutes, or fails over to a model in another tier. `loadbalanced` is
> the **only** virtual that crosses tiers (its free → local → paid waterfall);
> the `*/free` and `*/local` families never do.

### The `loadbalanced` virtual model

`llmproxy/loadbalanced` (also accepted: the `llmproxy/loadbalanced` slash form) is the "give me a
strong answer for ~free" default. For each request it walks a **cost waterfall**,
keeping spend at or near zero while preferring the most capable model available
in the cheapest tier:

1. **Free-tier cloud** models first — among free models that still have headroom
   (quota left, see [`free_limits`](#free_limits)), the **most sophisticated is
   tried first** (`best-first`): models tagged `deep` outrank `standard` outrank
   `exploratory` in [`model_reasoning`](#reasoning-level-virtual-models), and
   untagged models are ranked by an inferred size/reasoning signal (e.g. a `70b`
   in the name, or a known reasoning model). Remaining capacity only breaks ties
   between equally-capable models. A saturated free model drops to the back but is
   still reachable as a failover. A provider that grants a provider-wide free
   quota/session is also treated as free *while that allowance has headroom* (see
   [`free_allowance`](#free_allowance)).
2. **Local** models next — also $0, but kept a step below free *cloud* so local
   compute is reserved for when free cloud is exhausted. Local models are likewise
   ordered strongest-first (the bigger/deeper local model is preferred).
3. **Cheapest capable paid** model as a last resort — only when no free or local
   model can serve the request **and** [`server.allow_implicit_paid`](#allow_implicit_paid)
   is enabled. By default paid is dropped from this implicit waterfall (an
   exhausted free + local pool returns a clear 429/503), so paid stays reachable
   only by direct `provider/model` name. Among paid candidates the least expensive
   (per the [`pricing`](#configuration) block) is tried first.

This deliberately favors quality over load-spreading within the free tier: a
short prompt no longer gets routed to a weak model just because it's short, so
thinking-heavy cron jobs and agent turns get a capable model while cost stays at
~$0. Failover (below) handles a rate-limited top pick by moving to the next-best.

Cost is the dominant rule: a paid model is **never** tried before a free or local
one, even when only a paid model is tagged for a needed capability — failover is
silent and robust, so the free/local attempts are made first and the request only
falls through to paid if they can't answer. Transient failures (HTTP 429/5xx,
timeouts) **fail over immediately** to the next candidate down the waterfall while
alternatives remain (see [cycling & failover](#how-cycling--failover-works)), so a
rate-limited free model never stalls the request.

```bash
# Keep costs near zero; let llmproxy choose a reasonable model per request.
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy/loadbalanced", "messages": [{"role": "user", "content": "Hello!"}]}'
```

`llmproxy/loadbalanced` appears in `GET /v1/models` whenever at least one model
is exposed to virtual routing.

### Reasoning-level virtual models

You can optionally tag individual models in the config with a **reasoning
level** — `exploratory`, `standard`, or `deep` — to group them by how much
thinking effort they are expected to apply.  When at least one model is tagged
with a given level, llmproxy exposes corresponding virtual endpoints.

There is a fourth tier, [`flagship`](#flagship-tier), which sits above `deep`.
It works differently in one important way: you do not tag models with it.
Membership is computed from benchmark data and refreshed automatically, and it
is an *overlay* rather than a fourth exclusive level, so a flagship model keeps
whatever `deep` or `standard` tag it already carries and `llmproxy/deep` does
not lose its strongest models.

| Virtual model name           | Selects                                                       |
|------------------------------|---------------------------------------------------------------|
| `llmproxy/exploratory`      | All models tagged `exploratory`                               |
| `llmproxy/standard`         | All models tagged `standard`                                  |
| `llmproxy/deep`             | All models tagged `deep`                                      |
| `llmproxy/exploratory__free` | Models tagged `exploratory` **and** qualifying as free-tier   |
| `llmproxy/exploratory__local`| Models tagged `exploratory` **and** served on localhost       |
| `llmproxy/standard__free`    | Models tagged `standard` **and** qualifying as free-tier      |
| `llmproxy/standard__local`   | Models tagged `standard` **and** served on localhost          |
| `llmproxy/deep__free`        | Models tagged `deep` **and** qualifying as free-tier          |
| `llmproxy/deep__local`       | Models tagged `deep` **and** served on localhost              |
| `llmproxy/flagship`          | [Computed](#flagship-tier) — the models nearest the state of the art |
| `llmproxy/flagship__free`    | Flagship models **and** free on that provider                 |
| `llmproxy/flagship__local`   | Flagship models **and** served on localhost                   |

Each endpoint cycles through its pool using the
[shared failover rules](#how-cycling--failover-works); the `/free` variants are
additionally capacity-aware. The `__free` and `__local` variants are also
[request-fit triaged](#request-fit-triage-every-free-and-local-virtual): within
a single-tier pool (all `deep`, all `exploratory`, …) the proxy still prefers the
right-*sized* model for the request — a smaller one for a light prompt, the
largest for heavy reasoning.

The three `flagship` endpoints are the exception to both. They are the only
pools with a measured per-model ranking, so they are walked
[strictly best-first by benchmark score](#flagship-ordering), with capacity and
health demoting a saturated candidate rather than reordering the rest, and
neither capacity sampling nor request-fit triage applies. The `llmproxy/...` slash form (e.g. `llmproxy/deep`, `llmproxy/deep__free`) and the
three-part slash form (e.g. `llmproxy/deep/free`) are also accepted on input.

```bash
# Use the deep reasoning virtual model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy/deep", "messages": [{"role": "user", "content": "Prove P≠NP"}]}'

# Inspect which backends are eligible for llmproxy/standard__free
curl http://localhost:8080/v1/models/llmproxy/standard__free | jq '._candidates'
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
  candidate, one of the failure cases in the
  [shared failover rules](#how-cycling--failover-works). Today this covers:
  - **tools** — `tool_choice` forced a call (`"required"` or a specific
    function) but the response contained no `tool_calls`.
  - **json** — `response_format` requested JSON but the body wasn't valid JSON.

  (Reactive 200-body detection runs on **non-streaming** requests only; streaming
  responses still benefit from proactive ordering and from the first-chunk error
  peek. Capabilities without a reliable 200 signal — **vision**, **reasoning** —
  rely on the upstream returning an HTTP error, which already triggers failover.)

The `tool_choice: "auto"` case is never treated as a failure — a model may
legitimately answer without calling a tool.

Detected capabilities: `tools`, `vision`, `reasoning`, `json`.

When at least one model is tagged, dedicated capability virtual endpoints appear:

| Virtual model name        | Selects                                                  |
|---------------------------|----------------------------------------------------------|
| `llmproxy/tools`         | All models tagged `tools`                                 |
| `llmproxy/tools__free`    | Models tagged `tools` **and** qualifying as free-tier     |
| `llmproxy/vision`        | All models tagged `vision`                                |
| `llmproxy/vision__free`   | Models tagged `vision` **and** qualifying as free-tier    |

```bash
# Route a tool-calling request only to tool-capable free models, failing
# over automatically if one returns no tool call:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy/tools__free",
       "tool_choice": "required",
       "tools": [{"type": "function", "function": {"name": "get_weather"}}],
       "messages": [{"role": "user", "content": "Weather in Paris?"}]}'

# llmproxy/free also benefits — it now orders/fails over by capability when
# the request carries tools or images.
```

Tags are configured via the `model_capabilities` field, which **auto-populates**
from the scraper (OpenRouter's `supported_parameters` / image modality) and the
setup wizard's *Manage model tags → Tag model capabilities* menu — see
[Configuration → model_capabilities](#model_capabilities) below. The `llmproxy/...` slash form (e.g. `llmproxy/tools`, `llmproxy/tools__free`) and the
three-part slash form (e.g. `llmproxy/tools/free`) are also accepted on input.

### Per-provider virtual models

The reasoning, capability, and `free` families above aggregate across **all**
providers. To scope failover to a **single** provider, llmproxy also advertises
per-provider virtual models of the form:

```
llmproxy/<provider>            # cycles through ALL of that provider's models
llmproxy/<provider>__<dimension>
```

where `<dimension>` is one of `exploratory`, `standard`, `deep` (reasoning
levels), `tools`, `vision` (capabilities), or `free`. For example, with a Google
provider:

| Virtual model name              | Selects                                                |
|---------------------------------|--------------------------------------------------------|
| `llmproxy/google`              | All of Google's models                                 |
| `llmproxy/google__deep`         | Google models tagged `deep`                            |
| `llmproxy/google__standard`     | Google models tagged `standard`                        |
| `llmproxy/google__exploratory`  | Google models tagged `exploratory`                     |
| `llmproxy/google__tools`        | Google models tagged `tools`                           |
| `llmproxy/google__vision`       | Google models tagged `vision`                          |
| `llmproxy/google__free`         | Google's free-tier models (capacity-aware, like `llmproxy/free`) |

```bash
# Deep reasoning, but only ever route to Google:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy/google__deep", "messages": [{"role": "user", "content": "Prove P≠NP"}]}'

# Inspect which of Google's models back a per-provider virtual:
curl http://localhost:8080/v1/models/llmproxy/google__free | jq '._candidates'
```

Eligibility: per-provider virtuals are advertised only for providers that are
**enabled**, **non-local** (not `localhost` / `host.docker.internal` / `*.local`),
and **not** opted out via `expose_to_virtual_models: false`. Each variant appears
in `GET /v1/models` only when the provider actually has a backing model for that
dimension. `llmproxy/<provider>__free` uses the same capacity-aware load
balancing and usage tracking as `llmproxy/free`. The `llmproxy/...` slash form and three-part slash input forms are also accepted.

> **Precedence / naming note:** existing global virtual names always take
> precedence. If you name a provider exactly `free`, `local`, `deep`, `standard`,
> `exploratory`, `tools`, or `vision`, then that one colliding name (e.g.
> `llmproxy/standard` or `llmproxy/standard__free`) resolves to the **global**
> virtual; the provider's other per-provider variants (e.g.
> `llmproxy/standard__deep`) still work.

---

### Fusion virtual models (multi-model deliberation)

The virtual models above each select **one** upstream and return its response,
cycling to the next only on failure. The fusion virtual models work differently:
they fan a prompt out to a **panel** of models in parallel, have a **judge**
compare the answers, and have a **synthesizer** write the final reply grounded in
that comparison. This trades latency and cost for quality, so it suits research,
expert critique, and high-stakes prompts rather than quick interactive chat.

| Virtual model name        | Panel drawn from                                          |
|---------------------------|-----------------------------------------------------------|
| `llmproxy/fusion`        | The full non-local pool (or an explicit `fusion.panel`); paid models allowed by default |
| `llmproxy/fusion__free`   | The capacity-ordered free-tier pool (panel, judge, and synthesizer all free) |

The pipeline has four steps. First, llmproxy selects a panel of `panel_size`
models, preferring distinct providers so the deliberation benefits from genuinely
different training and decoding rather than near-identical siblings. Second, it
sends the prompt to the panel in parallel. Third, a judge model compares the
panel answers and emits structured analysis (consensus, contradictions, coverage
gaps, unique insights, and blind spots). Fourth, a synthesizer model writes the
final answer from that analysis. The pipeline degrades gracefully: it proceeds
when at least one panel member answers, falls back to the first successful panel
answer if the judge or synthesizer fails, and errors only when every panel member
fails.

```bash
# Free-tier fusion: panel, judge, and synthesizer all drawn from the free pool
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy/fusion__free",
       "messages": [{"role": "user", "content": "Compare REST and gRPC for a mobile backend."}]}'
```

The models that participated are reported two ways, both additive: a top-level
`llmproxy_fusion` object on non-streaming responses (panel members, judge,
synthesizer, any `failed_models`, a `fell_back` flag, and the judge `analysis`),
and an `X-LLMProxy-Fusion` response header carrying the same provenance without
the analysis, which also works for streamed responses. Strict OpenAI clients
ignore the extra field.

Behavior is controlled by the `fusion` config object:

| Key                  | Default      | Meaning |
|----------------------|--------------|---------|
| `enabled`            | `true`       | Master switch; when false the fusion models are not advertised or served. |
| `panel`              | `null`       | Explicit list of model ids for bare `fusion`; `null` uses the full non-local pool. |
| `panel_size`         | `4`          | Number of panel members (minimum 2). |
| `diversity`          | `"provider"` | `"provider"` prefers distinct providers when selecting the panel; `"none"` takes the pre-ordered prefix. |
| `judge_model`        | `null`       | Model that compares the panel answers; `null` auto-picks a capable pool model. |
| `synthesizer_model`  | `null`       | Model that writes the final answer; `null` auto-picks, preferring one different from the judge. |
| `allow_paid`         | `true`       | Whether bare `fusion` may recruit paid models. `fusion/free` is always free regardless. |
| `report.metadata`    | `true`       | Emit the `llmproxy_fusion` provenance block. |
| `forced_capability`  | `"restrict"` | When a request forces tools or JSON: `"restrict"` limits the panel and synthesizer to capable models; `"bypass"` orders capable-first without restricting. |

When a request forces a capability (a `tool_choice` that demands a call, or a
`response_format` requesting JSON), the panel and judge deliberate in plain text
while the synthesizer call re-attaches the original tools and `response_format`,
so the final answer honors the forced-output contract. The legacy `llmproxy__...`
input form (`llmproxy/fusion`, `llmproxy/fusion__free`) is accepted as well.

> **Scope notes (v1).** Fusion is available on chat/completions only. The
> `llmproxy_fusion` body field and `X-LLMProxy-Fusion` header are populated on the
> OpenAI surface; Anthropic/Gemini inbound requests receive the synthesized answer
> with the header but without the in-body block. The panel and judge are not
> web-augmented, since llmproxy has no server-side web tools.

---

## API dialects — OpenAI **and** Anthropic, in and out

llmproxy speaks more than one API dialect on both edges. Internally everything is
normalized to the OpenAI chat/completions schema, so all routing, virtual models,
capability ordering, caching, and usage accounting work identically regardless of
which dialect a client or upstream uses.

### Inbound — what clients can speak

| Surface | Endpoints | Notes |
| --- | --- | --- |
| **OpenAI** | `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings` | The original surface. Streaming via SSE. `/v1/completions` forwards to the provider's own legacy endpoint and, when the upstream returns 404 (or the model is virtual/streamed), transparently falls back to `chat/completions` — the `prompt` is wrapped as a user message and the reply is rendered back as `text_completion`. |
| **OpenAI Responses** | `POST /v1/responses`, `GET`/`DELETE /v1/responses/{id}` | The shape newer OpenAI-ecosystem agents speak. `input` items, `instructions`, flat `tools`, `max_output_tokens`, `reasoning.effort` and `text.format` all map onto the canonical chat form; the reply is rendered back as a `response` object with an `output` array, and streaming emits the typed event vocabulary (`response.created`, `response.output_text.delta`, `response.function_call_arguments.delta`, `response.completed`, …). See [Responses API](#responses-api). |
| **Anthropic** | `POST /v1/messages`, `POST /v1/messages/count_tokens` | Point an Anthropic SDK at llmproxy. Streaming emits the Anthropic event format (`message_start`, `content_block_delta`, …). |
| **Gemini** | `POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent`, `:countTokens` | Point the Google GenAI SDK at llmproxy. The model id rides in the URL path; streaming emits Gemini `GenerateContentResponse` SSE chunks. |

All four surfaces accept any model id llmproxy knows — direct (`provider__model`) **and**
the virtual models (`llmproxy/free`, `llmproxy/deep`, …). So an Anthropic SDK call
with `model="llmproxy/free"` is routed and load-balanced exactly like the OpenAI path.
(xAI/Grok, Mistral, Groq, DeepSeek, etc. are OpenAI- and/or Anthropic-compatible, so they
need no separate inbound surface — use the OpenAI or Anthropic endpoints for them.)

> **`/api` prefix.** Every endpoint above is also served under an `/api` prefix
> (`/api/v1/models`, `/api/v1/chat/completions`, `/api/v1beta/...`), so clients that
> assume an OpenRouter-/Open WebUI-/Ollama-style base URL (`http://host/api` or
> `http://host/api/v1`) work without hitting a 404 fallback. The bare `/v1` surface is
> unchanged. The admin UI/API is **not** aliased — it stays at `/admin` only.

> **Bare `/models`.** The model listing is additionally served at `/models` and
> `/models/<id>`, alongside `/v1/models`. Some clients treat their configured base URL as
> already being the API root and probe `/models` directly; without the alias that is a
> 404, for the same reason `/version` has its own bare route. Because the `/api` rewrite
> runs before routing, the one alias also makes `/api/models` resolve. Only the listing is
> aliased this way — `/chat/completions` and the other endpoints stay under `/v1`, since
> nothing probes those without a prefix.

```python
# Anthropic SDK pointed at llmproxy — works with streaming and tools
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8080", api_key="unused")
client.messages.create(model="llmproxy/free", max_tokens=256,
                       messages=[{"role": "user", "content": "hi"}])
```

<a name="responses-api"></a>
### The Responses API (`POST /v1/responses`)

Before this surface existed, a Responses-shaped request fell through to the generic
`/v1/<anything>` pass-through, which resolves a provider directly and has **no
cycling engine behind it** — so a virtual model like `llmproxy/free` could not be
used from a Responses client at all. It now goes through exactly the same pipeline as
`/v1/chat/completions`: capacity ordering, capability ordering, context fit, quota
cooldowns and failover all apply.

Three shapes are reconciled:

- **Input.** `input` is a string or a list of items, where an item is a message, a
  `function_call`, or a `function_call_output`. Chat's flat `messages` list has a slot
  for each, so the mapping is total. Consecutive `function_call` items collapse onto
  one assistant message, which is how chat represents parallel tool calls.
  `instructions` becomes a leading system message, and an `input_image` part keeps its
  structure so vision routing can still see it.
- **Output.** A chat completion has one message carrying optional `tool_calls`; a
  Response has an `output` **array** of `message` and `function_call` items, so one
  choice fans out to several. `output_text` is provided directly as the SDKs'
  convenience accessor.
- **Streaming.** Chat streams homogeneous deltas; Responses streams a typed, ordered
  lifecycle with a running `sequence_number`. llmproxy emits the full sequence —
  `response.created`, `response.in_progress`, `response.output_item.added`,
  `response.content_part.added`, `response.output_text.delta`/`.done`,
  `response.function_call_arguments.delta`/`.done`, `response.output_item.done`, and
  `response.completed` (or `response.incomplete` on truncation).

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

client.responses.create(
    model="llmproxy/tools__free",          # any virtual model works
    instructions="You are a terse coding assistant.",
    input="Summarize README.md",
    tools=[{"type": "function", "name": "read_file",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}}}],
)
```

**Tool shape.** Both the flat Responses form (`{"type": "function", "name": ...}`) and
the nested chat form are accepted. OpenAI's server-side built-ins (`web_search`,
`file_search`, `code_interpreter`) are **dropped rather than forwarded**: they are run
by OpenAI's own infrastructure, and nothing behind llmproxy could execute them, so
passing them through would advertise a capability no upstream in the pool has.

**Statefulness.** `store` and `previous_response_id` work against a bounded
**in-process** conversation store, and the limits are worth knowing before you rely on
them: nothing is written to disk, so a restart drops every stored conversation, and a
second gunicorn worker has its own store (another reason the
[worker default](#workers) is 1). At most 256 conversations are kept, evicted
oldest-first. `store: false` opts out.

A `previous_response_id` this process does not hold returns a **400** rather than
answering without the referenced history — silently dropping most of a conversation
would produce a confident, wrong reply, which is worse than an explicit error. Clients
that cannot tolerate that should resend the conversation in `input`, which is fully
supported and entirely stateless.

`GET /v1/responses/{id}` reports whether a conversation is still known; llmproxy
stores the transcript rather than rendered response bodies, and says so in the reply
instead of fabricating one. `DELETE /v1/responses/{id}` forgets it.

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
> OpenAI-shaped `/v1/models` to discover). Best-effort: some provider-specific
> extras (Anthropic thinking, Gemini safety settings) are still not mapped.
> Prompt caching **is** now carried end to end — see below.

<a name="prompt-caching-passthrough"></a>
#### Prompt caching (`cache_control`) is preserved

Anthropic marks a cacheable prefix by hanging `cache_control` off the last block it
should cover. Flattening blocks to a plain string — which every translation path here
used to do — silently deleted those markers, so a client could place its breakpoints
perfectly and still be billed uncached, with nothing in the response to say why.

Markers are now carried across the canonical middle on `system`, on message content,
and on tool definitions, in both directions. The canonical representation puts the
marker on an OpenAI content *part*, which is the same shape OpenRouter uses, so it
also survives to an OpenAI-protocol upstream. Structure is only introduced when a
marker is actually present: an ordinary uncached request still flattens to a plain
string exactly as before.

`anthropic-beta` is also relayed upstream now (it names features, never credentials),
since prompt caching was historically gated behind it. Cache activity comes back in
`usage` as `prompt_tokens_details.cached_tokens` (the read side, OpenAI's standard
field) and `cache_creation_input_tokens` (the write side, which has no OpenAI
equivalent) — without which a caching setup is impossible to verify.

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

      "accounts": [
        {"key": "sk-a", "label": "team-a"},
        {"key": "${KEY_B}", "label": "team-b", "priority": 1}
      ],
      "account_strategy": "round_robin",

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
  "favorite_free_models": [
    "google/gemini-2.5-flash",
    "groq/llama-3.1-8b-instant"
  ],

  "free_tier": {
    "sync_on_startup": true,
    "update_on_startup": false,
    "probe": { "enabled": false, "autoremove": false, "frequency_days": 0 }
  },
  "providers_pr": {
    "enabled": false,
    "repo": "owner/repo",
    "base": "main",
    "branch": "llmproxy-auto/providers",
    "token": "${GITHUB_TOKEN}"
  },
  "fusion": {
    "enabled": true,
    "panel": null,
    "panel_size": 4,
    "diversity": "provider",
    "judge_model": null,
    "synthesizer_model": null,
    "allow_paid": true,
    "report": { "metadata": true },
    "forced_capability": "restrict"
  },

  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "log_level": "INFO",
    "third_party_log_level": "WARNING",
    "forward_user_agent": "auto",
    "user_agent": null,
    "request_timeout": 120,
    "stream_timeout": 300,
    "response_cache_ttl": 120,
    "models_cache_ttl": 60,
    "stream_include_usage": true,
    "report_route": true,
    "allow_implicit_paid": false,
    "saturation_cooldown_seconds": 60,
    "virtual_timeout_seconds": 0,
    "request_log": "off",
    "request_log_max_body_bytes": 0,
    "tool_signal_routing": true,
    "workers": 1,
    "cycle_deadline_seconds": 0,
    "context_aware_routing": false,
    "context_safety_factor": 1.3,
    "context_output_reserve": 4096,
    "stream_commit_on_content": false,
    "stream_precommit_max_bytes": 8192,
    "stream_precommit_max_seconds": 2.0,
    "stream_buffer_full": false,
    "stream_buffer_max_bytes": 8388608,
    "budget_escalation": true,
    "budget_escalation_factor": 16,
    "budget_escalation_ceiling": 65535,
    "budget_escalation_max_retries": 4,
    "free_tier_cache_affinity": false
  }
}
```

<a name="user-agent"></a>
> **llmproxy identifies itself upstream.** It used not to. Where a client sent
> no `User-Agent`, whatever HTTP library was in use filled one in, so upstreams
> saw `python-requests/2.33.1` — a library default leaking out rather than a
> decision. Where a client *did* send one it was relayed verbatim, so a caller
> using urllib got a CDN block page instead of an answer. Measured through the
> proxy against a real provider: `Python-urllib/3.11` returns `403` and four
> kilobytes of Cloudflare HTML, while `curl/8.19.0` and `OpenAI/Python 2.24.0`
> return `200`. A caller's choice of HTTP library should not decide whether an
> upstream responds.
>
> `server.forward_user_agent` picks the policy:
>
> | Value | Behaviour |
> | --- | --- |
> | `"auto"` (default) | Replace a **missing** or **bare library-default** `User-Agent` with `llmproxy/<version>`; pass anything naming a product through unchanged. |
> | `true` | Relay whatever arrived and nothing else — exactly what shipped before this setting existed. |
> | `false` | Always send our own, never relay. |
>
> Under `auto`, strings like `python-requests/…`, `python-httpx/…`,
> `Go-http-client/…`, `Java/…`, `okhttp/…`, `libwww-perl/…`, `axios/…` and
> `aiohttp/…` are replaced, because they identify an HTTP *stack* rather than a
> client and are exactly what CDN bot filters match on. Anything that names a
> product, such as `OpenAI/Python 2.24.0`, is kept, since upstreams use it for
> attribution. Matching is on a **prefix**, so `JavaScriptRuntime/2.0` survives.
>
> **`curl` and `wget` are deliberately not rewritten.** Both pass Cloudflare's
> Browser Integrity Check, and rewriting them would mislead anyone reproducing a
> problem by hand.
>
> `server.user_agent` overrides the string llmproxy calls itself; leave it
> `null` for `llmproxy/<version>`.
>
> This covers more than proxied requests. Model-listing fetches, the flagship
> catalog fetch, local sync, the setup wizard and every scraper and probe now
> identify themselves too. The listing fetch matters most: it builds the route
> cache, so a CDN refusing it makes a provider disappear from every pool at
> once, which reads as "that provider has no models" rather than as a block.

<a name="third_party_log_level"></a>
> **`log_level` means llmproxy's own level.** `logging.basicConfig` sets the
> *root* logger, so `log_level: DEBUG` — the setting you reach for to watch
> routing decisions — used to switch on urllib3, requests and werkzeug debug
> output for the whole process as well. urllib3 alone emits a line per upstream
> call, and making upstream calls is this proxy's entire job, so on a busy
> deployment those libraries were most of the log by volume and buried the lines
> that were the point. Those loggers are now pinned at `WARNING`.
> `server.third_party_log_level` is the escape hatch when you are genuinely
> debugging a transport problem: set it to `DEBUG` for the old behaviour, or to
> any level name. It cannot make a library more verbose than `log_level` itself,
> since the root handler filters first.

> **Three timeouts, not one.** `request_timeout` and `stream_timeout` bound a
> single socket read; [`virtual_timeout_seconds`](#virtual_timeout_seconds)
> bounds *silence* in a virtual pool and is the one that catches a stream which
> starts normally and then stops. They are not interchangeable and they compose
> (the smaller wins) — see [which one you want](#which-timeout).

> **Routing metadata is not in `config.json` any more.** `believed_free`,
> `cost_observed_free_tier`, `model_reasoning`, `model_capabilities` and
> `free_limits` are resolved across four layers, none of which is your config
> file. It is an inbox instead: the server drains those five keys out of it at
> startup, into the curated layer of `routing_metadata.json`, backing the file
> up first. Anything shown here is therefore honoured, and honoured at the same
> precedence it always had, but after the first boot it lives in the sidecar and
> `config.json` holds none of it. See
> [where routing metadata lives](#routing-metadata).

> **Config layout (free_tier / providers_pr).** The free-tier maintenance
> switches and the auto-PR settings live under two grouped objects, `free_tier`
> and `providers_pr`, rather than as loose top-level keys. Configs written with
> the older flat keys (`probe_cost`, `autoremove_believed_free`,
> `probe_frequency_days`, `sync_believed_free_on_startup`,
> `update_believed_free_on_startup`, `pr_providers_list`, `pr_providers_repo`,
> `pr_providers_base`, `pr_providers_branch`, `pr_providers_token`) are still
> accepted: a migration shim in the config loader lifts them into their nested
> homes at load time, with the nested form taking precedence when both are set.
> The mapping is: `free_tier.sync_on_startup`, `free_tier.update_on_startup`,
> `free_tier.probe.enabled`, `free_tier.probe.autoremove`,
> `free_tier.probe.frequency_days`, and `providers_pr.{enabled,repo,base,branch,token}`.

> **`model_filter` doubles as the fallback for a provider with no catalog.**
> When the `/models` fetch fails outright, llmproxy synthesizes the model list
> from `model_filter` rather than leaving the provider with nothing, so a
> provider that publishes no catalog is still fully usable — you just name its
> models yourself. This applies only to a *failed* fetch: a provider whose
> catalog loads and then filters down to zero models is a filter result, not a
> discovery failure, and is left alone.

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
  the chat endpoint. For example, Cloudflare Workers AI has no
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
them automatically when you add a provider such as Cloudflare Workers AI (with
the `{account_id}` placeholder substituted into `models_url`).

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

> **Free-tier accuracy:** The `believed_free` entries in
> `llmproxy/providers.json` are best-effort estimates based on publicly-stated provider
> free tiers. Provider offerings change without notice — no guarantee is made as to accuracy.
> Verify directly with each provider before relying on free availability in production. The
> [`scripts/update_free_models.py`](#keeping-the-free-models-list-current) scraper exists to
> keep these entries current.

<a name="coding-agent-settings"></a>
### Recommended settings for a coding agent

The defaults are tuned for a general-purpose proxy serving mixed traffic. A
long-running tool-calling agent — Hermes, Claude Code, Aider, OpenHands — wants
a different set of trade-offs, because it is one person's load, it cannot act on
half a tool call, and a turn abandoned mid-task is expensive in a way a dropped
chat reply is not. Paste this into the `server` block:

```json
"server": {
  "free_tier_cache_affinity": true,
  "context_aware_routing": true,
  "cycle_deadline_seconds": 240,
  "virtual_timeout_seconds": 120,
  "stream_commit_on_content": true
}
```

| Setting | Why it helps an agent |
|---|---|
| [`free_tier_cache_affinity`](#free_tier_cache_affinity) | Keeps one conversation on one model, so tool-calling conventions and instruction-following do not change mid-task and the upstream prompt cache keeps paying. The load being spread by the default was yours anyway. |
| [`context_aware_routing`](#context_aware_routing) | A long agentic session outgrows small windows; this sinks candidates known not to fit instead of walking the pool collecting `400 context_length_exceeded`. |
| [`cycle_deadline_seconds`](#cycle_deadline_seconds) | Bounds the whole candidate walk. Without it a run of slow upstreams can keep a client waiting for minutes, which presents to the user as "the proxy is broken". |
| [`virtual_timeout_seconds`](#virtual-timeout-agentic) | Raises the per-candidate idle bound above its hard-coded 60s, which the deadline above otherwise clamps the first-token wait down to. Reasoning models routinely think for longer than a minute before emitting anything. |
| [`stream_commit_on_content`](#stream_commit_on_content) | Waits for a chunk carrying real output before committing, so the common free-tier failure "accept, emit a role preamble, then die" becomes a clean failover instead of a corrupt stream the client has already started reading. |

<a name="streaming-posture"></a>
#### Choosing a streaming posture

Once a stream is committed its bytes belong to the client and no failover is
possible, so the only question is how long llmproxy waits before committing.
Three settings answer it, and they layer:

| Posture | Unprotected window | Added time-to-first-token | What the client sees |
|---|---|---|---|
| **Default** (neither flag) | everything after the first non-empty chunk — including a bare role preamble | none | tokens as they arrive |
| **`stream_commit_on_content`** | everything after the first chunk carrying real output | up to `stream_precommit_max_seconds` (default `2.0`) | tokens as they arrive, after that window |
| **`stream_buffer_full`** | none | the **entire generation** | nothing until the response is complete |

`stream_buffer_full` is deliberately **absent** from the block above. It is the
only setting that gives genuine end-to-end failover, which suits an agent well
— but it also means the client receives nothing for the whole turn, and an agent
SDK's own read timeout is measured against exactly that silence. A model that
takes three minutes over a long tool call is, from the client's side,
indistinguishable from a dead one. llmproxy will not time out, because it is
receiving tokens and its own bounds measure silence; the client may abandon the
request underneath it.

So enable it only once the pool is healthy **and** your client's read timeout
comfortably exceeds your longest generation. Until then
`stream_commit_on_content` is the better trade for an agent: it closes the
failure that actually happens on free tiers, costs a bounded couple of seconds
rather than the whole generation, and never withholds the response. It is
redundant once `stream_buffer_full` is on, since buffering re-validates the
entire response with the same checks — so set one or the other, not both.

Three things worth knowing before you enable these:

- **Stream.** All of llmproxy's streaming bounds measure *silence*, never total
  duration, so a model producing tokens steadily is never cut off however long
  it runs. A non-streamed request gets no such protection: the upstream sends
  nothing until the whole reply is ready, so the read timeout necessarily bounds
  total generation time and a slow-but-healthy model is indistinguishable from a
  dead one. Streaming is what makes that distinction possible at all.
- **If you do enable `stream_buffer_full`, your client's read timeout bounds
  time-to-first-byte, not `cycle_deadline_seconds`.** The deadline bounds
  time-to-*commit*, and buffering moves the commit to the end of generation.
- **Reasoning effort and any fallback provider belong in your agent's config,
  not llmproxy's.** llmproxy routes; it does not set reasoning effort on your
  behalf. `flagship` is also rejected if you try to set it in
  [`model_reasoning`](#model_reasoning), since membership is computed rather
  than tagged.

When a request does fail, [`GET /v1/failures`](#v1-failures) reports which
models failed recently and why.

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
all virtual models and powers the `llmproxy/tools` / `llmproxy/vision`
endpoints (advertised when at least one model carries the tag).  Omit it (or set
it to `{}`) to disable capability-aware behavior — the proxy then behaves exactly
as before.  The field **auto-populates** from the scraper (OpenRouter's
`supported_parameters` and image input modality) and from the setup wizard's
"Manage model tags → Tag model capabilities" menu.

What each of these four keys means is unchanged; where it is stored is not. The
setup wizard still writes them into `config.json`, and they are still honoured
there, but the next restart moves them into the curated layer of
`routing_metadata.json` and leaves `config.json` without them. The admin UI
writes to that layer directly. See
[where routing metadata lives](#routing-metadata).

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

When a `…/free` virtual model (`llmproxy/free`, `llmproxy/<level>__free`,
`llmproxy/<provider>__free`, …) picks which backend to use, it scores each
candidate by how much of its quota is still unused and prefers the one with the
most headroom (weighted random, so load is still spread). **Both** request
limits (`requests_per_minute` / `requests_per_day`) **and** token limits
(`tokens_per_minute` / `tokens_per_day`) are now enforced — a model that has
burned through its per-minute token budget is scored down and skipped just like
one that hit its request cap, which keeps traffic inside the free tier for
providers that meter by tokens. Any field set to `null` is ignored. Counters are
in-memory and **per worker process** (see the note on multi-worker below).

<a name="free_allowance"></a>
### `free_allowance` — provider-wide free quota ("free in the moment")

Some providers grant a **provider-wide** free allowance or session that applies
across their models, on top of any explicitly free models. `free_allowance` is an
**optional** per-provider object (inside a provider's block in `config.json` /
`providers.json`) using the same four keys as `free_limits`:

```json
"providers": {
  "someprovider": {
    "base_url": "https://api.someprovider.example/v1",
    "free_allowance": {
      "requests_per_minute": 20,
      "requests_per_day": 200,
      "tokens_per_minute": null,
      "tokens_per_day": null
    }
  }
}
```

The cost-tiered [`llmproxy/loadbalanced`](#the-loadbalanced-virtual-model) virtual
uses it to decide what counts as free *right now*: while the provider's aggregated
recent usage is within this allowance, its models are treated as **free** (tried
before paid); once the allowance is exhausted in the current window they fall back
to the **paid** tier. This is best-effort — counters are in-memory and per worker
process — so it is "as far as we can tell in the moment". Any field set to `null`
is ignored; a provider with no `free_allowance` simply never gains free-in-the-
moment status.

A bundled template may declare a `free_allowance` of its own, in which case the
setup wizard, the admin "add from template" action and `scripts/add_provider.sh`
all copy it onto the new `config.json` entry, since that block is where the
virtual reads it. **ModelScope** ships this way, carrying its account-wide
allowance of 2,000 requests/day. Edit or delete the copied value freely; nothing
writes it back over yours.

<a name="accounts"></a>
### Multiple accounts per provider — credential rotation

A single provider can carry **several credentials** ("accounts") so the proxy
rotates across them and multiplies that provider's free-tier headroom. Declare
them with `accounts` (or the shorthand `api_keys`) alongside — or instead of — the
single `api_key`:

```json
"providers": {
  "groq": {
    "base_url": "https://api.groq.com/openai/v1",
    "accounts": [
      {"key": "${GROQ_KEY_A}", "label": "team-a"},
      {"key": "${GROQ_KEY_B}", "label": "team-b", "priority": 1}
    ],
    "account_strategy": "round_robin"
  }
}
```

- Each account's `key` resolves `${VAR}` references at request time, exactly like
  the single `api_key`. `api_keys: ["sk-a", "sk-b"]` is a bare-string shorthand.
  The legacy `api_key` remains the fallback when neither is set, so **every
  existing single-key config keeps working unchanged**.
- **`account_strategy`** — `round_robin` (default) spreads load across accounts;
  `priority` always prefers the lowest-`priority` account first, falling through
  to the next only when it is exhausted.
- **Rotation is accounts-first, then models.** On every virtual endpoint
  (`llmproxy/free`, `llmproxy/loadbalanced`, `llmproxy/fusion`, the per-provider
  and reasoning/capability families) a request tries a model's accounts before
  moving to the next model — same model with fresh quota is the cheapest way to
  keep serving. Each account meters its **own** free-tier quota (`free_limits`)
  and is tracked separately in [`GET /v1/usage`](#usage-accounting).

<a name="quota-rotation"></a>
### Quota-aware rotation — 402/429 cools a candidate until it recovers

When an upstream reports quota exhaustion — HTTP **402** or **429**, or an error
body carrying a quota signal (`RESOURCE_EXHAUSTED`, `insufficient_quota`,
`rate_limit_exceeded`, or a "quota"/"rate limit" message) — llmproxy marks that
account/model **saturated** and rotates to the next candidate. The mark is
*sticky*: subsequent requests on any virtual endpoint skip the cooled candidate
until it recovers, so `llmproxy/free` transparently avoids a rate-limited model
instead of re-picking it every time. The cooldown honors an upstream
`Retry-After` header when present, otherwise `server.saturation_cooldown_seconds`
(default 60). When a provider has a shared [`free_allowance`](#free_allowance), a
quota error also opens a short provider-wide circuit so concurrent requests stop
draining an already-depleted allowance. All of this is in-memory and per worker
process, and requires no configuration.

### Health-aware ordering — a broken provider sinks in the rotation

Quota cooling above handles a provider that says *"stop asking"*. It does not
handle one that simply doesn't work: before this, a provider returning 500s or
timing out kept its place at the front of the ordering and was retried first on
every request, forever, because only 402/429 earned a cooldown.

llmproxy now tracks the outcome and latency of the last 20 attempts per
model/account and folds a health multiplier into the same score that quota
headroom feeds. Quota says what a provider will still *accept*; health says
whether it currently *works*.

It is a **multiplier, not a filter** — consistent with every other ordering pass
here, an unhealthy candidate sinks to the back but is never dropped, because a
degraded provider that answers still beats a 503. A candidate with fewer than 5
observed attempts scores neutral, so cold models and newly added providers are
never demoted on noise, and a proven-healthy candidate scores exactly 1.0 rather
than being penalised against an untried one. Slowness is folded in
logarithmically and only above 2s, so "slow" nudges the order while "broken" can
cost an order of magnitude.

**What does not count as a provider failure:** client disconnects, llmproxy's own
stream-lifecycle errors, and plain 4xx (a malformed request fails identically
everywhere). llmproxy streams by default, so counting a cancelled stream against
the upstream would let one user pressing Ctrl-C cascade into provider cooldowns —
and on a pool whose last candidate is the only one left, dead-end the rotation.
A forced-capability miss is likewise not ill health: the model answered, it just
lacks the capability, which is what capability ordering is for.

Per-model health is reported in [`GET /v1/usage`](#usage-endpoint)
as `success_rate`, `avg_latency_ms`, `health_samples` and `health_score`. Read
`health_samples` alongside the rate — a `success_rate` of 1.0 over 0 samples
means untried, not proven good. Like all other counters this is in-memory and
per worker process.

### Prompt-cache affinity — keeping a conversation on one upstream

Providers that cache prompt prefixes (Anthropic, OpenAI, DeepSeek) only pay out
when successive requests in a conversation reach the same upstream **and** the
same credential. Spreading load defeats that.

So affinity is applied narrowly — to credential choice within a provider, and to
the **paid** tier of the `loadbalanced` waterfall. It is deliberately *not*
applied to free-tier capacity ordering, where spreading is the entire point.

The key is a client-supplied `prompt_cache_key` (top-level or under `metadata`)
when present, since only the client knows what it considers one conversation.
Otherwise it is derived from the conversation's **root**: every system turn,
plus the first non-system turn. Those are the only messages an agentic client
does not rewrite, so the key is identical on turn 1 and on turn 40. A bare first
user turn with no substantial system prompt (200+ characters) gets **no** key:
there is nothing durable to identify it by yet, so pinning would cost load
spreading and buy nothing. A later turn of that same conversation does get one,
because by then the first user turn is a settled part of the transcript.

> **This used to key on the whole cacheable prefix** — everything before the
> trailing user turn. That is a correct description of what the upstream has
> cached and completely wrong as an *identity*: an agent loop appends an
> assistant turn and a tool result on every iteration, so the key was different
> on every single request. Anything remembering a choice under it, such as
> [`free_tier_cache_affinity`](#free_tier_cache_affinity), wrote a pin each turn
> and read one back never — so a conversation restarted from the top of the
> ranking every turn, paying a `429` on each model above its own before reaching
> the one that had just worked. If your client rewrites its system prompt every
> turn (injecting a timestamp, say), send `prompt_cache_key` instead.

Selection uses rendezvous (highest-random-weight) hashing rather than a modulo
ring, so adding or losing an account reshuffles only that account's share instead
of remapping every conversation — measured at ~10% churn when growing a pool from
8 to 9, against a theoretical floor of 11%. As with every ordering pass, it only
reorders: a pinned upstream that goes down fails over normally.

<a name="route-provenance"></a>
### Route provenance: which model actually answered

A virtual model stands for a pool, so the reply alone never used to say which
candidate produced it. Every response now reports that, through two channels.

**Response headers, on every reply.** `X-LLMProxy-Selected-Model` names the
provider and model that served the request, and `X-LLMProxy-Route-Reason` names
the ordering passes that put it first:

```
X-LLMProxy-Route-Reason: capacity,request_fit=standard(tool_signals:dimensions),capability=tools
X-LLMProxy-Selected-Model: groq/llama-3.3-70b-versatile
```

A `failover#N` suffix means the ranked pick failed and this is the Nth fallback,
so the header alone distinguishes *chosen* from *settled for*. When a tier was
adjusted by tool signals, the log line also records the evidence behind it —
severity, turn depth, recent read/write/edit counts and the resulting score —
rather than just the verdict.

A [flagship](#flagship-tier) pool reports `flagship_rank=<ranked>/<total>`
instead of `capacity` or `cycling`, naming how much of the pool the benchmark
ranking actually covered:

```
X-LLMProxy-Route-Reason: flagship_rank=4/5,capability=tools
```

`4/5` says four of the five candidates carried a score and one did not — an
unscorable pin, sorted last. A flagship pool whose membership cache holds no
scores at all reports the ordering it genuinely used (`cycling` or `capacity`)
rather than claiming a ranking, so the header never overstates what is known.

The headers are not confined to virtual models or to the OpenAI surface. A pinned
`provider__model` request reports itself, so a client never has to branch on what
kind of id it asked for; `/v1/messages`, `/v1/responses` and the Gemini route
carry them through dialect translation; a fusion reply names its synthesizer, the
model whose words actually reach you; a cached reply names the model that
originally produced it; and when *every* candidate fails, the error reply still
names the last one tried, which is precisely when you most want to know.

For a durable record rather than a per-reply signal, see
[`server.request_log`](#request_log), which emits one structured JSON object per
request to stdout — including the route reason above, the real duration of a
streamed reply, and a request id echoed back as `X-LLMProxy-Request-Id`.

**A body field, on virtual models.** Most SDK clients surface a parsed body and
never expose response headers to their callers, so a virtual-model reply also
carries an additive top-level `llmproxy_route` object:

```json
{
  "id": "chatcmpl-...",
  "model": "llama-3.3-70b-versatile",
  "choices": [ ... ],
  "llmproxy_route": {
    "object": "route.report",
    "virtual": "llmproxy__deep/free",
    "provider": "groq",
    "model": "llama-3.3-70b-versatile",
    "selected_model": "groq/llama-3.3-70b-versatile",
    "route_reason": "capacity,failover#1",
    "attempt": 1,
    "failed_over": true
  }
}
```

Strict OpenAI clients ignore unknown top-level keys, which is the same bet the
[`llmproxy_fusion`](#fusion-virtual-models-multi-model-deliberation) block makes.
A pinned request does not get the field, since it already names its own model.

Streamed responses carry the same object in a single synthetic
`chat.completion.chunk` sent ahead of the upstream's own frames. Its `choices`
array is empty, which is the shape clients already receive from the final usage
chunk, so it needs no special handling:

```
data: {"object":"chat.completion.chunk","choices":[],"llmproxy_route":{ ... }}
data: {"choices":[{"delta":{"content":"Hel"}}]}
...
```

Three scope notes. The upstream's own chunks are relayed byte for byte, so the
block is added rather than merged into an existing frame. Non-OpenAI inbound
dialects (`/v1/messages`, `/v1/responses`, Gemini, legacy completions) receive
the headers but not the in-body block, since their renderers build strict
per-dialect shapes. And provenance never costs you an answer: if a body cannot be
parsed it is served exactly as the upstream sent it, field omitted.

| Key                   | Default | Meaning |
|-----------------------|---------|---------|
| `server.report_route` | `true`  | Emit the `llmproxy_route` body block. Set it to `false` for a client that validates its response schema strictly enough to reject an unknown key; the response headers are unaffected either way. |

<a name="error-reporting"></a>
### Error reporting — what a failed request tells you

#### What an exhausted pool returns

Failover is unconditional: every upstream status at or above `400` fails over to
the next candidate. A non-transient status such as `404` simply skips the
same-candidate retries (which only apply to `429` and `5xx`, since a `404` will
not improve against the same endpoint) and moves on immediately.

The question is what to return once **every** candidate has failed. llmproxy is
a gateway, so the status it returns describes *its* boundary. Replaying the last
candidate's status attributes the upstream's problem to the caller, and one case
is actively damaging: a relayed `404` tells an OpenAI-compatible client that the
model does not exist. Clients classify that as terminal, so a transient pool
outage silently disabled the client's own retry logic — the caller gave up with
its retry budget untouched, having asked for a virtual model that exists.

The rule is narrow, so only the misleading case changes:

| Situation | Returned | Why |
|---|---|---|
| No candidates at all | `503` | "There was nothing to try" |
| Candidates tried, none answered at all (every one timed out or refused the connection) | **`502`** | Nothing to relay, but five minutes were spent on five models; the roll-call reports each with `status: null` |
| Every candidate agreed on `400` / `413` / `422` | that status | If all of them reject it identically, the request really is the problem |
| Every candidate returned `429` | `429` + `Retry-After` | Accurate, and it clears by itself |
| Every candidate returned the same `5xx` | that status | Already says "server side"; relaying preserves the diagnostic |
| **Anything else, including any mixture** | **`502`** | No single upstream status can speak for the pool |

So `503` keeps meaning *nothing to try* and `502` means *tried, and all of them
failed* — the status line alone tells them apart. The upstream body is still
included as the diagnostic, alongside a roll-call of every candidate tried:

```json
{
  "error": {
    "message": "All 3 'flagship__free' candidate(s) failed. Last upstream error: No endpoints found that support tool use.",
    "type": "upstream_error",
    "code": "all_candidates_failed",
    "llmproxy_candidates": [
      {"target": "teamorouter/glm-5.3-flash-free", "status": 504},
      {"target": "openrouter/qwen/qwen3.8-27b:free", "status": 504},
      {"target": "openrouter/z-ai/glm-5.2:free", "status": 404}
    ]
  }
}
```

A `200` that failed a content check — a forced tool call that never arrived, an
unusable body — is still returned as-is. There is no upstream error status to
sanitise, and handing the client the real body is deliberate.

<a name="v1-failures"></a>
#### `GET /v1/failures` — which models have been failing, and why

Health scores record *that* a candidate is unwell, not *why*: `record_outcome`
stores a bare boolean. This endpoint answers the question a `502` always raises,
without needing [`server.request_log`](#request_log) turned on.

```bash
curl -s localhost:8080/v1/failures | jq '.by_model'
```

```json
[
  {
    "target": "openrouter/z-ai/glm-5.2:free",
    "provider": "openrouter",
    "model": "z-ai/glm-5.2:free",
    "failures": 12,
    "last_status": 404,
    "last_seen": "2026-09-20T04:50:22.573000+00:00",
    "last_detail": "No endpoints found that support tool use.",
    "kinds": {"capability": 12},
    "statuses": {"404": 12}
  },
  {
    "target": "teamorouter/glm-5.3-flash-free",
    "provider": "teamorouter",
    "model": "glm-5.3-flash-free",
    "failures": 4,
    "last_status": 504,
    "last_seen": "2026-09-20T04:49:41.880000+00:00",
    "last_detail": "upstream timed out",
    "kinds": {"timeout": 4},
    "statuses": {"504": 4}
  }
]
```

Two views of the same data: `by_model` aggregates per routing target, busiest
first, so a repeatedly failing model reads as one row with a count rather than
forty lines; `recent` is the flat newest-first list, each entry carrying the
virtual model that was requested. Each failure is classified as `timeout` (went quiet),
`connection` (was never there), `stream` (opened and then died, produced no
output, or failed mid-generation), `quota`, `capability`, `oversize`, `server`,
`cdn_block` or `upstream`. That is what separates "this model is rate limited" from "this
model cannot do what you asked" from "this model is simply not answering" at a
glance.

`cdn_block` is the newest of those and the least obvious. An upstream behind a
CDN answers a refused request with an HTML interstitial rather than an API
error, so the record used to read `Backend request failed with status 403`
followed by four kilobytes of markup — and the one fact that explains it, that
the **CDN and not the API** said no, was the fact that got lost. The detail now
names it:

```
CDN blocked the request before it reached the API: Cloudflare error 1010
(browser signature). The upstream never saw it.
```

Detection requires two independent markers, so an upstream serving a model
called `cloudflare/llama-3` does not get every ordinary JSON error relabelled.
A `cdn_block` almost always means the outbound
[`User-Agent`](#user-agent) — see below.

Every failover path records, not merely the ones that returned an HTTP status.
A connect timeout is the most common way a free-tier pool fails and was, for one
release, the one thing this report could not see.

| Parameter | Default | Meaning |
|---|---|---|
| `limit` | `50` | How many entries the flat `recent` list carries. `by_model` is never truncated. |
| `since` | — | A unix timestamp or a relative age (`30s`, `15m`, `2h`, `1d`). An unparseable value means "no filter" rather than a `400`: a diagnostic endpoint should not fail on a typo. |

Each record carries `duration_ms`, and each `by_model` row a `slowest_ms`. That
distinction matters more than it sounds: a pool returning fast `404`s and one
burning a full candidate timeout on every attempt produce the same status code
and want opposite responses — the first is a routing or capability problem, the
second a patience one (see
[raising `virtual_timeout_seconds`](#virtual-timeout-agentic)).

`POST /v1/failures/reset` clears the ring and is gated by the same admin auth
guard as [`/v1/usage/reset`](#usage-accounting).

**No secrets are recorded.** Request headers never enter a record at all — they
carry the client's `Authorization` — and because an upstream is free to quote a
credential back inside an error *message*, every detail is additionally scrubbed
of credential-shaped text (`sk-…`, `Bearer …`, `api_key=…`, long opaque tokens)
and truncated to 300 characters. Ordinary text, including hyphenated model ids,
survives intact.

Like [`/v1/usage`](#usage-accounting), this is in-memory and **per process**: the ring
holds the most recent 250 failures from the last six hours, and under a
multi-worker WSGI server each worker reports only the requests it served. That
is one more reason [`server.workers`](#workers) defaults to `1`.

<a name="capability-enforcement"></a>
### Capability enforcement — a model that cannot, is not picked

When a request needs a capability (`tools`, `vision`, `json`, `reasoning`), any
candidate **known** to lack it is removed from the pool rather than merely
sorted to the back.

**This applies to every virtual model, not just the flagship tier, and to paid
candidates exactly as to free ones.** The gate runs where each pool's candidate
list is finalised: once for `llmproxy/free`, `llmproxy/flagship`, the reasoning
tiers, the `llmproxy/<provider>__*` slices and every other virtual, and once per
bucket of the `llmproxy/loadbalanced` cost waterfall, so a paid model that
cannot call tools is no more eligible for a tool-calling request than a free one
is. Cost tier buys no exemption.

The one deliberate exception is a **direct** `provider/model` request. There you
named the model, so llmproxy routes to it and lets the upstream answer for
itself rather than second-guessing you.

**This applies to every virtual model, not just the flagship tier, and to paid
candidates exactly as to free ones.** The gate runs where each pool's candidate
list is finalised: once for `llmproxy/free`, `llmproxy/flagship`, the reasoning
tiers, the `llmproxy/<provider>__*` slices and every other virtual, and once per
bucket of the `llmproxy/loadbalanced` cost waterfall, so a paid model that
cannot call tools is no more eligible for a tool-calling request than a free one
is. Cost tier buys no exemption.

The one deliberate exception is a **direct** `provider/model` request. There you
named the model, so llmproxy routes to it and lets the upstream answer for
itself rather than second-guessing you. Ordering alone cannot express a hard requirement: a later
pass that re-sorts, such as affinity, could put the model back in front, and a
model that cannot call tools is not a usable fallback for a request that needs
them — it is a guaranteed failure wearing the costume of a retry.

What makes removal safe is that "known to lack it" and "not known to have it"
are kept apart. Capability metadata is sparse, and untagged models include some
of the strongest tool-callers available:

| State | Evidence | Treatment |
|---|---|---|
| **Capable** | tagged with the capability | kept, sorted first |
| **Unknown** | no capability metadata at all | **kept**, sorted after capable |
| **Incapable** | carries metadata that omits it, or has refused it | **dropped** |

Two guarantees bound it. **The pool is never emptied**: if every candidate is
known incapable, all of them are kept, because a request that is attempted and
fails over is strictly better than a `503` with no upstream call made, and a
capability map can be wrong. And when candidates are dropped, the route reason
says so (`capability_dropped=N`) and the count is logged, so a pool that
silently shrank is never a mystery.

**Refusals are learned.** A provider that advertises a capability it cannot
actually route to is exactly the case no amount of metadata-reading catches, so
a capability-shaped rejection — OpenRouter's `Filter by Tool Compatibility`
routing step, or a message such as *"No endpoints found that support tool
use"* — records a gap against that **exact** routing target. The same weights
on another provider are unaffected. A learned gap outranks any listing that
claims otherwise, because the request already failed against that endpoint, and
each one is logged at `warning` when first recorded. Gaps live in process
memory and clear on restart.

The matcher is deliberately narrow: a recognised routing-funnel step or one of a
few exact phrases, never the mere presence of the word "tool". A false positive
here sidelines a working model.

**`fusion.forced_capability: "restrict"`** is the stricter opt-in: it
additionally excludes models that are merely *unproven*, not just ones
disproven. It now shares the same non-empty floor, so a forced-tools request
can no longer produce an empty panel.

<a name="budget-escalation"></a>
### `server.budget_escalation` — when a reasoning model thinks itself mute

Send `max_tokens: 5` to a reasoning model and it spends all five tokens
thinking, then returns a `200` with an empty completion and
`finish_reason: "length"`. That is a real reply, technically, and useless.

llmproxy recognises exactly that signature — **every** choice empty **and** at
least one cut off on length — and retries the *same* candidate with a larger
budget rather than failing over, because the model that reasons hardest is
usually the one you most want an answer from. At the defaults:

```
max_tokens: 5   ->   80   ->   1280   ->   20480
```

which is why a five-token ping can take thirty seconds and make three upstream
calls. Nothing is wrong; it is this, working.

**It never fires when the model produced output.** Any content, or any tool
call, and the response is returned untouched. It only ever replaces an answer
you could not have used anyway.

**When it gives up, it fails over.** Once the retries are spent, a still-empty
body is caught by the ordinary unusable-response check and the walk moves to the
next candidate. You are never handed an empty `200`.

| Key | Default | Meaning |
| --- | --- | --- |
| `budget_escalation` | `true` | Master switch. `false` returns the empty body immediately and lets ordinary failover handle it. |
| `budget_escalation_factor` | `16` | Multiplier per retry. A value of `1` or less makes no progress and is treated as "do not bump". |
| `budget_escalation_ceiling` | `65535` | Never ask for more than this, before the per-model clamp below. |
| `budget_escalation_max_retries` | `4` | At most this many retries. `0` is the same as switching it off. |

The factor is deliberately steep rather than a gentle ramp: a reasoning model
starved at 8 tokens is starved at 32 and at 128 too, so small increments just
buy more empty replies at full latency.

> **The ceiling is clamped to the model's context window** when one is known.
> 65535 is higher than most models will accept, and asking for more comes back
> as a `400` — which walks to the next candidate safely enough, but spends a
> round trip and loses the answer the retry existed to recover. A model whose
> window is known to be 8192 is never asked for more, whatever the ceiling says.
> An unknown window is neutral and the configured ceiling applies, matching how
> [context-fit ordering](#context_aware_routing) treats missing metadata. Set
> `model_context` to correct a gateway that misreports its window.

Two other bounds apply and neither is affected by these keys:
[`cycle_deadline_seconds`](#cycle_deadline_seconds) skips a retry once there is
no wall-clock room for it, and the per-candidate timeout still applies to each
attempt.

<a name="context_aware_routing"></a>
### Context-window-aware routing

> **Your `max_tokens` is relayed verbatim.** llmproxy never clamps, lowers or
> rewrites a budget you set — the upstream decides whether it is acceptable, and
> a value above a model's output cap comes back as its `400`. But the budget is
> **added** to the estimated prompt size when judging whether a model's window
> fits, since the reply has to live in that window too. A deliberately large
> `max_tokens` therefore demotes every model that cannot hold prompt plus
> budget, which is correct and worth knowing before you set one by hand.


Per-model context limits used to be parsed only to populate `GET /v1/models` and
were then discarded, so nothing in routing knew how large a candidate's window
was. A conversation that outgrew a candidate got a `400 context_length_exceeded`,
which is non-transient, so it failed straight over to the next candidate — chosen
with no regard for context, and so failing the same way. A long agentic session
walked the whole pool and ended on the last `400` or a `503`, which is exactly the
point at which the work was most valuable.

Set `server.context_aware_routing` to `true` and a further stable reordering runs:
candidates whose window is **known** to be smaller than the request needs sort last.
Two properties are deliberate and worth stating, because they are what keep bad
metadata from turning a working request into a hard failure:

- **It demotes; it never drops.** An oversized candidate is still tried, just last,
  so a wrong `context_length` costs one wasted attempt rather than a `503`.
- **Unknown is neutral, not a demotion.** A model with no reported window keeps its
  incoming position rather than sinking alongside one known to be too small.

Windows are discovered from each provider's `/models` listing. Several
OpenAI-compatible gateways report their *output* cap in `context_length`, so a
top-level `model_context` map overrides discovery for any model, keyed by either a
bare upstream id or a qualified `provider/model` one:

```json
"model_context": {
  "groq/llama-3.3-70b-versatile": 131072,
  "some-mislabelled-model": 8192
}
```

The request size is estimated over the message text **plus** the serialized `tools`
array and any assistant `tool_calls` arguments — the parts an agentic conversation
actually spends its window on, and which the reasoning-tier estimator ignores. That
estimate is inflated by `server.context_safety_factor` (default `1.3`, since code
and JSON tokenize nearer 3 chars/token than 4) and `server.context_output_reserve`
tokens (default `4096`) are held back for the completion, so a model that only just
fits the prompt is not chosen and then made to fail on its own output. The margin is
asymmetric on purpose: overestimating costs one suboptimal but working pick, while
underestimating costs a `400` and a walk down the pool.

When the pass fires it appends `context_fit=~Ntok` to `X-LLMProxy-Route-Reason`.

<a name="request_log"></a>
### `server.request_log` — one structured record per request

llmproxy logs two human lines per request, `→ POST /v1/chat/completions` and
`← POST /v1/chat/completions 200 412ms`. They are fine for watching it work and
useless for reconstructing anything afterwards: there is no id joining them, so
under concurrency you cannot tell which `←` belongs to which `→`; the `←` line
names no model, provider, route reason or token count, though all of those are
known by then; and because Flask runs `after_request` before the WSGI server
iterates a streamed body, every streamed request reports time-to-headers rather
than its real duration.

`server.request_log` adds one JSON object per request, written to **stdout** —
while every other log line goes to stderr, so a collector can tee the record
stream apart from human log chatter without grepping. Three settings:

| Value | What it emits |
|---|---|
| `"off"` (default) | Nothing. Only the `→`/`←` lines, exactly as before. |
| `"metadata"` | One record per request, with everything llmproxy knows *about* it and **no message content**. |
| `"full"` | The same record plus the complete request and response bodies. |

```json
"server": {
  "request_log": "full"
}
```

A record looks like this (`full` mode, a streamed reply):

```json
{
  "object": "request.record",
  "id": "af018083d96e4b83a1812eed19972f67",
  "ts": "2026-09-19T21:38:04.432781+00:00",
  "method": "POST", "path": "/v1/chat/completions",
  "status": 200, "duration_ms": 101.6, "streamed": true,
  "model": "llmproxy/flagship",
  "selected_model": "groq/llama-3.3-70b-versatile",
  "route_reason": "flagship_rank=4/5,capability=tools",
  "failed_over": false,
  "request_body": { "model": "llmproxy/flagship", "messages": [ ... ] },
  "response_body": "data: {\"choices\": ...}\n\ndata: [DONE]\n\n"
}
```

Four details worth knowing:

- **`model` is what the client asked for; `selected_model` is what answered.**
  The proxy rewrites `payload["model"]` in place while routing, so the record
  captures the requested id before that happens.
- **A streamed record is emitted when the stream ends**, not when the headers
  go out, so `duration_ms` is the real duration and `response_body` is the whole
  stream. A client that hangs up mid-stream still produces a record.
- **`X-LLMProxy-Request-Id`** is returned on every response and is the record's
  `id`, so a user reporting a bad answer can name the exact request.
- Bodies are parsed into the record where they are JSON, rather than embedded as
  an escaped string, so `jq` works on them directly.

**What is never recorded, in any mode.** Request headers, which carry the
caller's `Authorization` — a record stream that must itself be handled as
credential material is one nobody will keep. And the bodies of `/admin/*`
requests, because the admin API takes provider API keys in the clear (that is
how you set one); admin requests still produce a record, marked
`"bodies_omitted"`, so the audit trail keeps the fact that a config change
happened without becoming a key dump.

**`full` mode is a data-handling decision, not a log level.** It writes every
prompt and every completion to stdout, so wherever your container logs go, user
content goes too, for as long as they are retained. That is why it is off by
default. `metadata` answers almost every operational question — what was slow,
what timed out, what it cost, which model actually served it, how often the
ranked pick failed over — with no content at all, and is the better default for
a deployment carrying anyone's data but your own.

**`full` also requires `log_level: DEBUG`.** Body-carrying records are emitted at
DEBUG, so setting `request_log: "full"` is not by itself enough to put prompts
and completions in your logs — `server.log_level` has to be `DEBUG` as well.
Content therefore takes two deliberate switches rather than one, and you can
leave `full` configured permanently without it doing anything until you flip the
level. `metadata` records are content-free and still emit at INFO.

`request_log_max_body_bytes` caps each captured body, with the record marking
`"truncated": true` and the original length. `0` (the default) means no cap,
which is what `full` means. Consider setting one: a record holds the whole body
in memory until the response completes, and a streamed answer has no size known
in advance.

**Note:** the first 200 bytes of every streamed reply's first chunk are also
logged at `INFO` on the ordinary (stderr) log, independently of this setting.
That line predates the record stream and is useful for spotting a provider
returning an error frame; be aware it puts a little generated text in the normal
log even with `request_log` off.

<a name="oversized-requests"></a>
### Oversized requests — routing around a model that returned `413`

A `413 Payload Too Large` means this *request* was too big for this endpoint. It
says nothing about whether the model is healthy, and nothing about the next
request, which may be a tenth the size.

Cycling past it already worked: `413` is not a transient status, so the pool
fails straight over to the next candidate with no same-candidate retry. What was
missing was memory. Unlike a `429`, a `413` left no trace, so the candidate kept
its rank and was tried first again on the very next request. Under the old random
rotation that cost one wasted call in N; under the
[benchmark ranking](#flagship-ordering) it costs one on *every* request, because
a deterministic order re-picks the same leader every time.

llmproxy now remembers **the smallest request each routing target has ever
rejected**, and routes requests at least that large around it. Requests below
that size still prefer it, exactly as before.

This is deliberately not a cooldown. Cooling the model for a minute, the way a
`429` does, would also divert the small requests it would happily have accepted —
and the failure being described is a property of the request, not of the model's
availability.

Four things worth knowing:

- **Nothing is persisted.** The watermarks live in memory, per process, and start
  empty on every restart. A body limit is cheap to relearn — the first oversized
  request after a restart rediscovers it, at the cost of one failover — so
  writing it to disk would buy a schema and a staleness problem and nothing else.
  There is no config to set and no file to clean up.
- **The limit only ever tightens.** A later, smaller rejection lowers the
  watermark; a larger one never raises it. The limit can only be bounded from
  above by what was actually refused.
- **A candidate is demoted, never dropped.** It moves to the back of the pool and
  stays reachable, so a pool where everything has been rejected still serves and
  you get the upstream's real `413` rather than a `503` the proxy invented. It is
  also what lets a watermark be disproved: if the candidate is reached as a last
  resort and *accepts* a request that size, the recorded limit was wrong and is
  forgotten, so one spurious rejection cannot sideline a model until restart.
- **The limit belongs to the endpoint, not the credential.** Every account on a
  provider shares one watermark, so one account's `413` informs the rest. This is
  the one place that deliberately differs from the per-account scoping used for
  quota and rate limits.

When the pass moves anything, the route reason says so:

```
X-LLMProxy-Route-Reason: flagship_rank=5/5,oversize=1
```

Until some candidate has actually returned a `413`, the pass is an exact no-op
and does not even measure the request.

<a name="virtual_timeout_seconds"></a>
### `server.virtual_timeout_seconds` — one patience setting for every pool

`request_timeout` and `stream_timeout` are what `requests` calls timeouts, which
means they bound a single socket read rather than the request. A streamed reply
that arrives normally and then stops is the case that exposes the difference: the
socket is not dead, it is merely quiet, so nothing fires until `stream_timeout`
(300s by default) has elapsed — and because the bound is per-read, an upstream
that emits one byte every 299s can hold the connection open indefinitely without
ever tripping it. The symptom is a `200`, a first chunk, and then nothing.

`server.virtual_timeout_seconds` is a single **idle** bound covering every
virtual pool — `llmproxy/free`, the reasoning tiers, flagship, `loadbalanced`,
the per-provider slices — because "how long am I willing to wait" is a property
of the caller, not of the pool. `0` (the default) disables it and leaves the
existing timeouts exactly as they were.

```json
"server": {
  "virtual_timeout_seconds": 90
}
```

It bounds **silence, never total duration**, at every stage of a request:
connect, first byte, and — the case nothing else reaches — the gap between chunks
*after* a stream has committed. A long but steadily-producing generation is never
cut off, however long it runs, which is the thing a total budget gets wrong and
why this is not simply a deadline.

<a name="virtual-timeout-agentic"></a>
#### Raising it for agentic use

There is one case where the default is too tight, and it is easy to hit without
realising why: **a reasoning model that thinks before it speaks.**

Because the bound is idle time, a model emitting tokens steadily is never cut
off. The exposure is entirely *time to the first* token — before a stream
commits, the same bound covers connect and first byte, and a model that reasons
for ninety seconds before emitting anything is, on the wire, indistinguishable
from a dead one.

What makes this surprising is its interaction with
[`cycle_deadline_seconds`](#cycle_deadline_seconds):

| `cycle_deadline_seconds` | Silence tolerated before the first token |
|---|---|
| `0` (default) | `server.stream_timeout` — 300s by default |
| set, e.g. `240` | collapses to the **per-candidate** timeout, 60s |

Enabling the deadline therefore cuts first-token patience from five minutes to
one, because the per-candidate timeout is what the read bound is clamped to. On
a pool of reasoning models that is often the difference between a working
request and a walk of spurious timeouts.

`virtual_timeout_seconds` is the lever, because the per-candidate timeout is
`min(stream_timeout, virtual_timeout_seconds or 60)` — so setting it *raises*
the floor that the clamp lands on:

```json
"server": {
  "cycle_deadline_seconds": 240,
  "virtual_timeout_seconds": 120
}
```

That gives each candidate 120 seconds to produce *something* while still
bounding the whole walk at 240, which is roughly two candidates per request.
That is the trade: fewer candidates reached, each given a fair hearing. Lower it
to 90 to reach more of them, or leave the deadline at `0` for maximum patience
and no wall-clock bound at all.

**A timeout is then treated as a `429`.** Both tell the next request the same
thing: this candidate is not answering. The candidate is cooled for
`server.saturation_cooldown_seconds` (60s by default) and demoted to the
back of its pool, so the following request rotates off it instead of paying the
same timeout again. It stays reachable as a last resort, exactly like a
rate-limited model. This part is not conditional on the knob: a timeout is cooled
whenever one happens.

Only genuine timeouts are cooled. A connection reset or a DNS failure already
fails over on its own, and cooling it too would pull a candidate out of rotation
over a blip that cost nothing. Because urllib3 re-raises a mid-stream read
timeout as a `ConnectionError` rather than a `ReadTimeout`, the message is what
distinguishes them, not the exception type.

Two limits worth knowing:

- **A committed stream cannot fail over.** Once bytes have reached the client
  there is nothing to fail over to, so a stall is terminated cleanly (a real
  `finish_reason` and `[DONE]`, not a truncated stream) and the candidate is
  cooled for next time. To get genuine mid-generation failover you need
  [`server.stream_buffer_full`](#stream_buffer_full), which trades away
  time-to-first-byte for it.
- **Pinned `provider/model` requests are unaffected.** They have no pool to
  rotate within, so they keep using `stream_timeout`; lower that if you want
  them bounded too.

Pick a value above your slowest legitimate inter-chunk gap and below your
client's own read timeout. `90` is a reasonable starting point for agent traffic.

<a name="which-timeout"></a>
#### Which of the three timeouts do I want?

There are three, and they are not interchangeable.

| Key | Default | Applies to | Bounds |
|-----|---------|------------|--------|
| `request_timeout` | `120` | Every upstream call, virtual or pinned, plus provider `/models` discovery and the admin "test provider" button | A single socket read on a non-streaming call |
| `stream_timeout` | `300` | Every streamed call, virtual or pinned | A single socket read on a streamed call, which incidentally bounds the gap between chunks |
| `virtual_timeout_seconds` | `0` (off) | Virtual pools only | Silence, at every stage, streaming or not |

**Lowering `stream_timeout` really would catch a stalled stream.** It becomes the
socket read timeout, and because that is a property of the socket it stays in
force inside the chunk loop, so `stream_timeout: 90` does bound the inter-chunk
gap. At the default `300` the bound exists but is far beyond any agent's
patience, which is why a stall presents as a hang. So if all you want is to stop
streams hanging, that one key will do it.

Three things it cannot do, which is why `virtual_timeout_seconds` exists:

- **`stream_timeout` is wired to gunicorn's worker timeout** (`max(stream_timeout, 120)`),
  so changing it also changes when a worker is killed, and the two can no longer
  be tuned apart.
- **`request_timeout` cannot express this at all for non-streaming requests.** A
  virtual candidate's timeout is `min(request_timeout, 60)`, and that `60` is a
  constant — so `request_timeout: 90` still gives you 60 per candidate, and you
  can only move it down. It is also read in a dozen places including provider
  discovery, so lowering it shortens those too.
- **Neither is scoped to virtual pools.** Both also govern pinned
  `provider/model` requests, which have no pool to rotate within.

Use `virtual_timeout_seconds` when you want one patience setting for the pools
and nothing else disturbed. Use `stream_timeout` when you want every stream
bounded, pinned requests included, and do not mind moving the worker timeout
with it. They compose: the effective bound is whichever is smaller.

**The cooldown is separate from all three.** A timeout is treated as a `429`
whichever key produced it, and that is arguably the larger half of the fix —
without it, a model that has started timing out keeps its rank and costs a full
timeout on *every* request rather than just the first.

<a name="cycle_deadline_seconds"></a>
### `server.cycle_deadline_seconds` — bound the whole candidate walk

`_VIRTUAL_CANDIDATE_TIMEOUT` is 60 seconds **per candidate**, and there is no cap on
how many candidates a virtual model may have. A run of slow or hanging upstreams can
therefore keep a client waiting for minutes, and most agents give up first — which
presents as "the proxy is broken" rather than "the pool is slow".

Set `server.cycle_deadline_seconds` to a wall-clock budget for the candidate walk.
`0` (the default) disables it and preserves the historical behavior. Three details:

- The **first candidate always gets its attempt**, so a tight budget can never
  return a `503` with no upstream call made.
- Each candidate's timeout is shrunk to the remaining budget, and same-candidate
  retries (including the budget escalation for a reasoning model that spent its
  whole token budget thinking) are skipped once there is no room for them.
- The deadline bounds **time-to-commit, not stream duration**. Once a stream is
  committed it runs for as long as the model talks; a twenty-minute legitimate
  generation is never cut off by a routing budget.

Pick a value comfortably below your client's read timeout and above 60s, so at least
one full candidate attempt always fits. `240` is a reasonable starting point.

**The one path that spends several timeouts on a single candidate** is the token-budget
escalation: when a model answers `200` with an empty completion truncated on `max_tokens`
— a reasoning model that spent its whole budget thinking — llmproxy retries it with a
larger budget rather than failing over, multiplying by 16 up to a 4096 ceiling, at most
four times. The ceiling is reached in at most three bumps from any starting budget, so the
worst case is four attempts on one candidate; at the default 60s candidate timeout that is
a few minutes before it even considers the next model. `cycle_deadline_seconds` is what
bounds it. Note this applies to non-streaming requests only — the streaming path fails
over instead.

<a name="stream_commit_on_content"></a>
### `server.stream_commit_on_content` — widen the pre-commit window

By default llmproxy commits to a stream as soon as the first non-empty chunk arrives
and is not an error. Many providers emit an SSE role preamble
(`delta: {"role": "assistant"}`) before anything else, so surviving that check does
not mean the generation works — and once committed, a failure cannot be failed over.

With this enabled, llmproxy buffers until either the first chunk carrying **real
output** (visible text, a tool-call fragment, or a refusal) or a bounded budget
(`server.stream_precommit_max_bytes`, default 8192;
`server.stream_precommit_max_seconds`, default 2.0) is exhausted. Inside that window
nothing has reached the client, so a failure fails over cleanly; on commit the whole
buffer is replayed verbatim, so no token is lost. A stream that *ends* inside the
window having produced no output is also treated as a failure — the streaming
counterpart of the non-streaming "200 with an unusable body" check.

The cost is that time-to-first-token rises by up to `stream_precommit_max_seconds`
for a healthy but slow upstream, which is why it is off by default.

<a name="stream_buffer_full"></a>
### `server.stream_buffer_full` — true end-to-end streaming failover

The pre-commit window narrows the unprotected window; it does not close it. A
provider that dies at token 500 of 2000 has already sent the client 500 tokens, and
no proxy can un-send them.

`server.stream_buffer_full` closes it, by declining to stream incrementally at all:
each candidate's entire response is read and validated before the client sees a
byte, so a mid-generation death is failed over exactly like a `500`. The response is
then replayed from memory as a normal SSE stream, so the client still speaks
streaming and needs no changes.

The trade is real and worth being explicit about: **time-to-first-token becomes
time-to-last-token.** For an interactive chat UI that is the wrong choice. For a
tool-calling coding agent, which cannot act on half a tool call anyway and mostly
renders a turn once it is complete, it is often the right one.

> **It can make an agentic client time out.** Because nothing reaches the client
> until generation completes, the client sees a silent connection for the whole
> turn — and most agent SDKs apply their own read timeout to exactly that. A
> model that takes three minutes to produce a long tool call is, from the
> client's point of view, indistinguishable from a dead one. llmproxy will not
> time out (its own bounds measure *silence*, and it is receiving tokens), but
> the client may abandon the request underneath it. If you enable this, raise
> the client's read timeout above your longest expected generation; if you
> cannot, leave it off and accept the narrower
> [pre-commit window](#stream_commit_on_content) instead.
`server.stream_buffer_max_bytes` (default 8 MiB) caps the buffer; past it the
response commits and the remainder streams incrementally as usual.

<a name="free_tier_cache_affinity"></a>
### `server.free_tier_cache_affinity` — keep one conversation on one free model

[Prompt-cache affinity](#prompt-cache-affinity--keeping-a-conversation-on-one-upstream)
is deliberately *not* applied to free-tier ordering, because spreading load across
quotas is the entire point of that tier.

That default is wrong for a single-user coding agent. There, consecutive turns
landing on different models means different tool-calling conventions and different
instruction-following *inside one task*, and the load being spread is one person's.

Enabling this makes a conversation **stick to the model that last served it
successfully**, for as long as that model keeps answering. The pin is written on
success rather than on selection, so it always names a model that demonstrably
worked; a turn whose pinned model fails and is then served by another candidate
re-pins to that candidate in the same request, so the pin corrects itself in one
turn and no explicit expiry is needed.

Three properties make it safe to combine with every other pass:

- **It only ever moves the pinned target forward.** Nothing is dropped, so
  failover is unaffected, and `favorite_free_models` still outranks it.
- **The first turn is unpinned**, so it takes whatever the ordering chose. On
  [`flagship__free`](#flagship-ordering) that is the top-ranked member, which is
  why this can now apply to a ranked pool at all.
- **The key identifies the conversation, not the turn.** See
  [the note above](#prompt-cache-affinity--keeping-a-conversation-on-one-upstream)
  on why a key derived from the growing transcript made this flag inert.
- **A cooling pin is not promoted.** A candidate demoted for saturation stays
  demoted, rather than being hoisted back and wasting the turn on the one model
  already known to be rate limited.

The cost is the thing it trades away: a pinned conversation stops spreading load
across quotas, which is the free tier's whole default purpose. Pins are held in
process memory, capped, and expire after six hours.

> **Earlier builds used rendezvous hashing here.** That was stateless — it
> remembered no choice, it derived one — and its winner was uncorrelated with
> rank, which is why the pass had to be suppressed on ranked flagship pools.
> Sticky-until-failure replaces it and behaves the same way on an unranked pool.

<a name="workers"></a>
### `server.workers` — why the default is 1

Quota counters, health scores and the saturation registry all live in process
memory and are **not shared between workers**. With more than one, a `429` that
cools a candidate in one worker is invisible to the others, which hit the same
exhausted endpoint on the very next request, and `free_limits` quotas are counted
per worker — so the proxy believes it has roughly *N* times the headroom it really
has and overruns the provider's limits.

The default is therefore `1` worker with 4 gthread threads, which still serves
concurrent requests. Raise `server.workers` only if you want the CPU parallelism and
accept the accounting drift; llmproxy logs a warning at startup when you do.

<a name="allow_implicit_paid"></a>
### `server.allow_implicit_paid` — keep cost-avoiding routes free

`llmproxy/loadbalanced` walks a cost waterfall (free → local → paid). By default
(`server.allow_implicit_paid: false`) the **paid** tier is dropped from that
implicit waterfall: when the free and local tiers are exhausted the request
returns a clear 429/503 rather than silently spending money, and paid models stay
reachable only by their direct `provider/model` name. Set
`server.allow_implicit_paid: true` to restore the historical free → local → paid
fallback. `llmproxy/free` never routes to paid regardless of this flag.

<a name="routing-metadata"></a>
### Where routing metadata lives

Five keys decide how llmproxy routes: which models are free, which of those
turned out to cost money, what tier each sits in, what each can do, and how fast
you may call it.

| Key | What it decides | Which layer normally supplies it |
|---|---|---|
| `believed_free` | which models the free pools may use | `providers.json`, kept current by the free-models sweep |
| `cost_observed_free_tier` | which of those reported a real cost and are treated as paid | learned, the moment a free-marked model bills something; `providers.json` can also ship one, via the [providers PR](#pr-providers-list) |
| `model_reasoning` | the tier a model sits in (`exploratory` / `standard` / `deep`) | learned, inferred from the model's own name; `providers.json` also ships a tier for almost every model it lists as free |
| `model_capabilities` | `tools` / `vision` / `reasoning` / `json` | each provider's own listing and the OpenRouter catalog, unioned; `providers.json` ships a set of its own for the models it lists as free |
| `free_limits` | per-model rate and token quotas | `providers.json`, kept current by the free-models sweep |

They are resolved across four layers, weakest first:

| Layer | Where | Who writes it | Committed |
|---|---|---|---|
| **Defaults** | `llmproxy/providers.json` | the repo, via the [providers PR](#pr-providers-list) | yes |
| **Learned** | `routing_metadata.json`, in `by_model` and `by_provider` | the routing-metadata refresh, and runtime cost observations | **no** |
| **Provider listings** | in memory | each provider's own `/models`, as the route cache is built | n/a |
| **Curated** | `routing_metadata.json`, in `curated` | you, through the admin UI, plus whatever the startup migration drained out of `config.json` | **no** |

**Curated wins.** A correction you make by hand, or through the admin UI,
outranks everything discovered and survives every refresh. That is what makes
the admin editors worth using rather than something the next cadence quietly
undoes.

**A provider outranks the catalog about its own models.** A gateway knows what
it actually deployed, so its listing sits above the refresh; `providers.json` is
the broad base beneath both.

**`config.json` is not one of the layers.** It is an inbox that is emptied at
startup, described under
[config.json is drained, not read](#routing-metadata-inbox) below.

> **The defaults layer only started arriving recently.** Every one of
> `providers.json`'s 603 entries was double-prefixed on the way in: the file
> stores ids already qualified (`google/gemini-2.0-flash`) and the layer
> prepended the provider name again, producing `google/google/gemini-2.0-flash`,
> which matches no lookup. If you read an older README saying the shipped
> defaults apply, they did not: none of that data reached a routing decision
> until this was fixed. Each layer now states its own convention rather than
> guessing, because the same string can be qualified in one file and bare in
> another (Groq's own ids carry a `groq/` vendor namespace, so `providers.json`
> holds `groq/groq/compound` while the sidecar holds `groq/compound`).

#### What is keyed by what

A capability belongs to the *weights*; a rate limit belongs to the *deployment*.
So the learned layer splits them:

- `model_capabilities` and `model_reasoning` are keyed by **normalized model**,
  so one entry covers every provider serving those weights. `glm-5.3-flash`
  appears as `z-ai/glm-5.3-flash`, `zai/glm-5.3-flash`, `zai-org/glm-5.3-flash`
  and bare across a couple of dozen providers, and one learned fact answers for
  all of them.
- `believed_free`, `cost_observed_free_tier` and `free_limits` are keyed **per
  provider**, because the same weights can be free on one provider and metered
  on another.

Lookups try the qualified id, then the bare id, then the normalized model. The
normalized form is tried last, so an entry for this exact model on this exact
provider always beats a fact inherited from the same weights elsewhere.

<a name="free-tier-provenance"></a>
#### A free-tier claim is scoped to whoever made it

The bare-id arm above is the useful one and the dangerous one. Writing
`glm-5.3-flash-free` in your own `believed_free` is documented to mean "this
model, on every provider I have it", and that still works. But a **provider
template** in `providers.json` is not speaking about every provider. It is
vouching for its own, and its entries are qualified ids in that provider's own
key space.

Those two key spaces collide. The `google` template declares
`google/gemini-3.8-flash`, meaning the `gemini-3.8-flash` that Google's own API
serves free. A gateway that namespaces its catalog by vendor serves the very
same weights under the **upstream id** `google/gemini-3.8-flash` — an identical
sequence of characters in a different key space — and bills for it. Matched
bare, Google's free tier leaked onto a paid gateway and llmproxy routed billable
traffic through `llmproxy/free` and `flagship__free`.

So the distinction is **provenance**, not spelling:

| Where the entry came from | Bare match | Qualified match |
| --- | --- | --- |
| A provider template in `providers.json` | **rejected** — scoped to the declaring provider | honoured |
| The learned layer (`routing_metadata.json`) | n/a, always written qualified | honoured |
| Your `config.json`, or the curated layer | honoured on every provider | honoured |

If a gateway genuinely does serve something free, say so by **qualifying** it:
`"believed_free": ["gmi/google/gemini-3.8-flash"]`. A qualified match is
unambiguous and is always honoured, whatever declared it.

Going the other way, `cost_observed_free_tier` is the override when a provider
turns out to bill for something believed free. It is matched qualified-only by
design, so flagging one provider's copy as paid says nothing about anyone
else's, and it beats every belief including the declaring provider's own. The
runtime cost flagger writes it for you the first time a supposedly-free model
reports a real cost.

**Marking a whole provider paid.** An entry of the form `<provider>/*` covers
every model that provider serves, now and in future:

```jsonc
"cost_observed_free_tier": ["somereseller/*"]
```

This is for aggregators and resellers. They re-serve other vendors' upstream
ids, so a provider with no free tier of its own still collects free-tier beliefs
written about those ids elsewhere, and listing its catalog model by model is
both tedious and permanently out of date. The provider-wide mark outranks
everything, including an id that literally spells `:free` — that suffix
describes what the *original* vendor charges, not what this one does. Only the
explicit `/*` spelling is recognised; a bare provider name is not, because it
cannot be told apart from an unqualified model id.

It keeps the provider reachable by its direct `provider/model` name and in the
paid tiers, which is what distinguishes it from the provider-level
`expose_to_virtual_models: false`, which removes the provider from *every*
virtual model.

#### What combines, and what is simply overridden

The five keys do not merge the same way, because they do not mean the same kind
of thing:

- **The two lists accumulate.** `believed_free` and `cost_observed_free_tier`
  union across layers. That still gives complete control because they are a
  pair: one adds a model to the free pool and the other takes it out again.
  Replacing would mean one hand-added entry silently discarding everything the
  refresh had learned. It also means the curated layer can only *add* to a list,
  which is why taking a model out of the free pool is done by recording it as
  cost-observed rather than by deleting it.
- **Capabilities union too, in both directions.** When written, the refresh
  unions what every provider serving the same weights publishes; when read, the
  qualified, bare and normalized entries are unioned rather than stopping at the
  first hit. A provider that omits a tag is silent, not authoritative, so a
  gateway publishing a thin `supported_parameters` cannot retract what the
  catalog or another provider asserted about the same model.
- **A tier and a quota are single-valued.** `model_reasoning` and `free_limits`
  merge per model, with the higher layer winning that entry outright.

#### Provenance: a reading is not a guess

The learned layer holds facts of very different quality side by side, so each
one carries a `<fact>_source` grade beside it. Weakest first:

| Grade | What it means |
|---|---|
| `inferred` | derived from this model's own name |
| `family` | unanimous across the models sharing its family |
| `observed` | a provider listing or the OpenRouter catalog said so |
| `curated` | set by hand, in the admin UI or migrated out of a `config.json` |

A refresh may replace a fact of equal or weaker grade, never a stronger one, so
an inference can never overwrite a reading and nothing can overwrite a hand
correction. A fact with no recorded source reads as `observed`, so a sidecar
written before the field existed is not quietly overwritten by a guess.

Two further rules protect the same data. A model this pass could not see keeps
what it had, so one provider being unreachable at refresh time cannot thin the
routing data; and a pass that learns nothing at all writes nothing, leaving the
previous state in place.

#### Inferring a tier, and lending a family its capabilities

A model that no source describes is exactly the one that most needs a tier, so
the refresh fills two gaps on its own:

- **A reasoning tier from the model's name**, by the same inference the scraper
  and the setup wizard already use: a deep keyword (`qwq`, `deepseek-r1`,
  `magistral`, an `-r1` or `o3-` marker, the word `reasoning`) wins outright,
  then a parameter count (100B and up is `deep`, 15B and up is `standard`,
  anything smaller is `exploratory`), then a few size hints (`large`, `medium`,
  `mixtral`, `70`, `72`, `32`) for `standard`. Everything else is
  `exploratory`. Recorded as `inferred`, the weakest grade, so anything better
  replaces it later.
- **A family's unanimous capabilities**, lent only to a family member that
  publishes none of its own. Unanimity rather than a majority: a family spanning
  coder, omni and vision variants agrees on what the weights share and disagrees
  on the rest, and only the agreement is safe to lend. The generation-scoped
  family is tried first, so `llama4` never lends to `llama2`, falling back to the
  bare family when a generation is too sparse to speak, and falling back once
  more to the longest known family key contained in the model's own key when
  neither exact lookup matches. Families are computed over every observation,
  the catalog included, so a deployment serving three members still benefits
  from what is known about the other twenty. Recorded as `family`.

Both derivations read the **raw upstream id**, never the normalized join key
the facts are filed under. `normalize_model_id` strips separators, so
`llama-3.1-8b` becomes `llama318b`, and a size regex run afterwards reads "318b"
as the parameter count: the raw id infers `exploratory` and the normalized one
infers `deep`. The family derivation splits on those same separators, so fed the
normalized form `llama-2-7b` arrives as `llama27b` and groups with nothing at
all.

The two exact lookups assume the vendor sits behind a path separator, which is
not always where a provider puts it. `zai-glm-5-turbo` folds the vendor into the
name, so it derives `zaiglm5` and `zaiglm`, a family of its own with no observed
members; `claude-haiku-4.5-us-east-1` folds a region onto the end and derives
`claudehaiku45useast1`. Neither model reached a family that had anything to
lend, and on one real 1,751-model deployment 916 models carried no capabilities
at all. So when the generation-scoped and bare lookups both miss, the model's
normalized key is searched for any known family key appearing inside it, and the
longest match wins. `zaiglm5` contains `glm`, and `claudehaiku45useast1`
contains `claudehaiku`. Longest wins is load bearing rather than a tie-break: it
lets a more specific family supersede a more general one, so `llama32` beats
`llama3` where llama-3.2's vision variants do not share llama-3's capability
set. Only a family key of at least three characters is eligible to match this
way, so a very short family name cannot catch models that merely happen to
contain its letters. Three rather than four because the shortest families that
carry real weight on a live deployment are `glm` and `gpt`, and excluding them
would discard most of what the fallback is for. An exact family match is
trusted at any length; the floor applies to the substring pass alone.

Nothing else about the lending changes. The substring match only chooses which
family speaks, and that family still has to be unanimous, still has to clear
`min_family_members`, still lends only to a model publishing no capabilities of
its own, and the result is still recorded as `family`, the weakest grade, so a
later reading from a provider or a hand correction in the admin UI overrides it
and no refresh undoes that.

Measured on real data, 12 of 25 sampled chat models that carried no capabilities
gained them, with zero false positives among the embedding, reranker, TTS and
summarisation models alongside them. Those stay empty for a structural reason
rather than by luck: chat families are named after chat models, so `bge`,
`allminilm` and `aura` share no stem with any of them. A holdout test against
the shipped `providers.json` agreed independently, raising coverage by 50% with
no over-predictions.

What this does not fix is a family with too little data to lend. A model whose
family has fewer than `min_family_members` members carrying observed
capabilities still inherits nothing, because a substring match can only find a
family that already has data, never invent one. That is the `min_family_members`
lever, and it is tunable.

Either inference can be switched off, and the family evidence threshold raised,
in the `routing_metadata` block below.

#### What updates itself, and when

Paste this block at the top level of `config.json`. Every key is optional; a
config without the block behaves exactly as if it contained these values:

```json
"routing_metadata": {
  "enabled": true,
  "refresh_frequency_days": 7,
  "infer_reasoning": true,
  "infer_family_capabilities": true,
  "min_family_members": 3
}
```

| Key | Default | What it does |
|-----|---------|--------------|
| `enabled` | `true` | Master switch. When false the cadence never fires, the learned layer keeps whatever it last held, and an on-demand refresh is refused too: "refresh now" changes the timing, never the decision to maintain this at all. |
| `refresh_frequency_days` | `7` | How often the refresh runs. `0` relearns every time the interval is checked. |
| `infer_reasoning` | `true` | Derive a reasoning tier from the model's name when nothing stronger says otherwise. Recorded as `inferred`. |
| `infer_family_capabilities` | `true` | Lend a family its unanimous capabilities to members that publish none. Recorded as `family`. |
| `min_family_members` | `3` | How many members carrying observed capabilities a family needs before it may lend anything. One or two models agreeing is not evidence about a third. |

The refresh relearns capabilities from each provider's listing and from the
OpenRouter catalog, then fills the gaps by inference. It has its own cadence, so
disabling the flagship tier does not also stop llmproxy learning what its models
can do, and neither source costs anything extra: the provider listings are
already fetched to build the route cache, and the catalog fetch is the one the
flagship refresh already makes. The last-run timestamp lives in
`routing_metadata.json` itself, which sits in the
[state directory](#state-directory) with the other machine-managed files.

Which layers are machine-written, and on what schedule:

- **Defaults** are rewritten by the free-models sweep on
  [`free_tier.update_frequency_days`](#refresh-cadence) (default 7), which is also what a
  [providers PR](#pr-providers-list) proposes back to the repo.
- **The learned `by_model` section** is rewritten by the routing-metadata
  refresh on `routing_metadata.refresh_frequency_days` (default 7). It writes
  capabilities and reasoning tiers only, and only for models this deployment
  actually has a route to: the catalog covers thousands of models a deployment
  never touches, and their capabilities still feed the family evidence without
  being persisted.
- **The learned `by_provider` section** is written on no cadence at all. A
  cost observation lands the moment a `believed_free` model serves a request
  reporting a non-zero cost, which adds it to `cost_observed_free_tier` and
  drops it from the learned `believed_free` so the file does not assert both.
- **The provider-listing layer** is in memory and has no file. It is refilled
  whenever the route cache is rebuilt.
- **The curated section** is written when you write it, when the startup
  migration drains `config.json` into it, and by the local-model sync, which
  tags each model a local provider serves and prunes the entries a local
  provider no longer has.

Every write to `routing_metadata.json` is a read-modify-write under one lock, a
thread lock within the process and an advisory file lock across workers, so a
cost observation and a refresh landing at the same moment cannot lose each
other's write.

#### Reading and editing the merged view

The admin API exposes the resolved result rather than any one layer:

- **`GET /admin/api/routing-metadata`** returns the effective facts per model,
  paged and filtered (`q`, `offset`, `limit`, with `limit` capped at 500). Each
  row names the layers contributing to each fact and the learned grade behind
  it, which answers "why is this model tagged that way" in a way a merged dict
  cannot.
- **`PUT /admin/api/routing-metadata`** sets one model's `capabilities`,
  `reasoning`, `free_limits` or `free` flag by hand, recorded as curated. Per
  model rather than whole-section, so editing one row cannot blank the rest.
- **`POST /admin/api/refresh`** runs a pass now instead of waiting out a
  cadence, with `{"what": "routing_metadata"}` or `{"what": "flagship"}`.

The four older section editors (`believed-free`, `model-reasoning`,
`model-capabilities`, `free-limits`) now read the merged view and write the
curated layer, recording only the entries that **differ** from the layers
below. Saving a form you did not touch is a no-op, and clearing an override
restores whatever the machine had learned instead of freezing today's guesses as
permanent corrections.

#### The same knobs from the admin API

`PUT /admin/api/maintenance` keeps the historical flat field names while storing
into the nested config blocks. The fields covering the two learning cadences,
the inference switches and the PR throttle are:

| Field | Stored at | Default |
|-------|-----------|---------|
| `routing_metadata_enabled` | `routing_metadata.enabled` | `true` |
| `routing_metadata_frequency_days` | `routing_metadata.refresh_frequency_days` | `7` |
| `infer_reasoning` | `routing_metadata.infer_reasoning` | `true` |
| `infer_family_capabilities` | `routing_metadata.infer_family_capabilities` | `true` |
| `min_family_members` | `routing_metadata.min_family_members` | `3` |
| `flagship_enabled` | `flagship_tier.enabled` | `true` |
| `flagship_frequency_days` | `flagship_tier.refresh_frequency_days` | `7` |
| `pr_providers_frequency_days` | `providers_pr.frequency_days` | `0`, matching what the server reads for a missing key: no throttle. Showing anything else would mean saving the form once silently turned "PR on every update" into "every N days" |

#### What gets PR'd, and what never leaves the deployment

The free-models sweep rewrites `llmproxy/providers.json` in place, and with
[`providers_pr`](#pr-providers-list) enabled the running deployment proposes that
file (plus the regenerated `config.example.json`) as a pull request, so every
deployment benefits rather than just yours. Two things go into it:

- **What the sweep scraped**: `believed_free`, `free_limits`, `pricing`, and the
  reasoning tier and capability tags it derives for the models it added.
- **What this deployment learned**, folded in from the sidecar on the way to the
  PR. For every route it serves, the capability set and reasoning tier are taken
  from the curated layer if you set one there and from `by_model` otherwise;
  free status and quotas come from `by_provider`, which is where they already
  belong per provider. Everything is written under the qualified
  `provider/model` id, which is the convention `providers.json` uses.

Promotion is how one deployment's observations become everyone's starting point,
which is the whole reason the PR flow exists. **All five routing keys cross
over**, including hand-set ones: a `model_capabilities` entry you verified
yourself is promoted at the `curated` grade, which is the strongest on the
ladder, and the PR body says so.

`cost_observed_free_tier` is the newest of the five and the one to read most
carefully. It is the negative counterpart to `believed_free` — a provider seen
*billing* for something the catalog calls free — and the pair is why promoting
either is safe: one adds a model to the free pool, the other takes it back out.
But billing is the most deployment-specific fact llmproxy knows. Trial credits,
promotional tiers and per-account pricing mean one deployment's `402` may simply
not be true for the next, and unlike a capability tag, merging one **removes**
the model from every deployment's free pool. The PR body flags such entries as a
single deployment's billing observation rather than a catalog reading.

Promotion is bounded in two ways. Only
providers this repo already ships are touched, because a provider someone added
locally is theirs rather than a default for everyone, and the PR body names the
ones left out rather than dropping them silently. And every promoted fact
carries its grade: the body breaks the count down by provider and by
`curated` / `observed` / `family` / `inferred`, and says plainly that the last
two are llmproxy's guesses rather than anything a provider published. A wrong
capability tag in `providers.json` sends every deployment's request to a model
that cannot serve it, where a wrong one in a local sidecar costs one deployment
a retry, so the grades are what make promoting a guess reasonable rather than
reckless. The whole step is best-effort: a `providers.json` that will not parse,
or a promotion that raises, yields a PR without promotion rather than no PR.

Neither sidecar is itself ever committed. `routing_metadata.json` describes one
deployment's providers and is rewritten on a schedule, and `flagship_models.json`
is computed from whatever that deployment can reach. What crosses into the repo
crosses as a reviewable diff to `providers.json`, never as the sidecar.

`providers.json` is also never copied next to your `config.json`. It is read
from the repo checkout, and a second copy beside your config would drift from it
immediately and invite hand-edits to a machine-written file.

<a name="routing-metadata-inbox"></a>
#### `config.json` is drained, not read

`config.json` stopped being a routing layer because it could not be a stable
record of intent. It is the file a person hand-edits, and machine processes were
writing it too: the local-model sync tagged every model a local provider served
and saved the file back, silently re-seeding the very keys a migration had just
stripped.

It is now an inbox. At startup, before any background work, the server moves the
five keys out of `config.json` and into the sidecar's curated section, at the
same precedence they had, and in their original shape, so migrated data resolves
through exactly the same lookup it always did and moving it cannot change a
single routing decision. Specifically:

- `config.json` is **backed up first**, as `config.json.backup-YYYYmmdd-HHMMSS`
  beside itself. If the backup fails, nothing is migrated.
- An existing curated entry **wins** over the incoming config value, so running
  the migration again after you have corrected a fact in the admin UI cannot
  resurrect the older `config.json` value over your correction.
- The five keys are then removed from `config.json`, which afterwards holds none
  of them. The move is logged with a `[config-migration]` prefix, naming how
  many entries of each key were new and which backup it wrote.

Anything you add to `config.json` by hand afterwards is still honoured, because
the file is read as part of the curated layer, but only until the next restart
absorbs it. The curated section wins where both speak.

`config.example.json` deliberately no longer carries these five keys. It is
generated from `providers.json`, and a `config.json` copied from it would have
them drained into the curated section on first boot, pinning the shipped
defaults in the strongest layer where no later refresh could improve them. The
same data still reaches routing as the defaults layer, read straight from
`providers.json`, which is where the sweep and the providers PR keep it
current.

<a name="favorite_free_models"></a>
### `favorite_free_models` — ranked priority list for free-tier routing

`favorite_free_models` is an **optional** top-level array of model IDs listed in
preference order.  When a `*/free` virtual endpoint (e.g. `llmproxy/free`,
`llmproxy/deep__free`) or the free tier of `llmproxy/loadbalanced` selects a
backend, models in this list are promoted to the front of the candidate pool
**in the order listed**, before the normal capacity/request-fit/capability
algorithm handles the rest.

```json
"favorite_free_models": [
  "google/gemini-2.5-flash",
  "anthropic/claude-3-5-haiku-20251001",
  "gpt-4o-mini"
]
```

Each entry is matched case-insensitively against the upstream model ID (bare,
e.g. `gpt-4o-mini`) or the fully-qualified proxy ID (e.g.
`openai/gpt-4o-mini`).  A favorite is only promoted if it is **currently
believed-free** (present in `believed_free` and not flagged as cost-observed);
if it is absent from the virtual model's candidate pool it is silently skipped
and the remaining favorites and the normal algorithm continue unchanged.

**Cost-observation persistence:** if a favorite is later removed from
`believed_free` because a cost was observed at runtime, it remains in
`favorite_free_models`.  When a future sync restores it to the free pool (e.g.
the provider makes it free again), it is automatically re-promoted without any
manual config change.

`favorite_free_models` has no effect on non-free virtual endpoints
(`llmproxy/deep`, `llmproxy/tools`, etc.) or on fusion virtuals.

The admin UI's **Models & Categorizations** tab includes a **Favorite free
models** panel where you can add models from a grouped-by-provider picker,
reorder them with up/down buttons, and remove entries — changes are saved
immediately.

<a name="usage-accounting"></a>
<a name="usage-endpoint"></a>
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
  a request reporting a **non-zero** cost. Use this to spot a model that has
  quietly left its free tier.
- **Per-account rows.** When a provider has [multiple accounts](#accounts), each
  metered credential is its own row carrying an `account` field (the account's
  label, never the key), so per-account free-tier consumption is visible; the
  `model` id stays the clean `provider/model` form and `totals` aggregate across
  accounts. Single-credential providers omit `account` and read exactly as above.

  On the **first** such observation the proxy also appends the model's qualified
  id to **`cost_observed_free_tier`** in the
  [learned layer](#routing-metadata) of `routing_metadata.json`, under the
  provider that billed it (a best-effort, idempotent, operator-editable
  denylist). It is written there rather than to `config.json`, which the proxy
  no longer writes at all: a machine process editing the file a person
  hand-edits is what the layering exists to stop. The same write drops the model
  from the learned `believed_free`, so the sidecar does not assert both at once,
  and routing avoids it from the next request onward. This stops a paid model
  from being repeatedly re-added (and re-opening a providers PR) every restart,
  without needing the cost probe.

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
  "free_tier": {
    "probe": { "enabled": false, "autoremove": false, "frequency_days": 0 }
  }
}
```

- **`free_tier.probe.enabled`** — when `true`, [`scripts/update_free_models.py`](#keeping-the-free-models-list-current)
  actively probes each `believed_free` model with a tiny real chat request
  (`max_tokens: 1`) using your configured API keys, inspects the returned
  `usage`/cost, and flags any model that reports a cost. This spends a small
  amount of quota, so it is off by default. (You can also trigger it for a single
  run with the `--probe` flag.)
- **`free_tier.probe.autoremove`** — when `true`, the scraper **removes** any
  model that probes (or prices) as non-free from `believed_free` (and therefore
  from the `/free` virtual model). When `false` (default), such models are
  reported in the scraper output and in `flagged_paid_free_models`, but left in
  place for you to review.
- **`free_tier.probe.frequency_days`** — throttles the probe so it runs at most once every
  _N_ days, which matters when `free_tier.probe.enabled` is combined with
  `free_tier.update_on_startup` (otherwise every server boot would spend
  quota). `0` (default) probes on every run; `1` is at most once a day, `7` once a
  week, etc. The last-run timestamp is cached in `probe_state.json` next to your
  `config.json` (not in `config.json` itself). The throttle applies to both
  `probe_cost: true` and the `--probe` flag; pass `--ignore-throttle` to
  `update_free_models.py` to force a probe regardless of how recently one ran.

<a name="sync-on-startup"></a>
### Syncing the live config on startup — `free_tier.sync_on_startup`

**On by default, and now a no-op.** This flag used to copy the bundled
`providers.json` sidecar's `believed_free` / `free_limits` / `model_reasoning` /
`model_capabilities` into your live `config.json` on every boot, which was how a
merged or `pip install -U` update reached a running proxy. The proxy now reads
`providers.json` directly as the [defaults layer](#routing-metadata), so a
shipped update reaches routing with nothing to copy, and copying is what froze
the data in the first place.

The flag is still read and the startup step still runs, logging a
`[startup-sync]` line saying there is nothing to sync. Setting it to `false`
skips that line and changes nothing else. The same is true of running the
reconcile by hand:

```bash
python scripts/update_free_models.py --sync-config-only --config ~/.config/llmproxy/config.json
```

`free_tier.update_on_startup` below is a different thing entirely: it runs the
**full network scrape** to refresh the sidecar itself.

<a name="update-on-startup"></a>
### Running the updater on startup — `free_tier.update_on_startup`

Set the top-level flag to refresh free-tier data automatically when the server
boots:

```json
{ "free_tier": { "update_on_startup": true } }
```

When `true`, the server runs `scripts/update_free_models.py` once per worker in a
background thread at startup (it never blocks request handling). It:

- **rewrites `llmproxy/providers.json`** (the sidecar) with any `believed_free` /
  `free_limits` / `model_reasoning` / `model_capabilities` / `pricing` changes,
  and regenerates `config.example.json`. Nothing is written into your
  `config.json`: the rewritten sidecar is the
  [defaults layer](#routing-metadata), so the change reaches routing directly.

Every line the updater prints — including `Updated …/providers.json` and each
`believed_free` add/remove — is re-emitted through the server log with a
`[startup-update]` prefix (at `INFO` level), so set
`server.log_level` to `INFO` to watch it work. If `free_tier.probe.enabled` is also `true`, the
startup run includes the active cost probe (and, with `free_tier.probe.autoremove`,
removes any model it finds is no longer free). Defaults to `false`.

The startup run is throttled by
[`free_tier.update_frequency_days`](#refresh-cadence) (default 7), so enabling
this flag on a deployment that restarts often does not re-scrape every provider
on every boot. Leaving it `false` does not mean the deployment never refreshes:
the periodic check described under [Refresh cadence](#refresh-cadence) still
runs on the same interval.

> The scraper lives in the repo-root `scripts/` package. The Docker image ships
> it, and the server adds its parent directory to `sys.path` so the import works
> under gunicorn. If a slimmed-down deployment omits `scripts/`, the server logs
> `[startup-update] updater unavailable …` and skips the update. The sidecar
> rewrite is **ephemeral in a container** (it lives in the image layer) — the
> durable effect is the `config.json` sync on your mounted volume. To land sidecar
> changes back in the repo, use the [CI auto-update workflow](#automated-providersjson-updates-ci).

<a name="refresh-cadence"></a>
### Refresh cadence — `free_tier.update_frequency_days`

The refresh is **not** a cron job and needs no scheduler. A running proxy checks
whether a refresh is due, and runs one if so, at startup and then periodically
while it serves traffic. `free_tier.update_frequency_days` sets how often that
may happen, and defaults to weekly:

```json
{
  "free_tier": {
    "sync_on_startup": true,
    "update_on_startup": false,
    "update_frequency_days": 7
  }
}
```

The cadence is what makes the free-model list self-maintaining. Each refresh
re-reads every default source, so within one interval:

- a **newly free model is picked up**, including an unsuffixed cloaked or
  "stealth" model. Detection keys on `$0` pricing in the provider catalog rather
  than on a `:free` suffix, so a model does not have to be named `…:free` to be
  found;
- a model that is **no longer free loses the tag**, because a non-zero price is
  high-confidence evidence against it; and
- a model that has been **withdrawn upstream is dropped**, because a catalog
  that no longer lists it is treated as authoritative about what exists.

Set a smaller number to refresh more eagerly, or `0` to refresh every time the
interval is checked. The last-run timestamp lives in `update_state.json` beside
your `config.json`, not in the config itself, so restarting the server does not
trigger a fresh scrape on every boot; a deployment that restarts constantly still
scrapes about once per interval.

Two related settings sit nearby and are easy to confuse:

- `free_tier.sync_on_startup` (default `true`) is the **no-network** path. It
  reconciles your `config.json` from the bundled `providers.json` sidecar at
  boot. See [`sync_on_startup`](#sync-on-startup).
- `free_tier.update_on_startup` (default `false`) additionally runs the network
  scrape during startup itself rather than leaving it to the periodic check. It
  honours `update_frequency_days` too. See
  [`update_on_startup`](#update-on-startup).
- `free_tier.probe_timeout_sec` is the read timeout for both probes. It is not
  a cadence at all.

<a name="the-two-cadences"></a>
<a name="the-three-cadences"></a>
#### There are three independent cadences

llmproxy runs three scheduled background jobs. They are unrelated, have separate
settings, and keep separate state files, so tuning one does not affect the
others:

| Job | What it refreshes | Setting | Default | State file |
|-----|-------------------|---------|---------|------------|
| **Free-models sweep** | `believed_free`, `free_limits`, `pricing` — which models are free and what their limits are | `free_tier.update_frequency_days` | 7 | `update_state.json` |
| **Routing-metadata refresh** | What each model this deployment serves can do and which tier it sits in, for [the learned layer](#routing-metadata) | `routing_metadata.refresh_frequency_days` | 7 | `routing_metadata.json` |
| **Flagship recompute** | Which models are near the state of the art, for [`llmproxy/flagship`](#flagship-tier) | `flagship_tier.refresh_frequency_days` | 7 | `flagship_models.json` |

All three default to weekly and all three run at startup when due, but they are
otherwise independent: the free-models sweep scrapes provider docs and catalogs,
the routing-metadata refresh reads provider listings and the OpenRouter catalog
and then infers what neither covers, and the flagship recompute reads benchmark
scores and re-ranks what you can reach. Each has its own gate, so turning one
off leaves the others running: the sweep's periodic check needs
`free_tier.sync_on_startup` or `free_tier.update_on_startup`, and the other two
are gated by `routing_metadata.enabled` and `flagship_tier.enabled`.

All three state files live in the [state directory](#state-directory), and each
one carries the last-run timestamp that throttles its own job. A job whose
timestamp cannot be recorded is due on every interval check, so a state
directory that cannot be written turns all three weekly cadences into a
continuous loop. llmproxy holds the timestamps in memory when the write fails,
which keeps the cadences honest for that process, and warns once at startup.

> **The interval keys are not spelled alike, and the nesting differs too.** The
> routing-metadata refresh and the flagship recompute both use
> `refresh_frequency_days`, one level down from the top of their block. The
> free-models sweep uses `update_frequency_days` at the same depth. The cost
> probe is the odd one out on both counts: it sits a level deeper, inside
> `free_tier.cost_probe`, and spells its interval plain `frequency_days`. So
> `free_tier.cost_probe.frequency_days` is the current spelling, and
> `free_tier.probe.frequency_days` is the older one, still accepted and lifted
> into place by the config loader. `free_tier.frequency_days` and
> `free_tier.cost_probe.refresh_frequency_days` match nothing and are silently
> ignored.

**Everything else is a throttle, not a cadence.** Within the free-models sweep,
individual sources have their own frequency settings that limit how often *that
source* may participate in a run. A source throttle can only ever make a source
run **less** often than the sweep — never more, because a source only runs as
part of a sweep:

| Setting | Scope | Meaning |
|---------|-------|---------|
| `free_tier.update_frequency_days` | the whole sweep | **How often the sweep runs at all.** Everything below is subordinate to it. |
| `free_tier.cost_probe.frequency_days` | one source | The billing probe (which spends real quota) runs at most this often. It is the only source with a throttle of its own, because it is the only one that costs money. |

The `:free`-discovery endpoint probe has **no** frequency setting: it only
issues `GET /models` per provider and spends no quota, so it simply runs on
every sweep. Both probes share one read timeout,
`free_tier.probe_timeout_sec`.

A practical consequence: if a source throttle is set to the *same* period as
`update_frequency_days`, the two can beat against each other. A sweep firing a
few minutes before the source's period has fully elapsed will skip that source,
which then waits a whole further period — so a 7-day source throttle under a
7-day sweep can effectively become 14 days. Set source throttles a little
**shorter** than the sweep (say 6 days under a 7-day sweep) if you want them to
run on every sweep.

> **Upgrading from an older version?** The `free_tier.endpoint_probe` block is
> gone. `frequency_minutes` used to be the de-facto master cadence — the job
> that consumed it ran the *entire* updater, so a stock config re-scraped every
> provider every 30 minutes as a side effect of a probe setting. The endpoint
> probe now has no frequency of its own, so the key is obsolete and ignored;
> `update_frequency_days` is the setting you want. Its sibling `timeout_sec`
> moved to `free_tier.probe_timeout_sec` and is now shared with the cost probe.
> Old configs are migrated automatically on load and the sweep prints a notice,
> but you can delete the `endpoint_probe` block once you see it.

<a name="flagship-tier"></a>
### The flagship tier — `flagship_tier`

`llmproxy/flagship` routes only to models near the current state of the art:
the ones actually capable of driving a long agentic loop. It sits above `deep`
in the tier order, so a flagship model outranks everything else when the proxy
picks a candidate.

Three things make it different from the other tiers.

**You never tag models with it.** Membership is computed from benchmark data
and refreshed on a schedule. `flagship` is rejected if you try to set it in
`model_reasoning`, and it is not offered in the setup wizard's level picker.

**It is an overlay, not a fourth exclusive level.** A flagship model keeps its
existing `deep` or `standard` tag and appears in both places. Promoting your
best models therefore does not empty out `llmproxy/deep`, which is what a
fourth exclusive tier would have done.

**Nothing about it is shipped or committed.** Which models qualify depends
entirely on which providers *you* have configured and what each of them
currently serves, so the list is different for every install. It is computed
locally into `flagship_models.json` in the [state directory](#state-directory),
alongside the other machine-managed files. Your config holds only policy — the pin and
exclude lists.

A prompt's size never drifts into flagship, either. The request-fit heuristic
tops out at `deep`; flagship is reachable only by asking for it by name.

#### How membership is decided

Four rules, in this order:

1. **Benchmarks rank.** Sources use incompatible scales, so each is
   rank-normalised to a percentile over the models it covers and the
   percentiles are combined with a median. Raw scores are never averaged. A
   model missing from one source is ranked on the sources that do cover it
   rather than penalised for the gap.
2. **Spec gates veto.** A context-window floor, and optionally tool-calling
   support. These are a veto rather than a selector: on their own they admit
   almost everything. A model whose context window cannot be determined fails
   the gate.

   **Capability is not vetoed here by default.** `require_tools` exists but is
   off, because capability belongs to the *request*, not to membership: a call
   that needs tools already cannot be routed to a model that lacks them. Having
   it on at membership time as well was actively harmful — it made the floating
   bar descend further hunting for free models that happened to carry the tag,
   so the tier filled with weaker models chosen on a capability rather than a
   score. Membership is decided on merit; capability is decided per request.

   Specs are read **per routing target**, from the most specific source that
   has anything to say: the provider's own listing for that exact
   `provider/model` id, then the catalog's entry for that exact id, and only
   then the merged profile joined on the normalised name. The last of those is
   the [carry-across](#coverage-and-when-to-use-a-pin) that lets an unscraped
   provider be gated at all; the first two exist because a *billing variant* is
   a different set of endpoints. `vendor/model:free` normalises onto
   `vendor/model`, so without them it inherits its paid sibling's tool support
   and context window, joins the tier, and then fails every tool call with an
   upstream `404` that no failover can fix.
3. **The bar floats.** Starting from `start_percentile`, the bar is lowered
   until at least `min_flagship_free_models` **distinct** free models qualify.
   There is no lower bound, so a thin free tier produces a smaller tier rather
   than an error.

   The floor is a statement about **what ends up in the tier**, not about how
   far the bar walked. Excluded targets are filtered out before the walk
   begins, so an exclude can never be counted toward the floor and then removed
   afterwards, leaving the pool one short. The same applies to free-ness: a
   model whose only free routing target is excluded does not count as free.

   It is also re-checked against the **live** tier. Membership is cached at
   refresh time but free status is re-evaluated on every read, so a member that
   becomes [cost-observed](#free-tier-provenance) leaves `flagship/free`
   immediately while the cache still lists it. When the live count drops below
   the floor, the next interval check recomputes regardless of
   `refresh_frequency_days` — otherwise the pool would sit short for up to a
   week. If the *previous* run also came up short, nothing is recomputed: the
   deployment simply does not have that many free models, and one warning says
   so rather than re-fetching every catalog on every tick.
4. **Pins and excludes win.** A pin bypasses both the bar and the spec veto.
   An exclude is applied last and beats everything, including a pin.

<a name="flagship-ordering"></a>
#### How the pool is ordered

Flagship is the only tier with a measured per-model ranking, so it is the only
one that is *ordered* rather than rotated or load-spread. The combined
percentile from rule 1 is not discarded once membership is settled: it is
written into `flagship_models.json` beside the member list, and the router
walks the pool **strictly best-first**, so failover descends the ranking rather
than sampling it. Every flagship entry point does this — `llmproxy/flagship`,
`flagship__free`, `flagship__local`, and the per-provider
`llmproxy/<provider>__flagship` slice.

The score is the *primary* key, and nothing continuous is folded into it.
Remaining quota and provider health break ties only; letting them scale the
score would quietly turn a strict ranking back into a weighted preference.
Ties are common, because cross-provider duplicates share one score, so
remaining capacity breaks them first and the provider name breaks what is left,
which keeps the order deterministic rather than dependent on iteration order.

Two departures from a pure sort, both deliberate:

- A candidate cooling after a recent `402`/`429`, or one with no headroom left,
  is demoted to the **back** of the list. It stays reachable, so a saturated top
  pick never causes an avoidable `503`, but a strict order would otherwise
  re-attempt a rate-limited leader first on every request until its window
  cleared.
- An **unscored** candidate sorts after every scored one. Nothing can rank it on
  evidence, and promoting it would let one pin preempt a measured
  top-of-the-field model on every request.

Because the ranking is measured capability, it outranks the softer ordering
passes, which are suppressed for a ranked flagship pool: [request-fit
triage](#request-fit-triage-every-free-and-local-virtual) (whose effective key
inside a single tier is a parameter count guessed from the model id — precisely
the crude proxy the benchmark score replaces) and
[`favorite_free_models`](#favorite_free_models).
The *hard* passes still win, because they predict an outright failure rather
than a preference: forced tool/vision/JSON
[capability enforcement](#capability-enforcement), and
[context fit](#context_aware_routing).

[Free-tier cache affinity](#free_tier_cache_affinity) is **not** suppressed
here, because it no longer competes with the ranking. It pins a conversation to
the model that last served it successfully, and the first turn of a
conversation is pinned by whatever the ranking already chose — so the strongest
member is still the first pick, and stickiness only decides whether later turns
stay there. A pinned model that is cooling after a `402`/`429` is never
promoted, so affinity cannot undo a saturation demotion.

One consequence is worth stating plainly: **`flagship__free` does not spread
load.** It walks the ranking instead of sampling by remaining quota, so the
top-scored free model absorbs every request until it saturates and drops to the
back. That is the point of asking for flagship, but it is a real change from how
the other `/free` virtuals behave — and with affinity enabled a conversation
stays put even longer, by design.

If the membership cache carries no scores yet — a first run, or a file written
by an older build — the pool falls back to its previous ordering and says so in
the route reason, rather than claiming a ranking it does not have.

**Checking whether the ranking is live.** The route reason tells you outright:

```bash
curl -sS -i localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"llmproxy/flagship","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
  | grep -i 'x-llmproxy-\(route-reason\|selected-model\)'
```

`flagship_rank=4/5` means the pool was walked in benchmark order, and that four
of its five candidates carried a score. `cycling` or `capacity` means it was
not: no scores are cached yet, and the tier is serving in its pre-ranking order.
The server also logs that once, naming the cause, the first time it happens.

**Upgrading rewrites the cache on the next interval tick.** Membership is
recomputed on its own [cadence](#refresh-cadence), but a cache that predates the
scores is not a staleness problem — it is a schema one, and waiting out
`refresh_frequency_days` would leave the tier unranked for up to a week after an
upgrade. So a cached membership with no scores in it is recomputed regardless of
its timestamp, once, and the ordinary cadence resumes afterwards. A deployment
whose models no benchmark source covers is not caught by this: it records an
empty score set, which is a real answer rather than a missing one, and is left
on its normal schedule.

#### Free is per-provider

The same weights can be free on one provider and paid on another, so free
status is a property of the *routing target*, not of the model. Every
provider's instance of a flagship model is its own candidate in
`llmproxy/flagship`, and only the free ones appear in `flagship/free`.

Cross-provider duplicates count **once** toward `min_flagship_free_models`,
because three providers serving the same weights is one model's worth of
capability. They remain separate routing targets, though, since each has its
own quota, rate limits and outage profile — which is exactly what failover
needs.

#### Configuration

Paste this block at the top level of `config.json`, beside `free_tier`. Every
key is optional; a config without the block behaves as if it contained these
values, so upgrading needs no edit.

```json
"flagship_tier": {
  "enabled": true,
  "min_flagship_free_models": 5,
  "start_percentile": 0.9,
  "min_context": 200000,
  "require_tools": false,
  "max_models": null,
  "pin": [],
  "exclude": [],
  "sources": ["openrouter_aa", "epoch"],
  "refresh_frequency_days": 7
}
```

| Key | Default | What it does |
|-----|---------|--------------|
| `enabled` | `true` | Master switch. When false, no recompute runs and no network calls are made for the tier. |
| `min_flagship_free_models` | `5` | Lower the bar until at least this many **distinct** free models are *in the tier*. Cross-provider duplicates count once, and excluded ids never count. Re-checked against the live tier, so a member that stops being free triggers a recompute instead of silently shrinking the pool. Best-effort: if the free pool is smaller, you get what there is, with one warning. |
| `start_percentile` | `0.9` | Where the bar starts before floating downward. Raise it for a stricter tier; the free floor may still pull it below this. |
| `min_context` | `200000` | Spec veto: minimum context window. `0` disables the check. |
| `require_tools` | `false` | **Opt-in** spec veto restricting *membership* to tool-callers. Off by default: a request that needs tools already cannot select a model that lacks them (see [capability enforcement](#capability-enforcement)), so vetoing here as well only made the floating bar hunt further down the ranking for free models carrying the tag — admitting weaker models on the strength of a capability rather than a score. Turn it on when you want the tier itself restricted. |
| `max_models` | `null` | Optional hard cap on distinct models. `null` means uncapped. |
| `pin` | `[]` | Ids always admitted, bypassing both the bar and the spec veto. A qualified `provider/model` id pins one routing target; a bare upstream id pins every provider serving it. An entry may instead be `{"name": …, "percentile": …}` to place it in the ranking — see [saying where a pin belongs](#pin-placement). |
| `exclude` | `[]` | Qualified ids never admitted. Applied last, so it beats a pin. |
| `sources` | `["openrouter_aa", "epoch"]` | Which benchmark sources to combine. |
| `refresh_frequency_days` | `7` | How often membership is recomputed. `0` recomputes every time the interval is checked. |

Pins and excludes take effect immediately, when membership is read, rather
than waiting for the next refresh.

#### Benchmark sources

One source is wired up today:

| Source | What it is |
|--------|------------|
| `openrouter_aa` | Artificial Analysis indices embedded in OpenRouter's model catalog, under `benchmarks.artificial_analysis`. The `agentic_index` is used, being the closest published measure of tool-loop capability rather than conversational preference. |

It needs no API key of its own and no second scraper: the scores arrive inside
a model listing llmproxy already fetches, so the ids match by construction
with nothing to reconcile.

`sources` is a list because adding another is meant to be easy. Each source is
rank-normalised to a percentile before the scores are combined, so a new
source on a completely different scale cannot swamp the existing one.

**Why there is only one.** Three obvious candidates are deliberately absent:

- **LLM Stats** forbids redistribution on every tier including paid ones,
  stating that "technical access is not a redistribution license".
- **BenchLM** publishes no licence at all, which is an absence of any grant
  rather than a restrictive one.
- **Epoch AI** publishes under CC-BY and would be usable, but its `epochai`
  Python client is an Airtable ORM: it reads `AIRTABLE_PERSONAL_ACCESS_TOKEN`
  and `AIRTABLE_BASE_ID` at import time and raises without them, and its data
  model is individual benchmark *runs* rather than a leaderboard. Adding it as
  a dependency would not make the source work for anyone lacking those
  credentials — it would just fail quietly. Epoch's public CC-BY CSV export is
  the way in if you want that data, with the attribution the licence requires.

Nothing fetched from any source is committed to the repository; scores live
only in your local `flagship_models.json`.

#### Coverage, and when to use a pin

Benchmark coverage stops at the major labs. A small or direct provider —
`atria-asi`, for example — appears on no leaderboard, so nothing can score it
and it will never be admitted automatically no matter how good it is. The pin
list is the intended answer:

```json
"flagship_tier": {
  "pin": ["atria-asi/Atria-Dawn-Preview"]
}
```

**A pin may be written either way.** A qualified id pins that one routing
target; a bare upstream id pins the model on *every* provider serving it, which
is usually what "pin this model wherever I have it" means:

```json
"flagship_tier": {
  "pin": [
    "atria-asi/Atria-Dawn-Preview",   // this provider's copy only
    "gemini-3.7-flash"                // every provider serving it
  ]
}
```

A `/` cannot tell the forms apart, because upstream ids routinely contain one
(`gmi` serves `google/gemini-3.8-flash`, `openrouter` serves
`qwen/qwen3.8-27b:free`). Resolution is therefore by precedence: an **exact
qualified match first, the bare upstream id second**, so the more specific
reading always wins. This mirrors [`believed_free`](#free-tier-provenance), the key a
pin is usually paired with, which has always accepted either form.

The *normalized* key is deliberately **not** a third option. That join is
heuristic — it is what lets one benchmark score cover several spellings of the
same weights — and honouring it here would let a pin reach models you never
named. A pin is an explicit instruction, so it stays literal.

<a name="pin-placement"></a>
#### Saying where a pin belongs

Admitting a model is only half of it. Nothing scores a pinned model — that is
usually *why* you pinned it — so it sorted below every scored candidate and
ended up **last in the failover queue**. You asked for the model and got it only
after everything else had been tried.

An entry may instead be an object carrying a `percentile`:

```json
"flagship_tier": {
  "pin": [{"name": "atria-asi/Atria-Dawn-Preview", "percentile": 100}]
}
```

`100` places it first. Bare strings and objects mix freely in one list, and
**`percentile` is optional**: an entry without one behaves exactly as before,
admitted but unplaced.

| You write | Meaning |
| --- | --- |
| `"atria-asi/Atria-Dawn-Preview"` | Admit it; place it nowhere (sorts last if unscored). |
| `{"name": "…", "percentile": 100}` | Admit it and place it first. |
| `{"name": "…", "percentile": 98}` | Admit it and place it at the 98th percentile. |
| `{"name": "…"}` | Same as the bare string. |

**Either scale works.** Percentiles are `[0, 1]` internally, matching
`start_percentile`, but `98` is the natural thing to write. One rule settles it:
a value **≤ 1 is a fraction**, a value **> 1 is a percentage**. No value is
ambiguous between the two readings, and an out-of-range value such as `980` is
clamped to the top rather than wrapping.

**A percentile overrides a measured score.** A pin is an explicit instruction,
consistent with it already bypassing the bar and the spec veto, so this is also
how you *demote* a model you have reason to distrust:

```json
"pin": [{"name": "someprov/overrated-model", "percentile": 10}]
```

Because an override displacing real evidence is otherwise invisible, it is
logged once per model:

```
[flagship] pin places someprov/overrated-model at 0.100, overriding its
measured percentile of 0.950
```

A bare name places **every** provider serving those weights; a qualified id
places only that routing target. Pins resolve against the live route cache by
the same precedence as above, rather than through the normalized key — that key
strips the provider, so `atria-asi/Atria-Dawn-Preview` and
`Atria-Dawn-Preview` collapse onto the same entry, and placing a qualified pin
through it would silently move every provider serving the model.

Placement takes effect on the **next request**, not the next refresh, so you can
retune a percentile without waiting out `refresh_frequency_days`.

> **Placement is not immunity.** A candidate cooling after a `402`/`429`, or
> with no headroom left, is demoted to the back of the list whatever its rank.
> A `percentile: 100` pin cannot spend every turn retrying the one model already
> known to be rate limited, and it does not exempt the model from the spec gate
> at refresh time either — `exclude` still beats it outright.

Because `exclude` is applied last, it still beats a pin, including beating one
arm of an expanded bare pin — which is how you pin a model everywhere except on
the one provider whose copy of it is broken:

```json
"flagship_tier": {
  "pin": ["gemini-3.7-flash"],
  "exclude": ["someprovider/gemini-3.7-flash"]
}
```

<a name="flagship-trial-credits"></a>
#### Getting a pinned model into `flagship__free`

A pin alone reaches `llmproxy/flagship`, **not** `flagship__free`. The free pool
is the intersection of flagship membership and *free* models, and "free" is
decided by [`believed_free`](#free-tier-provenance) rather than by membership — so
trial credits, a promotional window or a personal allowance do not make a model
free as far as llmproxy is concerned. Say so explicitly, with both keys:

```json
"flagship_tier": { "pin": ["gmi/google/gemini-3.7-flash"] },
"believed_free": ["gmi/google/gemini-3.7-flash"]
```

There is a safety valve worth knowing about. If that model ever answers a
request reporting a real cost, it lands in `cost_observed_free_tier` and is
treated as paid from then on, whatever `believed_free` says — which is exactly
what should happen when a trial runs out.

A pin bypasses the spec veto too, since an unscraped provider has no
capability data to check. The refresh logs a warning naming any pinned id it
could not verify, so an unnoticed typo does not silently do nothing.

**A pin buys membership, not rank.** Scores belong to the model rather than to
the provider serving it, so a pinned provider whose weights *are* scored under
some other provider's listing inherits that score and takes its rightful place
in the [ordering](#flagship-ordering) — which is the common case, since the join
is on normalised model names. Only a pin whose weights nothing scores at all is
unrankable, and that one sorts last, behind every measured member. It is still
reached by failover; it is simply not chosen ahead of a model that was actually
measured. To make such a model a first pick, address it by name, or use its
per-provider virtual `llmproxy/<provider>__flagship`.

Where a provider does not publish capability data but serves the same weights
as a provider that does, the specs and score are carried across by matching
normalised model names. That join is a heuristic and can be wrong; a pin or an
exclude is the way to overrule it.

**One thing is never carried across: a billing variant's specs.** When a
catalog or a provider lists `vendor/model:free` separately from
`vendor/model`, it is distinguishing two sets of endpoints rather than being
terse about one, so what it says about the variant is taken to be about the
variant alone. The *score* still joins, because the weights really are the
same; the tool-support and context-window **specs** do not. Only ids carrying a
recognised variant suffix (`:free`, `:batch`, `:nitro`, `:extended`, `:floor`)
are treated this way, so an ordinary model whose gateway publishes a thin
listing still inherits everything known about those weights elsewhere.

#### When the tier is empty

Until the first refresh completes — or if nothing qualifies — `llmproxy/flagship`
is hidden from `GET /v1/models`, and requesting it directly returns `503` with a
message explaining that membership is computed and pointing at the pin list.

It deliberately does **not** fall back to `deep`. Silently serving a weaker
model would defeat the purpose of asking for flagship, and would be impossible
to notice.

#### Refresh cadence

Membership is recomputed automatically: once at startup, and thereafter
whenever `refresh_frequency_days` has elapsed, checked on the same interval
tick as the other background jobs. The last-run timestamp lives in
`flagship_models.json`, so restarting the server does not trigger a fresh
recompute on every boot.

**If `flagship_models.json` does not exist yet**, the tier is treated as never
refreshed, which is always due — so a fresh deployment populates it on first
boot regardless of `refresh_frequency_days`. You do not need to set the
frequency to `0` to get a first run. The same rule is why the
[state directory](#state-directory) has to be writable: a timestamp that can
never be recorded would leave the tier permanently due.

This is a **separate cadence** from the free-models sweep
(`free_tier.update_frequency_days`), with its own setting and its own state
file. See [There are three independent cadences](#the-three-cadences).

The candidate pool is everything this deployment can actually reach: every
model of every configured provider, paid included. That is why the list is
deployment-specific, and why it changes when you add a provider.

<a name="pr-providers-list"></a>
### Proposing `providers.json` changes as a PR — `providers_pr.enabled`

When `free_tier.update_on_startup` refreshes the sidecar (optionally with
probing, if `free_tier.probe.enabled` / `free_tier.probe.autoremove` are on), set this flag to
have the running deployment open a **pull request** with the result instead of
only keeping the change in its ephemeral local copy:

```json
{
  "free_tier": { "update_on_startup": true },
  "providers_pr": {
    "enabled": true,
    "repo": "BillJr99/llmproxy",
    "base": "main",
    "branch": "llmproxy-auto/providers",
    "token": "${GITHUB_TOKEN}"
  }
}
```

The GitHub token is required. Provide it either via the `providers_pr.token`
config key (a literal token or a `${VAR}` reference, as above) **or** via the
`GITHUB_TOKEN` / `GH_TOKEN` environment variable. It needs `contents:write` +
`pull_requests:write` on the target repo.

When `true` **and** the startup run produced a `providers.json` that differs from
the bundled copy, the server pushes `llmproxy/providers.json` + `config.example.json`
to a branch and opens (or refreshes) a PR against the base branch — using the
GitHub API directly, so it **never touches a local git checkout** and works even
in a container with no `.git`. It logs `[providers-pr] opening PR …` and the PR URL.

Before pushing, the server folds what this deployment learned into the
`providers.json` it is about to propose, and the PR body breaks the promoted
facts down by provider and by provenance grade. See
[what gets PR'd](#routing-metadata).

This works **even when the bundled `providers.json` can't be saved locally**, for
example on a read-only container image. The computed `providers.json` and
`config.example.json` are held in memory for the run and the PR is opened from
that content; nothing is mirrored into the config directory, because a second
copy of `providers.json` beside your `config.json` would shadow nothing, drift
from the shipped file immediately, and invite hand-edits to a machine-written
file. (See also: the cost probe's `free_tier.probe.frequency_days` throttle so a
startup that probes + PRs doesn't spend quota or churn a PR on every restart.)

Required / optional settings (all top-level):

| Key | Required | Default | Meaning |
|-----|----------|---------|---------|
| `providers_pr.enabled` | — | `false` | Master switch. |
| `providers_pr.repo` | **yes** | — | Target repo as `"owner/repo"`. |
| `providers_pr.token` | yes¹ | — | GitHub token; may be a `${VAR}` ref. ¹Falls back to the `GITHUB_TOKEN` / `GH_TOKEN` environment variables. Needs `contents:write` + `pull_requests:write`. |
| `providers_pr.base` | — | `"main"` | Base branch for the PR. |
| `providers_pr.branch` | — | `"llmproxy-auto/providers"` | Head branch (force-updated each run; an open PR for it is reused). |
| `providers_pr.frequency_days` | — | `0` | Throttle: open at most one PR every _N_ days, with the last-run timestamp in `pr_state.json`. `0` (the default when the key is absent) means no throttle, and the admin API's `pr_providers_frequency_days` field shows the same `0`. |

If the token or `providers_pr.repo` is missing, the server logs a `[providers-pr]`
warning and skips — it never fails the startup update. This is the **deployment**
counterpart to the repo-level
[CI auto-update workflow](#automated-providersjson-updates-ci): the workflow
proposes PRs from a scheduled scrape, while `providers_pr.enabled` proposes them from
a live deployment (which can additionally probe real model costs).

### Provider templates

Provider templates and free-tier metadata both live in
[`llmproxy/providers.json`](llmproxy/providers.json) — the single source of
truth. The setup wizard reads from this file at startup; `config.example.json`
is regenerated from the same file. To add or update a provider, edit
`providers.json` directly (or run the scraper — see
[Keeping the free-models list current](#keeping-the-free-models-list-current)).

The wizard currently offers ready-made templates for these providers:

| Provider                                   | Default key             | Base URL                                                                       |
|--------------------------------------------|-------------------------|--------------------------------------------------------------------------------|
| OpenAI                                     | `openai`                | `https://api.openai.com/v1`                                                    |
| Nous Research (Hermes)                     | `nous`                  | `https://inference-api.nousresearch.com/v1`                                    |
| Nvidia NIM                                 | `nvidia`                | `https://integrate.api.nvidia.com/v1`                                          |
| Google Gemini (via OpenAI-compat endpoint) | `google`                | `https://generativelanguage.googleapis.com/v1beta/openai`                      |
| Cerebras                                   | `cerebras`              | `https://api.cerebras.ai/v1`                                                   |
| SambaNova Cloud                            | `sambanova`             | `https://api.sambanova.ai/v1`                                                  |
| Mistral AI                                 | `mistral`               | `https://api.mistral.ai/v1`                                                    |
| Groq                                       | `groq`                  | `https://api.groq.com/openai/v1`                                               |
| Together AI                                | `together`              | `https://api.together.xyz/v1`                                                  |
| Fireworks AI                               | `fireworks`             | `https://api.fireworks.ai/inference/v1`                                        |
| Cloudflare Workers AI                      | `cloudflare-workers`    | `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1`             |
| Zhipu AI (BigModel)                        | `zhipu`                 | `https://open.bigmodel.cn/api/paas/v4`                                         |
| Z.AI                                       | `z-ai`                  | `https://api.z.ai/api/paas/v4`                                                 |
| DeepSeek                                   | `deepseek`              | `https://api.deepseek.com/v1`                                                  |
| Cohere                                     | `cohere`                | `https://api.cohere.com/compatibility/v1`                                      |
| OpenRouter                                 | `openrouter`            | `https://openrouter.ai/api/v1`                                                 |
| Requesty (LLM router)                      | `requesty`              | `https://router.requesty.ai/v1`                                                |
| B.AI (unified LLM API)                     | `bai`                   | `https://api.b.ai/v1`                                                          |
| xKiro (multi-vendor gateway)               | `xkiro`                 | `https://api.xkiro.com/v1`                                                     |
| TeamoRouter (LLM routing gateway)          | `teamorouter`           | `https://api.teamorouter.com/v1`                                               |
| Token Harbor (multi-vendor gateway)        | `tokenharbor`           | `https://tokenharbor.ai/v1`                                                    |
| Ollama Cloud                               | `ollama-cloud`          | `https://ollama.com/v1`                                                        |
| Moonshot AI (Kimi)                         | `moonshot`              | `https://api.moonshot.ai/v1`                                                   |
| MiniMax                                    | `minimax`               | `https://api.minimax.io/v1`                                                    |
| Atria ASI                                  | `atria-asi`             | `https://api.atria-asi.ai/v1`                                                  |
| Unbiased AI                                | `unbiased-ai`           | `https://api.unbiased.ai/v1`                                                   |
| Kilo AI (multi-vendor gateway)             | `kilo`                   | `https://api.kilo.ai/api/gateway`                                             |
| ModelScope (Alibaba API-Inference)         | `modelscope`            | `https://api-inference.modelscope.cn/v1`                                       |
| Aion Labs                                  | `aion-labs`             | `https://api.aionlabs.ai/v1`                                                   |
| Agnes AI (multimodal gateway)              | `agnes-ai`              | `https://apihub.agnes-ai.com/v1`                                               |
| Hugging Face Inference                     | `huggingface`           | `https://router.huggingface.co/v1`                                             |
| xAI (Grok)                                 | `xai`                   | `https://api.x.ai/v1`                                                          |
| Cloudflare AI Gateway                      | `cloudflare-ai-gateway` | `https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/workers-ai/v1` |
| Vercel AI Gateway                          | `vercel`                | `https://ai-gateway.vercel.sh/v1`                                              |
| Venice AI                                  | `venice`                | `https://api.venice.ai/api/v1`                                                 |
| OpenCode Zen (free gateway)                | `opencode-zen`          | `https://opencode.ai/zen/v1`                                                   |
| Scaleway Generative APIs                   | `scaleway`              | `https://api.scaleway.ai/v1`                                                   |
| OVHcloud AI Endpoints                      | `ovhcloud`              | `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1`                             |
| Pollinations (free, no key)                | `pollinations`          | `https://text.pollinations.ai/openai`                                          |
| Nebius AI Studio                           | `nebius`                | `https://api.studio.nebius.com/v1`                                             |
| Novita AI                                  | `novita`                | `https://api.novita.ai/v3/openai`                                              |
| Hyperbolic                                 | `hyperbolic`            | `https://api.hyperbolic.xyz/v1`                                                |
| DeepInfra                                  | `deepinfra`             | `https://api.deepinfra.com/v1/openai`                                          |
| GMI Cloud                                  | `gmi`                   | `https://api.gmi-serving.com/v1`                                               |
| LLM7 (free, no key)                        | `llm7`                  | `https://api.llm7.io/v1`                                                       |
| Chutes AI                                  | `chutes`                | `https://llm.chutes.ai/v1`                                                     |
| Meta Llama API                             | `meta-llama`            | `https://api.llama.com/compat/v1`                                              |
| Anthropic (Claude, native Messages API)    | `anthropic`             | `https://api.anthropic.com/v1`                                                 |
| Google Gemini (native generateContent)     | `gemini`                | `https://generativelanguage.googleapis.com/v1beta`                             |

> **API keys.** Most providers in this table require an API key, and the setup
> wizard displays a hint showing where to obtain each one. The exceptions are
> **Pollinations**, **LLM7**, and **OVHcloud AI Endpoints**, which serve an
> anonymous tier — leave the key blank (OVHcloud accepts an optional key for
> higher limits). **Anthropic** and **Google Gemini (native)** are not
> OpenAI-compatible; their templates set `protocol` so llmproxy translates to
> the upstream Messages / generateContent API. For keyless local access (e.g. a
> local Ollama instance), use the manual "Add / edit a provider" option in the
> wizard.

> **Retired providers.** **GitHub Models** (`github`,
> `https://models.github.ai/inference`) was retired by GitHub on 30 July 2026 and
> has been removed from the bundled templates. Its catalog and inference
> endpoints now return HTTP 410, and there is no drop-in replacement: GitHub
> directs users to Microsoft (Azure) AI Foundry or GitHub Copilot, neither of
> which offers an OpenAI-compatible free tier. If an existing `config.json` still
> carries a `github` provider block, it is inert — the catalog fetch returns
> nothing, so no candidates enter the rotation — but you can remove it, along
> with any `github/`-prefixed entries under `believed_free`, `model_reasoning`,
> `model_capabilities` and `free_limits`, to stop the recurring
> `/models` fetch warning in the log.

Any OpenAI-compatible provider can also be added manually via the "Add / edit a
provider (manual)" menu option.

#### Adding a provider from a shell — `scripts/add_provider.sh`

For scripted or headless setups, `scripts/add_provider.sh` writes the same config
entry the wizard would, without launching the TUI:

```
scripts/add_provider.sh
```

It lists every template from `llmproxy/providers.json`, then prompts for the
provider name (the model-ID prefix), the base URL, any `{account_id}` /
`{gateway_id}` substitutions, the API key, and the config file path — defaulting
to `$LLMPROXY_CONFIG` or `~/.config/llmproxy/config.json`, and offering to create
it if missing.  It carries `protocol`, `models_url`, `models_id_field`,
`models_keep_task`, `_note` and `example_model_filter` across from the template,
backs the old config up to `config.json.bak`, writes atomically, and leaves the
result `chmod 600`.  Other providers and every other top-level key are preserved,
and re-running it updates just the one entry.  A `${VAR}` reference typed at the
key prompt is stored verbatim and resolved from the environment at request time.

It finishes by calling `<base_url>/models` to report how many models the provider
advertises — a non-200 is reported but does not block the write.  If the provider
you pick is missing from this checkout's `providers.json` (the script carries its
own copy of the newest templates), it offers to add it there and to
`provider_order`, then regenerates `config.example.json`.

> **Providers that do not support a standard `GET <base_url>/models` (as of June 2026)**
> Some providers return an error or non-JSON response for the default `/models`
> path. There are two ways to handle these:
>
> 1. **Point discovery at the real catalog** with the `models_url` /
>    `models_id_field` / `models_keep_task` overrides (see
>    [Schema](#schema)). The bundled template for **Cloudflare Workers AI**
>    already does this, so its models are discovered live.
> 2. **Synthesize from `model_filter`** — when no working catalog endpoint
>    exists, set `model_filter` to the upstream ids you want and llmproxy
>    advertises those when the `/models` fetch fails.
>
> | Provider | Default `/models` symptom | Handling |
> |---|---|---|
> | **Cloudflare Workers AI** | HTTP 405 — no `GET /v1/models` | `models_url` → `…/ai/models/search`, `models_id_field: "name"`, `models_keep_task: "Text Generation"` |
> | **Cloudflare AI Gateway** | HTTP 401 — gateway proxies inference only, no catalog | `model_filter` (synthesized); a 401 also means the API token is missing/under-scoped for Workers AI |
> | **Hugging Face Inference** | Returns HTML rather than JSON for `/v1/models` | `model_filter` (synthesized) |
> | **Unbiased AI** | HTTP 404 `unknown_url` — no catalog endpoint. An *unauthenticated* probe returns 401 on every path, because the gateway checks the key before routing, so a 401 there means a missing key rather than a missing endpoint | `model_filter: ["pareto"]` (synthesized) |
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
writes through a JSON API under `/admin/api/*`. Server settings and providers go
straight to `config.json`; the four model categorizations go to the curated
layer of `routing_metadata.json` instead, because that is where hand-set routing
facts live (see [where routing metadata lives](#routing-metadata)). Changes take
effect without a restart (host/port changes excepted), because every worker
re-reads both files when they change.

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
their rate limits are — used by the `llmproxy/free` virtual endpoint and by the
setup wizard's "auto-populate" step.

A scraper at `scripts/update_free_models.py` polls multiple sources, diffs the
result against the sidecar, and prints proposed adds / removes / limit changes
for human review.

You rarely need to run it by hand. A running proxy runs the same scraper on the
cadence set by [`free_tier.update_frequency_days`](#refresh-cadence), so its live
view of what is free stays current on its own; the manual and CI paths exist to
land those changes durably in the repository.

### Sources

| Source       | Confidence | What it does |
|--------------|------------|--------------|
| `openrouter` | high       | Hits `https://openrouter.ai/api/v1/models` and flags any model with `pricing.prompt == 0` as free, whether or not its id carries a `:free` suffix — this is what catches unsuffixed cloaked models. Also reports per-token prices for paid models into the sidecar `pricing` block, and, because the endpoint is the gateway's full catalog, drives removals for models withdrawn upstream. |
| `docs`       | high       | Per-provider HTML scrapers for published rate-limit / free-tier pages (Google, Groq, Cerebras, Cohere, Token Harbor). Add more under `scripts/sources/docs/`. |
| `api`        | medium     | Calls each provider's OpenAI-compatible `/v1/models` endpoint when `<PROVIDER>_API_KEY` is set in your environment. One of the sources that can detect *removals* — see [Removing withdrawn models](#removing-withdrawn-models). |
| `litellm_cost_map` | medium | Reads the public [litellm](https://github.com/BerriAI/litellm) pricing map: flags zero-priced models as free **and** snapshots per-token prices for paid ones into the sidecar `pricing` block (used by the proxy to cost tokens offline — see [Token + cost accounting](#usage-accounting)). |
| `together`   | high       | When `TOGETHER_API_KEY` is set, reads Together's `/v1/models` pricing — zero-priced models are free; paid models contribute per-token prices to the `pricing` block. |
| `fireworks`  | high       | When `FIREWORKS_API_KEY` is set, reads Fireworks' `/inference/v1/models` and flags models marked `is_free`/`serverless_billing: free` or zero-priced as free. |
| `requesty`   | high       | When `REQUESTY_API_KEY` is set, reads Requesty's `/v1/models` pricing — zero-priced models are free; paid models contribute per-token prices to the `pricing` block. |
| `xkiro`      | high       | Reads xKiro's public `/v1/models` catalog. No API key is required, so this one also runs in CI: models marked `access_tier: "free"` with zero input/output price are flagged free, paid models contribute per-token prices to the `pricing` block, and any `believed_free` id the catalog has dropped is reported as no longer free. |
| `community`  | low        | Pulls the [tashfeenahmed/freellmapi](https://github.com/tashfeenahmed/freellmapi) community list as a sanity signal. |
| `probe`      | high · **opt-in** | Sends a tiny real chat request to each `believed_free` model and flags any that report a cost. Off by default; enable with `probe_cost: true` in `config.json` or the `--probe` flag. Spends a little quota. |

The top-level **`pricing`** block is assembled from several of these sources: the
litellm cost map provides broad baseline coverage, and high-confidence live
provider sources (OpenRouter, Together) override individual models with their
authoritative per-token prices. The result powers offline cost accounting and the
[`llmproxy/loadbalanced`](#the-loadbalanced-virtual-model) paid-tier ranking, and
is committed alongside `believed_free` in the same providers.json refresh (and the
[automated PR](#keeping-the-free-models-list-current), when enabled).

<a name="removing-withdrawn-models"></a>
### Removing withdrawn models

A model leaves `believed_free` for one of three reasons. The first two are
straightforward: a high-confidence source reports a non-zero price for it, or the
proxy observed it billing a real cost at runtime and recorded it in
`cost_observed_free_tier` (see
[Verifying free models are actually free](#cost-flags)).

The third is absence. Short-lived models, cloaked previews especially, tend not
to be repriced when they end; they simply stop being listed. A model that no
source mentions produces no evidence at all, so absence is only treated as
removal under two conditions:

1. the model is missing from a source that enumerates the provider's **entire**
   catalog, currently OpenRouter's `/api/v1/models` and any provider's
   `/v1/models` listing via the `api` source. A docs scraper reads a free-tier
   page rather than a catalog, so its silence about a model means nothing and
   never removes anything; and
2. that catalog response looks whole. A response is trusted when it either lists
   a substantial number of models outright, or still accounts for at least half
   of what is currently believed free for that provider. A truncated or degraded
   fetch falls below that floor, and the run logs that it is skipping
   absence-based removal for the provider rather than emptying the list.

Together these mean a cloaked model that appears at `$0`, is used for a while,
and then disappears will be added and later dropped without anyone editing
`providers.json` by hand.

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

# Probes run with bounded per-provider concurrency (default 3) and show a
# progress bar if `tqdm` is installed. Tune the per-provider cap to stay under a
# provider's rate limit:
python scripts/update_free_models.py --probe --probe-concurrency 2
```

### Verifying free tiers and auto-removal (`free_tier.probe.enabled` / `free_tier.probe.autoremove`)

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
[`free_tier.update_on_startup`](#update-on-startup).

### Syncing your live config (`--config PATH`)

There is nothing left to sync. Copying the sidecar's free-tier sections into
`config.json` is what froze them: it ran once and the data never moved again,
which is how a deployment ended up routing on capability tags years out of date.
`providers.json` is now read directly as the defaults layer, the refresh keeps
`routing_metadata.json` current above it, and hand-set facts live in that file's
curated section. See [where routing metadata lives](#routing-metadata).

`--config PATH` and `--sync-config-only` are still accepted so existing scripts
and cron entries do not break, and the run prints a line saying the sync is no
longer needed. `--config PATH` is still worth passing for everything else it
does: it is where the run reads your `free_tier` settings (whether the cost
probe is enabled, whether autoremove is on, the shared probe timeout), where it
resolves the [state directory](#state-directory) from when `LLMPROXY_STATE_DIR`
is unset (which is what holds `cost_probe_state.json` and `update_state.json`),
and which API keys the sources may use.

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

A GitHub Actions workflow,
[`.github/workflows/update-providers.yml`](.github/workflows/update-providers.yml),
keeps the sidecar current **in the repository** without anyone running the
scraper by hand. It is **manual only**: there is no schedule, so it runs exactly
when you trigger it from the Actions tab (or with
`gh workflow run update-providers.yml`). When it runs it:

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

**Running it on a schedule (optional).** Most deployments do not need this: a
running proxy already refreshes itself on its own cadence, as described in
[Refresh cadence](#refresh-cadence) below, and the workflow exists to land those
same updates in the repository as a reviewable PR. If you would rather the repo
refresh itself without being asked, add a `schedule:` block to the workflow's
`on:` section, for example weekly on Monday at 06:00 UTC:

```yaml
on:
  workflow_dispatch: {}
  schedule:
    - cron: "0 6 * * 1"
```

The workflow needs `contents: write` and `pull-requests: write` permissions
(already declared in the file); if your organization disables PR creation by
`GITHUB_TOKEN`, enable it under *Settings → Actions → General → Workflow
permissions*.

> This repo-level workflow and the server-side refresh are complementary: the
> workflow lands durable updates in the repo via reviewable PRs, while the
> running proxy refreshes its own live config on the cadence set by
> [`free_tier.update_frequency_days`](#refresh-cadence).

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

The server hot-reloads config on each request (a cache keyed on the file's
`(st_mtime_ns, st_size)` fingerprint, so an edit is picked up reliably even on
filesystems with coarse mtime granularity), so provider changes take effect
immediately without a restart.  Only `host` or `port` changes require a restart.

---

## Tests, dev tooling, and CI

The repo has three distinct things named "test"-ish — each does something
different:

| File                       | What it is                                                                     |
|----------------------------|--------------------------------------------------------------------------------|
| `tests/`                   | The pytest unit/integration suite (run with `pytest`). Fully offline — every upstream is stubbed. |
| `test.sh`                  | Runs the offline suite, then smoke-tests a **running** server with curl.        |
| `llmproxy_test_client.py`  | Live integration test client. Talks to a running llmproxy over HTTP.           |
| `test_tui.py`              | Interactive chat TUI for hand-driving the proxy (despite the misleading name). |

### Running the unit suite

```bash
pip install -r requirements-dev.txt
pytest                                  # run everything
pytest --cov=llmproxy --cov=scripts     # with coverage
pytest tests/test_scraper                # just the scraper tests
ruff check llmproxy scripts tests        # lint
./test.sh                               # offline suite, then live curl checks
UNIT_ONLY=1 ./test.sh                   # offline suite only (no server needed)
SKIP_UNIT=1 ./test.sh                   # live checks only
```

The suites covering routing and streaming behavior are worth knowing by name,
because each was written around a failure mode rather than around a function:

| Suite | Covers |
| --- | --- |
| `tests/test_cycling_robustness.py` | The core failover classes: 200-with-error bodies, empty completions, forced-capability misses, quota cooldowns, route headers. |
| `tests/test_stream_hardening.py` | Terminal frames and health demotion on a mid-stream death, the identity fast path's status check, the pre-commit window, whole-response buffering, and the cycle deadline. Each gated behavior is asserted **off** by default as well as on. |
| `tests/test_context_routing.py` | Context-window fit: the demote-never-drop and unknown-is-neutral invariants, the `model_context` override, and the discovery cache. |
| `tests/test_capability_states.py` | Three-valued capability ordering, `cache_control` round-tripping, the free-tier affinity switch, and the worker default. |
| `tests/test_responses_dialect.py` | The Responses API: input-item mapping, output fan-out, the typed streaming events, and the conversation store. |

CI runs the same checks on every push and pull request — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). It runs:
- `pytest` across Python 3.11 and 3.12 (new files under `tests/` are collected
  automatically via the `testpaths` setting in `pyproject.toml`),
- `ruff` lint,
- `bash -n test.sh`, since that script ships with the repo and is a documented
  entry point but is never executed in CI (it needs a live server),
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
| `free`      | Sends several prompts to `model="llmproxy/free"`; tests cycling + streaming   | Yes (free tier)  |
| `local`     | Sends several prompts to `model="llmproxy/local"`; skipped if none configured | Yes (localhost)  |
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

Config is bind-mounted from `~/.config/llmproxy` on the host. The image runs as
a non-root user by default (no `--user` required); passing `--user $(id -u):0`
makes files created inside the container owned by you on the host.

```bash
mkdir -p ~/.config/llmproxy

docker run -it --rm \
  --user $(id -u):0 \
  -v ~/.config/llmproxy:/config \
  -e LLMPROXY_CONFIG=/config/config.json \
  llmproxy --setup
```

<a name="docker-user-group"></a>
#### Pass `:0` as the group, not `$(id -g)`

The group matters as much as the user. The image builds its unprivileged user
into group 0 and hands that group every directory the process writes:

```dockerfile
RUN useradd --uid 1000 --gid 0 --create-home --home-dir "$HOME" llmproxy \
    && mkdir -p /config /state "$HOME/.config/llmproxy" \
    && chgrp -R 0 /config /state "$HOME" /app \
    && chmod -R g+rwX /config /state "$HOME" /app
USER 1000:0
```

Group ownership rather than user ownership is what lets the image run under an
arbitrary UID, which is the same convention OpenShift requires: the UID may be
anything, so nothing useful can be keyed to it, but the GID is fixed at 0 and
the four directories above are group-writable. Those four are the whole set:
`/config`, `/state`, `/home/llmproxy` and `/app`.

`--user $(id -u):$(id -g)` breaks that. A host GID, normally 1000, is in neither
group 0 nor the owner of those directories, so only the "other" permission bits
apply, and `chmod g+rwX` never set a write bit for "other". The container can
read everything and write nothing.

`:0` confers no root privilege here, because the UID is still yours. It names
the group those directories are shared with. The symptom of getting this wrong
is a permission error against `/app` in the logs, from the free-models sweep
trying to refresh the bundled `providers.json`.

If the host directory was created by Docker rather than by `mkdir`, Docker made
it root-owned and nothing in the container can write it. Fix the ownership
rather than the `--user` flag:

```bash
sudo chown -R $(id -u):$(id -g) ~/.config/llmproxy
```

### Start the server

```bash
docker run -d \
  -p 8080:8080 \
  --user $(id -u):0 \
  -v ~/.config/llmproxy:/config \
  -v llmproxy_state:/state \
  -e LLMPROXY_CONFIG=/config/config.json \
  --name llmproxy \
  llmproxy
```

The `llmproxy_state` volume holds what this deployment learns; see
[where the machine-written state lives](#state-directory) for what goes in it
and why it is worth keeping. Omitting it is fine for a quick trial, in which
case the state lives inside the container and is lost when the container is
removed.

The [web admin UI](#web-admin-ui) is available on the same published port at
`http://localhost:8080/admin`. Because the container binds `0.0.0.0`, set an
admin token to allow access (the API otherwise serves loopback only), and use
`${VAR}` references in `config.json` to keep credentials in the environment
rather than the bind-mounted file:

```bash
docker run -d \
  -p 8080:8080 \
  --user $(id -u):0 \
  -v ~/.config/llmproxy:/config \
  -v llmproxy_state:/state \
  -e LLMPROXY_CONFIG=/config/config.json \
  -e LLMPROXY_ADMIN_TOKEN=choose-a-strong-token \
  -e OPENAI_API_KEY=sk-… \
  --name llmproxy \
  llmproxy
```

### Reconfigure without stopping the server

```bash
docker run -it --rm \
  --user $(id -u):0 \
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
of `llmproxy/local` routing, so the `__local` virtual model picks it up
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
  -v llmproxy_state:/state \
  --name llmproxy \
  llmproxy
```

No `--user` flag appears here because the image already runs as its own
non-root user; the flag is only needed to match ownership with a host directory,
and a named volume has no host-side ownership to match.

<a name="state-directory"></a>
### Where the machine-written state lives

llmproxy keeps two kinds of file, and the distinction decides which directory
each belongs in. `config.json` is yours: you write it, by hand or through the
wizard or the admin UI, and llmproxy only rewrites it to heal a missing field or
to drain the five routing keys out of it on first boot. Everything else is the
proxy's own working memory, rewritten on its own cadence, and never meant to be
edited by hand.

| File | Directory | What it is |
|---|---|---|
| `config.json` | config | Your configuration, plus its `.lock` and the `config.json.backup-*` taken before a routing-key migration |
| `routing_metadata.json` | state | The learned and curated routing layers, plus its `.lock` |
| `flagship_models.json` | state | This deployment's computed [flagship tier](#flagship-tier) |
| `update_state.json` | state | When the [free-models sweep](#the-three-cadences) last ran |
| `cost_probe_state.json` | state | When the cost probe last ran |
| `pr_state.json` | state | When a [providers PR](#pr-providers-list) was last opened |

The state directory is `$LLMPROXY_STATE_DIR` when that is set, and otherwise the
directory holding `config.json`, which is where every one of these files lived
historically. Setting the variable on an existing deployment does not reset
anything: each file is read forward from its old location on first use and
written to the new one.

**The state directory has to be writable.** Each of the last four files carries
the last-run timestamp that throttles its own refresh, and a refresh with no
recorded timestamp is due immediately. A directory that cannot be written used
to mean no timestamp was ever recorded, so every background refresh was
permanently due and the once-a-minute interval check re-ran a full provider
scrape for as long as the process lived. llmproxy now keeps the timestamps in
memory when it cannot persist them, which holds the cadences for that process,
and says so once at startup:

```
WARNING  State directory /config is not writable by uid 1000:1000 …
```

That is a warning rather than a fatal error, because a proxy that routes is more
useful than one that refuses to start. It is still worth fixing: a deployment in
that state relearns everything on every restart. See
[pass `:0` as the group](#docker-user-group) for the usual cause in Docker.

Separating the two directories also lets the config mount be read-only:

```bash
docker run -d \
  -p 8080:8080 \
  --user $(id -u):0 \
  -v ~/.config/llmproxy:/config:ro \
  -v llmproxy_state:/state \
  -e LLMPROXY_CONFIG=/config/config.json \
  --name llmproxy \
  llmproxy
```

The image sets `LLMPROXY_STATE_DIR=/state` by default. To put the state back
beside `config.json` instead, pass `-e LLMPROXY_STATE_DIR=/config` and drop the
state volume. Note that a read-only config mount also disables the admin UI's
config editing and the startup config heal, both of which write `config.json`.

#### The `/state` volume mount

`/state` exists in the image as an ordinary directory, so a container that
mounts nothing there still starts and still works. What it loses is
persistence: the directory then lives in the container's writable layer, and
`docker rm` takes it with the container. The proxy comes back up having
forgotten its routing metadata, its flagship membership and every refresh
timestamp, so all three background refreshes run again on the next boot.

Mounting a named volume is what makes that state outlive the container:

```bash
-v llmproxy_state:/state
```

Docker creates the volume on first use; nothing needs to be set up in advance.
It survives `docker stop`, `docker rm`, and re-creating the container against a
newer image, which is what makes an upgrade cheap. `docker volume rm
llmproxy_state` is the deliberate way to discard it, and costs only a relearn.

A host path works too, if you would rather see the files directly. It has to be
writable by the container UID, exactly like the config mount:

```bash
mkdir -p ~/.local/state/llmproxy
docker run ... -v ~/.local/state/llmproxy:/state ...
```

To look inside a named volume, or to back it up:

```bash
# List what is in there
docker run --rm -v llmproxy_state:/state alpine ls -la /state

# Copy it out to the current directory
docker run --rm -v llmproxy_state:/state -v "$PWD":/backup alpine \
  tar czf /backup/llmproxy_state.tgz -C /state .
```

None of it is secret and none of it is irreplaceable: every file is something
llmproxy worked out for itself and can work out again. Keeping it is about not
re-scraping every provider on every restart, not about protecting data.

One more file sits outside both directories: the bundled
`llmproxy/providers.json`, which ships inside the image and is refreshed in
place by the free-models sweep. On a read-only image layer that write fails,
which is expected and harmless; the sweep computes its update in memory, routing
uses it for that run, and a [providers PR](#pr-providers-list) can still be
opened from it.

---

## docker-compose

The `docker-compose.yml` at the repository root bind-mounts `~/.config/llmproxy`
from the host, keeps the machine-written state in an `llmproxy_state` named
volume, and runs containers as your UID in group 0 (see
[pass `:0` as the group](#docker-user-group) for why the group is 0 rather than
your own). Create a `.env` file so Compose picks up your UID, and create the
config directory yourself so Docker does not create it as root:

```bash
printf "UID=%s\n" "$(id -u)" > .env
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
  --user $(id -u):0 \
  -v ~/.config/llmproxy:/config \
  -e LLMPROXY_CONFIG=/config/config.json \
  ghcr.io/billjr99/llmproxy:latest --setup

# Start the server
docker run -d -p 8080:8080 --restart always \
  --user $(id -u):0 \
  -v ~/.config/llmproxy:/config \
  -v llmproxy_state:/state \
  -e LLMPROXY_CONFIG=/config/config.json \
  --name llmproxy \
  ghcr.io/billjr99/llmproxy:latest
```

`--restart always` brings the proxy back after a daemon restart or a reboot,
which is what you want for a long-lived deployment. Everything llmproxy has
learned survives that restart because it is in the `llmproxy_state` volume
rather than the container; see
[where the machine-written state lives](#state-directory).

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
| GET    | [`/v1/providers`](#v1-providers-and-v1-config) | Configured providers, no credentials |
| GET    | [`/v1/config`](#v1-providers-and-v1-config)    | Effective settings, no credentials   |
| GET    | [`/v1/usage`](#usage-endpoint) | Token, cost and health accounting |
| GET    | [`/v1/failures`](#v1-failures) | Which models have been failing, and why |
| POST   | `/v1/chat/completions`  | Chat completions (streaming supported)    |
| POST   | `/v1/completions`       | Legacy text completions (chat fallback)   |
| POST   | `/v1/responses`         | Responses API (streaming supported)       |
| GET    | `/v1/responses/<id>`    | Stored-conversation lookup                |
| DELETE | `/v1/responses/<id>`    | Forget a stored conversation              |
| POST   | `/v1/embeddings`        | Embeddings                                |
| *      | `/v1/<anything>`        | Pass-through to upstream (see note below) |

For pass-through endpoints not listed above (e.g., `/v1/audio/transcriptions`),
the proxy routes based on the `model` field in the request body.  For
GET/DELETE requests without a model field, append `?provider=<name>` to the URL.

<a name="v1-providers-and-v1-config"></a>
### `GET /v1/providers` and `GET /v1/config` — ask llmproxy about itself

Both are read-only, need no auth, and carry **no credentials**.

```bash
curl -s localhost:8080/v1/providers | jq
curl -s localhost:8080/v1/config    | jq
```

`/v1/providers` lists every configured provider with the facts that explain its
behaviour rather than the ones that would let you impersonate it: base URL,
how many models it is currently serving, how many accounts it has, its
`model_filter`, and the two switches that account for nearly every "why is this
provider missing from my pool" question,
`expose_to_virtual_models` and whether its base URL
is local (local providers route via `llmproxy/local`, never `/free`).

`/v1/config` reports the **effective** configuration, which is not the same as
the contents of `config.json`. The five routing keys are merged from four layers
before the router sees them, so reading `config.json` alone shows a deployment
as having no free models and no capability data at all. Those keys are reported
by **size**, not listed: they run to thousands of entries, and dumping them is a
different request that [`/admin/api/routing-metadata`](#web-admin-ui) already
serves, with editing.

> **Neither endpoint emits an API key, not even masked.** A mask still leaks its
> last characters, and these endpoints answer to anyone who can reach the port.
> Whether a credential is configured is the only fact about it worth publishing,
> and it is reported as a bool (`api_key_set`, `admin.token_set`). A field whose
> *name* looks credential-shaped is dropped from the `server` block outright, so
> a setting added there later cannot leak by default. To read or edit the real
> configuration, masks included, use the token-gated
> [web admin API](#web-admin-ui).

Both routes still honour `?provider=<name>` as a pass-through to that upstream's
own `/v1/providers` or `/v1/config`, so adding them cannot break a client that
was relying on the previous behaviour. Without the parameter, the question is
about llmproxy and is answered locally.

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
  aggregate response rather than causing an overall failure.  Each gunicorn
  worker pre-builds the `/v1/models` response at startup, so the first request is
  served from cache instead of triggering a full provider re-fetch; once the
  cached list expires it is served stale while a background thread refreshes it.
- Config is hot-reloaded on each request via a `(st_mtime_ns, st_size)` cache;
  provider changes take effect without a server restart.  Only `host` and `port`
  changes require one.
- Streaming responses are relayed as raw SSE byte streams via
  `stream_with_context`, preserving upstream chunk boundaries.

---

## Acknowledgements

Parts of llmproxy's routing draw on prior work. Full attribution, licence texts,
and a statement of what was changed live in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

- **[NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)**
  (Apache-2.0) — `llmproxy/signals.py` adapts Switchyard's tool-signal
  extraction and stage scoring, including its severity tiering and the
  calibrated constants that make one signal insufficient to switch tiers on its
  own. The practice of stamping every routing decision with its source, and of
  reporting the scorer's *inputs* rather than only its verdict, comes from the
  same project.
- **[diegosouzapw/OmniRoute](https://github.com/diegosouzapw/OmniRoute)** (MIT) —
  two ideas, implemented here independently: excluding client aborts and the
  proxy's own stream-lifecycle errors from provider failure accounting, and
  applying rendezvous hashing to prompt-cache affinity (with the correctness
  note that a bare first turn has no reusable prefix to key on).
- **Rendezvous (highest-random-weight) hashing** — Thaler & Ravishankar, 1996.
- **The power of two random choices** — Mitzenmacher, 1996.

---

## Licence

llmproxy is licensed under the **Apache License, Version 2.0**. See
[`LICENSE`](LICENSE) for the full text.
