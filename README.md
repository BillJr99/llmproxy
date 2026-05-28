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
│   ├── setup_wizard.py
│   ├── free_models.py       ← loader for the JSON sidecar
│   └── free_models.json     ← single source of truth for provider templates
│                                + believed_free / model_reasoning / free_limits
├── scripts/
│   └── update_free_models.py ← scraper that keeps free_models.json current
│       └── sources/         ← per-source plugins (openrouter, community, /models, docs)
├── tests/                   ← pytest unit/integration suite
├── requirements.txt
├── requirements-dev.txt     ← pytest, ruff, responses (test-only deps)
├── pyproject.toml           ← pytest + ruff config
├── setup.py                 ← only needed for pip install
├── Dockerfile
├── docker-compose.yml
├── config.example.json      ← auto-generated from llmproxy/free_models.json
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

---

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
      "model_filter": ["model-a", "model-b"]
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
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "log_level": "INFO",
    "request_timeout": 120,
    "stream_timeout": 300,
    "response_cache_ttl": 120
  }
}
```

`model_filter` is an optional list of upstream model IDs to allow (without the
provider prefix).  It is not set by default in `config.example.json`.  Set it to
`null` or omit it to permit all models from that provider.  It can be used as a
manual allowlist, or as a fallback model list for providers whose `/v1/models`
endpoint does not work (e.g. Cloudflare Workers AI, Cloudflare AI Gateway).

`believed_free` is an **optional** top-level array of model names that the
`free` virtual model should include even when their ID doesn't contain the
word `free`.  Omit the field entirely (or set it to `[]`) to keep the
default behaviour — only IDs that literally contain `free` are pulled in.
Each entry is matched (case-insensitively) against either the upstream
model ID (e.g. `gpt-oss-20b`) or the full proxy ID (e.g.
`openrouter/qwen/qwen3-coder:free`).  The setup wizard manages this field
via its "Manage model tags" menu and via the per-provider auto-populate step
when you add a templated provider; the merged defaults come from
[`llmproxy/free_models.json`](llmproxy/free_models.json).

> **Free-tier accuracy:** The `believed_free` entries in `config.example.json` and in
> `llmproxy/free_models.json` are best-effort estimates based on publicly-stated provider
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
"Manage model tags" menu (merged defaults come from `llmproxy/free_models.json`).

See `config.example.json` for a complete annotated example.

### Provider templates

Provider templates and free-tier metadata both live in
[`llmproxy/free_models.json`](llmproxy/free_models.json) — the single source of
truth. The setup wizard reads from this file at startup; `config.example.json`
is regenerated from the same file. To add or update a provider, edit
`free_models.json` directly (or run the scraper — see
[Keeping the free-models list current](#keeping-the-free-models-list-current)).

The wizard currently offers ready-made templates for these providers:

| Provider                          | Default key      | Base URL                                                   |
|-----------------------------------|------------------|-----------------------------------------------------------|
| Nous Research (Hermes)            | `nous`           | `https://inference-api.nousresearch.com/v1`                |
| Nvidia NIM                        | `nvidia`         | `https://integrate.api.nvidia.com/v1`                      |
| Google Gemini (OpenAI-compat)     | `google`         | `https://generativelanguage.googleapis.com/v1beta/openai`  |
| Cerebras                          | `cerebras`       | `https://api.cerebras.ai/v1`                               |
| GitHub Models                     | `github`         | `https://models.inference.ai.azure.com`                    |
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

> **Providers that do not support `GET /v1/models` (as of May 2026)**
> Some providers return an error or non-JSON response for the `/models` endpoint.
> For these, llmproxy synthesizes model entries from the provider's `model_filter`
> when the fetch fails.  Set `model_filter` manually in your config for these
> providers to enumerate the models you want available.
>
> | Provider | Reason |
> |---|---|
> | **Cloudflare Workers AI** | Returns HTTP 405 — method not supported |
> | **Cloudflare AI Gateway** | Returns HTTP 401 — no anonymous model enumeration |
> | **Hugging Face Inference** | Returns HTML rather than JSON for `/v1/models` |

---

## Keeping the free-models list current

Provider free tiers change without notice. The
[`llmproxy/free_models.json`](llmproxy/free_models.json) sidecar holds the
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
| `community`  | low        | Pulls the [tashfeenahmed/freellmapi](https://github.com/tashfeenahmed/freellmapi) community list as a sanity signal. |

### Usage

```bash
# Preview proposed changes (no files written)
python scripts/update_free_models.py --dry-run

# Apply the changes to llmproxy/free_models.json and regenerate config.example.json
python scripts/update_free_models.py

# Restrict to one provider
python scripts/update_free_models.py --provider google --dry-run

# Restrict to specific sources
python scripts/update_free_models.py --source openrouter,docs --dry-run

# Just regenerate config.example.json from the current sidecar (no scraping)
python scripts/update_free_models.py --regen-config-only
```

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
  `llmproxy/free_models.json` (regenerate locally with
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
