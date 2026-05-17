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
├── llmproxy_test_client.py  ← test client (standalone, no install needed)
├── llmproxy/                ← the package
│   ├── __main__.py
│   ├── config.py
│   ├── server.py
│   └── setup_wizard.py
├── requirements.txt
├── setup.py                 ← only needed for pip install
├── Dockerfile
├── docker-compose.yml
└── config.example.json
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

### The `free` virtual model

llmproxy advertises a special synthetic model named `llmproxy/free`.  When a request
arrives with `"model": "llmproxy/free"`, the proxy:

1. Collects every model across all providers whose upstream ID contains the
   word `free` (case-insensitive) **or** whose upstream ID (or full
   `provider/upstream` ID) appears in the top-level `known_free` config list
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
  -d '{"model": "llmproxy/free", "messages": [{"role": "user", "content": "Hello!"}]}'

# Inspect which backends are currently eligible
curl http://localhost:8080/v1/models/llmproxy/free | jq '._candidates'
```

The `llmproxy/free` model appears at the top of `GET /v1/models` whenever at least one
eligible backend is available.

### The `local` virtual model

llmproxy also advertises a synthetic model named `llmproxy/local`.  When a request
arrives with `"model": "llmproxy/local"`, the proxy:

1. Collects every model across all providers whose `base_url` hostname matches
   a loopback address (`localhost`, `127.x.x.x`, `::1`).
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
  -d '{"model": "llmproxy/local", "messages": [{"role": "user", "content": "Hello!"}]}'

# Inspect which backends are currently eligible
curl http://localhost:8080/v1/models/llmproxy/local | jq '._candidates'
```

The `llmproxy/local` model appears in `GET /v1/models` only when at least one model
from a localhost-backed provider is present in the route cache — meaning the
provider must be reachable and its `/models` listing must have been fetched
successfully.

### Reasoning-level virtual models

You can optionally tag individual models in the config with a **reasoning
level** — `exploratory`, `standard`, or `deep` — to group them by how much
thinking effort they are expected to apply.  When at least one model is tagged
with a given level, llmproxy exposes corresponding virtual endpoints:

| Virtual model name          | Selects                                                       |
|-----------------------------|---------------------------------------------------------------|
| `llmproxy/exploratory`      | All models tagged `exploratory`                               |
| `llmproxy/standard`         | All models tagged `standard`                                  |
| `llmproxy/deep`             | All models tagged `deep`                                      |
| `llmproxy/exploratory/free` | Models tagged `exploratory` **and** qualifying as free-tier   |
| `llmproxy/exploratory/local` | Models tagged `exploratory` **and** served on localhost       |
| `llmproxy/standard/free`    | Models tagged `standard` **and** qualifying as free-tier      |
| `llmproxy/standard/local`   | Models tagged `standard` **and** served on localhost          |
| `llmproxy/deep/free`        | Models tagged `deep` **and** qualifying as free-tier          |
| `llmproxy/deep/local`       | Models tagged `deep` **and** served on localhost              |

Each virtual endpoint uses the same random-start round-robin with automatic
failover as `llmproxy/free` and `llmproxy/local`.

```bash
# Use the deep reasoning virtual model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llmproxy/deep", "messages": [{"role": "user", "content": "Prove P≠NP"}]}'

# Inspect which backends are eligible for llmproxy/standard/free
curl http://localhost:8080/v1/models/llmproxy/standard%2Ffree | jq '._candidates'
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
  "known_free": [
    "openrouter/qwen/qwen3-coder:free",
    "gpt-oss-20b"
  ],
  "model_reasoning": {
    "anthropic/claude-3.5-haiku": "exploratory",
    "anthropic/claude-sonnet-4-5": "standard",
    "anthropic/claude-opus-4": "deep",
    "openrouter/deepseek/deepseek-r1": "deep"
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

`model_filter` is a list of upstream model IDs to allow (without the provider
prefix).  Set it to `null` or omit it to permit all models from that provider.

`known_free` is an **optional** top-level array of model names that the
`free` virtual model should include even when their ID doesn't contain the
word `free`.  Omit the field entirely (or set it to `[]`) to keep the
default behaviour — only IDs that literally contain `free` are pulled in.
Each entry is matched (case-insensitively) against either the upstream
model ID (e.g. `gpt-oss-20b`) or the full proxy ID (e.g.
`openrouter/qwen/qwen3-coder:free`).  The setup wizard does not edit this
field — add it by hand to the config file.

<a name="model_reasoning"></a>
`model_reasoning` is an **optional** top-level object that tags individual
models with a reasoning level.  Valid levels are `exploratory`, `standard`,
and `deep`.  Each key is matched (case-insensitively) against either the
upstream model ID (e.g. `anthropic/claude-opus-4`) or the full
`provider/upstream_model` proxy ID (e.g.
`openrouter/anthropic/claude-opus-4`).  When a level has at least one tagged
model in the route cache, the corresponding virtual endpoint is advertised in
`GET /v1/models`.  Omit the field entirely (or set it to `{}`) to disable
reasoning-level routing.  The setup wizard does not edit this field — add it
by hand to the config file.

See `config.example.json` for a complete annotated example.

### Provider templates

The interactive setup wizard (`--setup`) includes ready-made templates for the
following providers:

| Provider                          | Default key      | Base URL                                                   |
|-----------------------------------|------------------|------------------------------------------------------------|
| Nous Research (Hermes)            | `nous`           | `https://inference-api.nousresearch.com/v1`                |
| Nvidia NIM                        | `nvidia`         | `https://integrate.api.nvidia.com/v1`                      |
| Google Gemini (OpenAI-compat)     | `google`         | `https://generativelanguage.googleapis.com/v1beta/openai`  |
| Cerebras                          | `cerebras`       | `https://api.cerebras.ai/v1`                               |
| GitHub Models                     | `github`         | `https://models.inference.ai.azure.com`                    |
| SambaNova Cloud                   | `sambanova`      | `https://api.sambanova.ai/v1`                              |
| Mistral AI                        | `mistral`        | `https://api.mistral.ai/v1`                                |
| Groq                              | `groq`           | `https://api.groq.com/openai/v1`                           |
| Cloudflare Workers AI             | `cloudflare-workers` | `https://api.cloudflare.com/client/v4/accounts/.../ai/v1`  |
| Zhipu AI (Z.ai / BigModel)        | `zhipu`          | `https://open.bigmodel.cn/api/paas/v4`                     |
| Cohere                            | `cohere`         | `https://api.cohere.com/compatibility/v1`                  |
| DeepSeek                          | `deepseek`       | `https://api.deepseek.com/v1`                              |
| OpenRouter                        | `openrouter`     | `https://openrouter.ai/api/v1`                             |
| Ollama Cloud                      | `ollama-cloud`   | `https://ollama.com/v1`                                    |
| Moonshot AI (Kimi)                | `moonshot`       | `https://api.moonshot.cn/v1`                               |
| MiniMax                           | `minimax`        | `https://api.minimax.chat/v1`                              |
| Hugging Face Inference            | `huggingface`    | `https://router.huggingface.co/v1`                         |
| xAI (Grok)                        | `xai`            | `https://api.x.ai/v1`                                      |
| Cloudflare AI Gateway             | `cloudflare-ai-gateway` | `https://gateway.ai.cloudflare.com/v1/{account}/{gw}/workers-ai/v1` |

Any OpenAI-compatible provider can also be added manually via the "Add / edit a
provider (manual)" menu option.

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

## Test client

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
| `free`      | Sends several prompts to `model="llmproxy/free"`; tests cycling + streaming    | Yes (free tier)  |
| `local`     | Sends several prompts to `model="llmproxy/local"`; skipped if none configured  | Yes (localhost)  |
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

### First-time setup

The configuration lives in a named Docker volume (`llmproxy_config`) mounted at
`/root/.config/llmproxy` inside the container.  You never need to map host
filesystem paths into the container.

```bash
# Interactive setup wizard (creates/updates the config volume)
docker run -it --rm \
  -v llmproxy_config:/root/.config/llmproxy \
  llmproxy --setup
```

### Start the server

```bash
docker run -d \
  -p 8080:8080 \
  -v llmproxy_config:/root/.config/llmproxy \
  --name llmproxy \
  llmproxy
```

### Reconfigure without stopping the server

```bash
# Run setup in a temporary container sharing the same volume
docker run -it --rm \
  -v llmproxy_config:/root/.config/llmproxy \
  llmproxy --setup

# Restart only if host or port changed; otherwise hot-reload handles it
docker restart llmproxy
```

---

## docker-compose

```bash
# Build and start the server (detached)
docker-compose up -d

# First-time setup or reconfigure (interactive)
docker-compose run --rm setup

# Restart to apply host/port changes
docker-compose restart llmproxy

# View logs
docker-compose logs -f llmproxy

# Tear down containers (config volume is preserved)
docker-compose down

# Tear down everything including the config volume
docker-compose down -v
```

The `setup` service shares the `llmproxy_config` named volume with the server
service.  It is declared with `profiles: [setup]` so it is never started by a
plain `docker-compose up`.

---

## API endpoints

All endpoints mirror the OpenAI API.

| Method | Path                    | Description                               |
|--------|-------------------------|-------------------------------------------|
| GET    | `/health`               | Health check; returns provider list       |
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
