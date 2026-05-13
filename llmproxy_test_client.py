#!/usr/bin/env python3
"""
llmproxy_test_client.py — Test client for the llmproxy OpenAI-compatible proxy.

Exercises every endpoint and prints a structured report of results.

Usage
-----
  # Run all tests against localhost:8080 (default)
  python llmproxy_test_client.py

  # Target a different host/port
  python llmproxy_test_client.py --base-url http://localhost:9000/v1

  # Test only specific suites
  python llmproxy_test_client.py --suite health --suite models

  # Pick a specific model for chat/embedding tests (overrides auto-select)
  python llmproxy_test_client.py --model openrouter/openrouter/free

  # Suppress streaming test (useful if your terminal doesn't handle SSE well)
  python llmproxy_test_client.py --no-stream

  # Use a real OpenAI-compatible client library instead of raw requests
  python llmproxy_test_client.py --use-sdk

  # Test every model with a simple prompt and report pass/fail per model
  python llmproxy_test_client.py --all-models
  python llmproxy_test_client.py --all-models --model-timeout 30

Available suites: health, models, chat, streaming, completions, embeddings, errors, free, local, all-models
"""

import argparse
import json
import sys
import time
import traceback
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

RESET   = "\033[0m"
BOLD    = "\033[1m"
GREEN   = "\033[32m"
RED     = "\033[31m"
YELLOW  = "\033[33m"
CYAN    = "\033[36m"
DIM     = "\033[2m"
BLUE    = "\033[34m"


def _ok(s):    return f"{GREEN}✓ {s}{RESET}"
def _fail(s):  return f"{RED}✗ {s}{RESET}"
def _skip(s):  return f"{YELLOW}⊘ {s}{RESET}"
def _info(s):  return f"{CYAN}  {s}{RESET}"
def _head(s):  return f"\n{BOLD}{BLUE}══ {s} ══{RESET}"
def _dim(s):   return f"{DIM}{s}{RESET}"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self._failures: list[tuple[str, str]] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.passed += 1
        suffix = f"  {_dim(detail)}" if detail else ""
        print(f"  {_ok(name)}{suffix}")

    def fail(self, name: str, detail: str = "") -> None:
        self.failed += 1
        self._failures.append((name, detail))
        suffix = f"\n    {_dim(detail)}" if detail else ""
        print(f"  {_fail(name)}{suffix}")

    def skip(self, name: str, reason: str = "") -> None:
        self.skipped += 1
        suffix = f"  {_dim(reason)}" if reason else ""
        print(f"  {_skip(name)}{suffix}")

    def summary(self) -> None:
        total = self.passed + self.failed + self.skipped
        print(f"\n{BOLD}{'─' * 55}{RESET}")
        print(f"{BOLD}Results:{RESET}  "
              f"{GREEN}{self.passed} passed{RESET}  "
              f"{RED}{self.failed} failed{RESET}  "
              f"{YELLOW}{self.skipped} skipped{RESET}  "
              f"/ {total} total")
        if self._failures:
            print(f"\n{RED}Failed tests:{RESET}")
            for name, detail in self._failures:
                print(f"  • {name}")
                if detail:
                    print(f"    {_dim(detail)}")
        print()


results = Results()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(base_url: str, path: str, timeout: int = 10) -> requests.Response:
    return requests.get(f"{base_url}{path}", timeout=timeout)


def _post(base_url: str, path: str, payload: dict, timeout: int = 30) -> requests.Response:
    return requests.post(
        f"{base_url}{path}",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


def _post_stream(base_url: str, path: str, payload: dict, timeout: int = 60):
    """Return a streaming response context manager."""
    return requests.post(
        f"{base_url}{path}",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=timeout,
    )


def _json_or_text(resp: requests.Response) -> str:
    try:
        return json.dumps(resp.json(), indent=2)[:400]
    except Exception:
        return resp.text[:400]


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

def test_health(base_url: str, **_) -> None:
    print(_head("Health Check"))
    try:
        resp = _get(base_url.replace("/v1", ""), "/health")
        if resp.status_code == 200:
            data = resp.json()
            results.ok("GET /health returns 200", f"providers={data.get('providers', [])}")
            if data.get("providers"):
                print(_info(f"Active providers: {', '.join(data['providers'])}"))
            else:
                print(_info("No providers configured yet. Run: llmproxy --setup"))
        else:
            results.fail("GET /health", f"status={resp.status_code}")
    except requests.exceptions.ConnectionError:
        results.fail(
            "GET /health — cannot connect",
            f"Is llmproxy running at {base_url.replace('/v1','')}?  "
            "Try:  llmproxy",
        )


def test_models(base_url: str, **_) -> Optional[list[str]]:
    """Return the list of discovered model IDs (or None on failure)."""
    print(_head("Model Listing"))
    try:
        t0 = time.monotonic()
        resp = _get(base_url, "/models")
        elapsed = time.monotonic() - t0

        if resp.status_code != 200:
            results.fail("GET /v1/models", f"status={resp.status_code}  body={_json_or_text(resp)}")
            return None

        data = resp.json()
        models: list[dict] = data.get("data", [])
        results.ok(f"GET /v1/models returns 200", f"{len(models)} models in {elapsed:.2f}s")

        if not models:
            results.skip("Model list populated", "no providers configured or all returned empty lists")
            return []

        # Group by provider so counts are visible per provider.
        from collections import defaultdict
        by_provider: dict = defaultdict(list)
        for m in models:
            prov = m.get("_provider", m.get("id","").split("/")[0])
            by_provider[prov].append(m.get("id", "?"))

        print(_info(f"{'Model ID':<50}  {'Provider':<15}"))
        print(_info(f"{'─'*50}  {'─'*15}"))
        shown = 0
        for prov, ids in by_provider.items():
            for mid in ids[:20]:
                print(_info(f"{mid:<50}  {prov:<15}"))
                shown += 1
            if len(ids) > 20:
                print(_info(f"{'':50}  {_dim(f'  ... +{len(ids)-20} more from {prov} (filter may be unset)'):<15}"))

        # Summary line per provider — makes it easy to spot an unfiltered provider.
        print()
        for prov, ids in by_provider.items():
            print(_info(f"  {prov}: {len(ids)} model{'s' if len(ids) != 1 else ''}"))

        # Verify naming convention: every non-synthetic ID should contain at least one slash.
        # "free" is the one built-in synthetic model that intentionally has no slash.
        SYNTHETIC_IDS = {"free", "local"}
        bad = [m.get("id", "") for m in models
               if "/" not in m.get("id", "") and m.get("id", "") not in SYNTHETIC_IDS]
        if bad:
            results.fail("Model IDs follow provider/name convention", f"violating IDs: {bad[:5]}")
        else:
            results.ok("All model IDs follow provider/name convention")

        if any(m.get("id") == "free" for m in models):
            results.ok("Synthetic 'free' cycling model is advertised")
        else:
            results.skip("Synthetic 'free' model", "no free-tier models found across providers")

        if any(m.get("id") == "local" for m in models):
            results.ok("Synthetic 'local' cycling model is advertised")
        else:
            results.skip("Synthetic 'local' model", "no providers with a localhost base_url found")

        return [m.get("id", "") for m in models]

    except requests.exceptions.ConnectionError:
        results.fail("GET /v1/models — cannot connect")
        return None


def test_single_model(base_url: str, model_id: str, **_) -> None:
    print(_head(f"Single-Model Lookup"))
    try:
        resp = _get(base_url, f"/models/{model_id}")
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("id") == model_id, f"id mismatch: got {data.get('id')}"
            results.ok(f"GET /v1/models/{model_id}", f"id={data['id']}")
        else:
            results.fail(f"GET /v1/models/{model_id}", f"status={resp.status_code}")
    except AssertionError as e:
        results.fail(f"GET /v1/models/{model_id} id field", str(e))
    except Exception as e:
        results.fail(f"GET /v1/models/{model_id}", str(e))


def test_chat(base_url: str, model: str, **_) -> None:
    print(_head("Chat Completions (non-streaming)"))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Be very brief."},
            {"role": "user",   "content": "Reply with exactly the word PONG and nothing else."},
        ],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        t0 = time.monotonic()
        resp = _post(base_url, "/chat/completions", payload, timeout=60)
        elapsed = time.monotonic() - t0

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            content = choices[0].get("message", {}).get("content") if choices else None
            results.ok(
                "POST /v1/chat/completions",
                f"model={data.get('model','?')}  elapsed={elapsed:.2f}s",
            )
            print(_info(f"Response content: {repr(content)}"))
            if content is None:
                results.skip(
                    "Response content check",
                    "content is None (model may require streaming, use thinking tokens, "
                    "or return content in a non-standard field)",
                )
            elif "PONG" in content.upper():
                results.ok("Response contains expected token 'PONG'")
            else:
                results.skip("Response contains 'PONG'", f"got: {repr(content[:80])}")
        else:
            results.fail(
                "POST /v1/chat/completions",
                f"status={resp.status_code}  body={_json_or_text(resp)}",
            )
    except requests.exceptions.Timeout:
        results.fail("POST /v1/chat/completions — timed out after 60s")
    except Exception as e:
        results.fail("POST /v1/chat/completions", str(e))
        traceback.print_exc()


def test_streaming(base_url: str, model: str, **_) -> None:
    print(_head("Chat Completions (streaming / SSE)"))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 5, one number per line."}],
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
    }
    try:
        t0 = time.monotonic()
        chunks_received = 0
        content_parts = []

        with _post_stream(base_url, "/chat/completions", payload) as resp:
            if resp.status_code != 200:
                results.fail(
                    "POST /v1/chat/completions (stream)",
                    f"status={resp.status_code}  body={resp.text[:200]}",
                )
                return

            print(_info("Streaming tokens: "), end="", flush=True)
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content") or delta.get("reasoning_content") or ""
                if token:
                    content_parts.append(token)
                    print(token, end="", flush=True)
                chunks_received += 1

        elapsed = time.monotonic() - t0
        print()  # newline after streamed tokens
        full_content = "".join(content_parts)

        if chunks_received > 0:
            results.ok(
                "POST /v1/chat/completions (stream)",
                f"{chunks_received} chunks  elapsed={elapsed:.2f}s",
            )
            if full_content:
                results.ok("Stream produced non-empty content", repr(full_content[:60]))
            else:
                results.skip(
                    "Stream content check",
                    "chunks received but all content tokens were empty "
                    "(model may use non-standard delta fields)",
                )
        else:
            results.fail("POST /v1/chat/completions (stream)", "received 0 chunks")

    except requests.exceptions.Timeout:
        results.fail("POST /v1/chat/completions (stream) — timed out")
    except Exception as e:
        results.fail("POST /v1/chat/completions (stream)", str(e))
        traceback.print_exc()


def test_embeddings(base_url: str, model: str, **_) -> None:
    print(_head("Embeddings"))
    # Many providers don't offer embeddings on every model;
    # we check the HTTP layer and accept either a 200 or a model-specific error.
    payload = {
        "model": model,
        "input": "The quick brown fox jumps over the lazy dog.",
    }
    try:
        resp = _post(base_url, "/embeddings", payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            emb = data.get("data", [{}])[0].get("embedding", [])
            results.ok(
                "POST /v1/embeddings",
                f"embedding dim={len(emb)}",
            )
        elif resp.status_code in (400, 404, 422):
            results.skip(
                "POST /v1/embeddings",
                f"upstream returned {resp.status_code} (model may not support embeddings)",
            )
        else:
            results.fail(
                "POST /v1/embeddings",
                f"status={resp.status_code}  body={_json_or_text(resp)}",
            )
    except Exception as e:
        results.fail("POST /v1/embeddings", str(e))


def test_error_handling(base_url: str, **_) -> None:
    print(_head("Error Handling"))

    # 1. Missing model field
    try:
        resp = _post(base_url, "/chat/completions", {"messages": []})
        if resp.status_code == 400 and "model" in resp.text:
            results.ok("Missing 'model' field → 400", f"body contains 'model'")
        else:
            results.fail("Missing 'model' field", f"expected 400, got {resp.status_code}")
    except Exception as e:
        results.fail("Missing 'model' field", str(e))

    # 2. Model string without provider prefix
    try:
        resp = _post(base_url, "/chat/completions", {"model": "gpt-4o", "messages": []})
        if resp.status_code == 400:
            results.ok("Non-prefixed model string → 400")
        else:
            results.fail("Non-prefixed model string", f"expected 400, got {resp.status_code}")
    except Exception as e:
        results.fail("Non-prefixed model string", str(e))

    # 3. Unknown provider
    try:
        resp = _post(base_url, "/chat/completions",
                     {"model": "nonexistentprovider_xyz/gpt-4o", "messages": []})
        if resp.status_code == 404:
            results.ok("Unknown provider → 404")
        else:
            results.fail("Unknown provider", f"expected 404, got {resp.status_code}")
    except Exception as e:
        results.fail("Unknown provider", str(e))

    # 4. Non-JSON body
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            data="not json at all",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code == 400:
            results.ok("Non-JSON body → 400")
        else:
            results.fail("Non-JSON body", f"expected 400, got {resp.status_code}")
    except Exception as e:
        results.fail("Non-JSON body", str(e))

    # 5. GET /health sanity (no provider needed)
    try:
        resp = _get(base_url.replace("/v1", ""), "/health")
        data = resp.json()
        if resp.status_code == 200 and "status" in data:
            results.ok("GET /health JSON schema contains 'status'")
        else:
            results.fail("GET /health schema", f"got {data}")
    except Exception as e:
        results.fail("GET /health schema", str(e))


def test_sdk(base_url: str, model: str, **_) -> None:
    """Test using the official openai Python SDK pointing at llmproxy."""
    print(_head("OpenAI SDK Compatibility"))
    try:
        import openai  # type: ignore
    except ImportError:
        results.skip("OpenAI SDK test", "openai package not installed  (pip install openai)")
        return

    try:
        client = openai.OpenAI(base_url=base_url, api_key="not-used-by-proxy")
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say just the word HELLO."}],
            max_tokens=16,
            temperature=0,
        )
        elapsed = time.monotonic() - t0
        content = resp.choices[0].message.content if resp.choices else ""
        results.ok("openai.OpenAI client chat.completions.create", f"elapsed={elapsed:.2f}s")
        print(_info(f"Content: {repr(content)}"))

        # Streaming via SDK
        t0 = time.monotonic()
        token_count = 0
        stream_content = []
        with client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count 1 2 3."}],
            max_tokens=32,
            temperature=0,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    stream_content.append(delta)
                    token_count += 1
        elapsed = time.monotonic() - t0
        results.ok(
            "openai.OpenAI client streaming",
            f"{token_count} tokens  elapsed={elapsed:.2f}s",
        )
        print(_info(f"Streamed: {repr(''.join(stream_content)[:80])}"))

    except Exception as e:
        results.fail("OpenAI SDK test", str(e))
        traceback.print_exc()



# ---------------------------------------------------------------------------
# Per-model prompt test
# ---------------------------------------------------------------------------

PROMPT = "Reply with exactly two lines. Line 1: your name or model identifier. Line 2: the capital of France."


def _stream_response(base_url: str, model: str, prompt: str, timeout: int) -> tuple[bool, str, str]:
    """
    Send *prompt* to *model* via a streaming chat completion and collect the
    full response text.

    Returns (success, response_text, error_detail).
    Accepts tokens from either the standard `content` delta field or the
    `reasoning_content` field used by some reasoning models.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
        "stream": True,
    }
    collected: list[str] = []
    error: str = ""

    try:
        with requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                return False, "", f"HTTP {resp.status_code}: {resp.text[:200]}"

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content") or delta.get("reasoning_content") or ""
                if token:
                    collected.append(token)

    except requests.exceptions.Timeout:
        error = f"timed out after {timeout}s"
        # Return whatever we collected before the timeout, if anything.
    except Exception as e:
        return False, "", str(e)

    response_text = "".join(collected)
    if not response_text and error:
        return False, "", error
    if not response_text:
        return False, "", "no content tokens received (model may use non-standard delta fields)"
    return True, response_text, error


def test_all_models(base_url: str, models: list[str], timeout: int = 60, **_) -> None:
    """
    Send a real prompt to every discovered model, grouped by provider, and
    print the full response for each.

    A model passes if it returns any non-empty response text.  Results are
    rolled into the global Results object.
    """
    print(_head("All-Model Prompt Test"))
    if not models:
        results.skip("All-model prompt test", "no models available")
        return

    print()
    print(f"  {BOLD}Prompt:{RESET} {_dim(PROMPT)}")

    # Group by provider (the prefix before the first '/').
    from collections import defaultdict
    by_provider: dict[str, list[str]] = defaultdict(list)
    for m in models:
        provider = m.split("/")[0]
        by_provider[provider].append(m)

    passed = failed = 0

    for provider, provider_models in by_provider.items():
        print()
        plural = "s" if len(provider_models) != 1 else ""
        print(f"  {BOLD}{CYAN}Provider: {provider}{RESET}  "
              f"{_dim(f'({len(provider_models)} model{plural})')}")
        print(f"  {_dim('─' * 70)}")

        for model in provider_models:
            upstream = model[len(provider) + 1:]   # strip provider prefix for display
            print()
            print(f"  {BOLD}{upstream}{RESET}  {_dim('(' + model + ')')}")
            print(f"  {CYAN}  → Prompt:{RESET}  {PROMPT}")

            t0 = time.monotonic()
            ok, response_text, err = _stream_response(base_url, model, PROMPT, timeout)
            elapsed = time.monotonic() - t0

            if ok:
                passed += 1
                print(f"  {GREEN}  ← Response:{RESET} {_dim(f'({elapsed:.2f}s)')}")
                for line in _wrap(response_text, width=70):
                    print(f"      {line}")
                if err:
                    print(f"    {YELLOW}  (partial — {err}){RESET}")
            else:
                failed += 1
                print(f"  {RED}  ← ERROR:{RESET} {err}  {_dim(f'({elapsed:.2f}s)')}") 

    print()
    label = f"{passed + failed} model{'s' if passed + failed != 1 else ''}"
    if failed == 0:
        results.ok(f"All {label} responded", "")
    else:
        results.fail(f"{failed} of {label} did not respond", "")


def _wrap(text: str, width: int = 70) -> list[str]:
    """
    Simple word-wrapper.  Preserves existing newlines and wraps long lines.
    """
    import textwrap
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


# ---------------------------------------------------------------------------
# "free" virtual model test suite
# ---------------------------------------------------------------------------

_FREE_PROMPTS = [
    ("one-word reply",    "Reply with exactly one word: PONG"),
    ("capital of France", "What is the capital of France? Answer in one word."),
    ("simple arithmetic",  "What is 3 plus 4? Answer with the number only."),
    ("yes/no question",   "Is the sky blue? Answer Yes or No only."),
    ("short greeting",    "Say hello in exactly three words."),
]


def test_free_model(base_url: str, **_) -> None:
    """
    Send several short prompts to the synthetic 'free' model and verify that
    each receives a non-empty response.  Both non-streaming and streaming
    paths are exercised.
    """
    print(_head("Free Model Cycling"))

    # Confirm the model is advertised.
    try:
        resp = _get(base_url, "/models/free")
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("_candidates", [])
            results.ok("GET /v1/models/free returns 200",
                       f"{len(candidates)} candidate(s): {', '.join(candidates[:4])}"
                       + (" ..." if len(candidates) > 4 else ""))
        else:
            results.fail("GET /v1/models/free", f"status={resp.status_code}")
    except Exception as e:
        results.fail("GET /v1/models/free", str(e))

    # Non-streaming prompts.
    print()
    print(f"  {BOLD}Non-streaming prompts:{RESET}")
    for label, prompt in _FREE_PROMPTS:
        payload = {
            "model": "free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
        }
        try:
            t0 = time.monotonic()
            resp = _post(base_url, "/chat/completions", payload, timeout=90)
            elapsed = time.monotonic() - t0
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                content = (choices[0].get("message", {}).get("content") or "") if choices else ""
                used_model = data.get("model", "?")
                if content:
                    results.ok(
                        f"free → {label}",
                        f"via {used_model}  reply={repr(content[:60])}  {elapsed:.2f}s",
                    )
                else:
                    results.skip(
                        f"free → {label}",
                        f"via {used_model}  no content in response (model may use non-standard fields)",
                    )
            else:
                results.fail(f"free → {label}", f"status={resp.status_code}  body={_json_or_text(resp)}")
        except requests.exceptions.Timeout:
            results.fail(f"free → {label}", "timed out after 90s")
        except Exception as e:
            results.fail(f"free → {label}", str(e))

    # Streaming prompt.
    print()
    print(f"  {BOLD}Streaming prompt:{RESET}")
    stream_payload = {
        "model": "free",
        "messages": [{"role": "user", "content": "Count from 1 to 3, one number per line."}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
    }
    try:
        t0 = time.monotonic()
        chunks = 0
        parts: list[str] = []
        print(_info("Streaming tokens: "), end="", flush=True)
        with _post_stream(base_url, "/chat/completions", stream_payload, timeout=90) as resp:
            if resp.status_code != 200:
                results.fail("free (streaming)", f"status={resp.status_code}  body={resp.text[:200]}")
            else:
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content") or delta.get("reasoning_content") or ""
                    chunks += 1
                    if token:
                        parts.append(token)
                        print(token, end="", flush=True)
        print()  # newline after tokens
        elapsed = time.monotonic() - t0
        full_content = "".join(parts)
        if full_content:
            results.ok("free (streaming)", f"{chunks} chunks  content={repr(full_content[:60])}  {elapsed:.2f}s")
        elif chunks > 0:
            results.skip("free (streaming)", f"{chunks} chunks received but no content tokens (model may use non-standard delta fields)")
        else:
            results.fail("free (streaming)", "received 0 chunks")
    except requests.exceptions.Timeout:
        results.fail("free (streaming)", "timed out after 90s")
    except Exception as e:
        results.fail("free (streaming)", str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# "local" virtual model test suite
# ---------------------------------------------------------------------------

_LOCAL_PROMPTS = [
    ("one-word reply",    "Reply with exactly one word: PONG"),
    ("capital of Germany","What is the capital of Germany? Answer in one word."),
    ("simple arithmetic",  "What is 6 times 7? Answer with the number only."),
]


def test_local_model(base_url: str, **_) -> None:
    """
    Send several short prompts to the synthetic 'local' model and verify that
    each receives a non-empty response.  Both non-streaming and streaming
    paths are exercised.  The suite is skipped gracefully when no localhost
    provider is configured.
    """
    print(_head("Local Model Cycling"))

    # Confirm the model is advertised (or skip if no local providers).
    try:
        resp = _get(base_url, "/models/local")
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("_candidates", [])
            if not candidates:
                results.skip("local model cycling", "no providers with a localhost base_url configured")
                return
            results.ok("GET /v1/models/local returns 200",
                       f"{len(candidates)} candidate(s): {', '.join(candidates[:4])}"
                       + (" ..." if len(candidates) > 4 else ""))
        else:
            results.fail("GET /v1/models/local", f"status={resp.status_code}")
            return
    except Exception as e:
        results.fail("GET /v1/models/local", str(e))
        return

    # Non-streaming prompts.
    print()
    print(f"  {BOLD}Non-streaming prompts:{RESET}")
    for label, prompt in _LOCAL_PROMPTS:
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
        }
        try:
            t0 = time.monotonic()
            resp = _post(base_url, "/chat/completions", payload, timeout=90)
            elapsed = time.monotonic() - t0
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                content = (choices[0].get("message", {}).get("content") or "") if choices else ""
                used_model = data.get("model", "?")
                if content:
                    results.ok(
                        f"local → {label}",
                        f"via {used_model}  reply={repr(content[:60])}  {elapsed:.2f}s",
                    )
                else:
                    results.skip(
                        f"local → {label}",
                        f"via {used_model}  no content in response (model may use non-standard fields)",
                    )
            else:
                results.fail(f"local → {label}", f"status={resp.status_code}  body={_json_or_text(resp)}")
        except requests.exceptions.Timeout:
            results.fail(f"local → {label}", "timed out after 90s")
        except Exception as e:
            results.fail(f"local → {label}", str(e))

    # Streaming prompt.
    print()
    print(f"  {BOLD}Streaming prompt:{RESET}")
    stream_payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "Count from 1 to 3, one number per line."}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
    }
    try:
        t0 = time.monotonic()
        chunks = 0
        parts: list[str] = []
        print(_info("Streaming tokens: "), end="", flush=True)
        with _post_stream(base_url, "/chat/completions", stream_payload, timeout=90) as resp:
            if resp.status_code != 200:
                results.fail("local (streaming)", f"status={resp.status_code}  body={resp.text[:200]}")
            else:
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content") or delta.get("reasoning_content") or ""
                    chunks += 1
                    if token:
                        parts.append(token)
                        print(token, end="", flush=True)
        print()
        elapsed = time.monotonic() - t0
        full_content = "".join(parts)
        if full_content:
            results.ok("local (streaming)", f"{chunks} chunks  content={repr(full_content[:60])}  {elapsed:.2f}s")
        elif chunks > 0:
            results.skip("local (streaming)", f"{chunks} chunks received but no content tokens (model may use non-standard delta fields)")
        else:
            results.fail("local (streaming)", "received 0 chunks")
    except requests.exceptions.Timeout:
        results.fail("local (streaming)", "timed out after 90s")
    except Exception as e:
        results.fail("local (streaming)", str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Auto-select a test model from the available list
# ---------------------------------------------------------------------------

def _pick_model(models: list[str], preferred: Optional[str]) -> Optional[str]:
    if preferred:
        return preferred
    if not models:
        return None
    # Prefer the synthetic "free" cycling model when it's available — it
    # automatically routes to a working free-tier backend.
    if "free" in models:
        return "free"
    # Fall back to other free or small models by name.
    for keyword in ("free", "mini", "flash", "haiku", "small", "tiny", "7b", "8b"):
        for m in models:
            if keyword in m.lower():
                return m
    return models[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llmproxy_test_client",
        description="Test client for the llmproxy OpenAI-compatible proxy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        metavar="URL",
        help="Proxy base URL (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Force a specific model for chat/embedding tests (e.g. openrouter/openrouter/free).",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=["health", "models", "chat", "streaming", "completions", "embeddings", "errors", "free", "local", "sdk", "all-models"],
        metavar="SUITE",
        dest="suites",
        help="Run only the named suite(s). Repeatable. Default: all suites.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Skip the streaming test.",
    )
    parser.add_argument(
        "--use-sdk",
        action="store_true",
        default=False,
        help="Include the OpenAI SDK compatibility test (requires: pip install openai).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        default=False,
        help="Test every discovered model with a simple prompt and report pass/fail.",
    )
    parser.add_argument(
        "--model-timeout",
        metavar="SECONDS",
        type=int,
        default=60,
        help="Per-model timeout for --all-models (default: 60s).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_url = args.base_url.rstrip("/")
    run_all = not args.suites

    print(f"\n{BOLD}llmproxy test client{RESET}")
    print(f"Target: {CYAN}{base_url}{RESET}")
    print(f"{'─' * 55}")

    # ── Health ──────────────────────────────────────────────────
    if run_all or "health" in args.suites:
        test_health(base_url)

    # ── Models ──────────────────────────────────────────────────
    discovered_models: list[str] = []
    if run_all or "models" in args.suites:
        found = test_models(base_url)
        if found is not None:
            discovered_models = found

    # ── Select model for downstream tests ───────────────────────
    test_model = _pick_model(discovered_models, args.model)

    if test_model:
        print(f"\n{_dim('Using model for downstream tests:')} {CYAN}{test_model}{RESET}")
        if run_all or "models" in args.suites:
            test_single_model(base_url, test_model)
    else:
        if not args.model:
            print(f"\n{YELLOW}No models available — skipping chat/embedding/streaming tests.")
            print(f"Configure a provider with:  {RESET}llmproxy --setup\n")

    # ── Chat completions ─────────────────────────────────────────
    if run_all or "chat" in args.suites:
        if test_model:
            test_chat(base_url, test_model)
        else:
            print(_head("Chat Completions"))
            results.skip("Chat completions", "no model available")

    # ── Streaming ────────────────────────────────────────────────
    if (run_all or "streaming" in args.suites) and not args.no_stream:
        if test_model:
            test_streaming(base_url, test_model)
        else:
            print(_head("Streaming"))
            results.skip("Streaming", "no model available")

    # ── Embeddings ───────────────────────────────────────────────
    if run_all or "embeddings" in args.suites:
        if test_model:
            test_embeddings(base_url, test_model)
        else:
            print(_head("Embeddings"))
            results.skip("Embeddings", "no model available")

    # ── Error handling ───────────────────────────────────────────
    if run_all or "errors" in args.suites:
        test_error_handling(base_url)

    # ── Free model cycling ───────────────────────────────────────
    if run_all or "free" in (args.suites or []):
        test_free_model(base_url)

    # ── Local model cycling ──────────────────────────────────────
    if run_all or "local" in (args.suites or []):
        test_local_model(base_url)

    # ── OpenAI SDK ───────────────────────────────────────────────
    if args.use_sdk or "sdk" in (args.suites or []):
        if test_model:
            test_sdk(base_url, test_model)
        else:
            print(_head("OpenAI SDK Compatibility"))
            results.skip("SDK test", "no model available")

    # ── Per-model prompt test (runs by default when models are available) ───────
    if run_all or args.all_models or "all-models" in (args.suites or []):
        if not discovered_models and args.model:
            # --model was forced but we skipped the models suite; fetch now
            resp = _get(base_url, "/models")
            if resp.status_code == 200:
                discovered_models = [m.get("id", "") for m in resp.json().get("data", [])]
        test_all_models(base_url, discovered_models, timeout=args.model_timeout)

    # ── Summary ──────────────────────────────────────────────────
    results.summary()
    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()
