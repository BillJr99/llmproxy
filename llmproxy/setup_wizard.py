"""setup_wizard.py — Interactive terminal wizard for configuring llmproxy.

Run via:  llmproxy --setup
or:       python -m llmproxy --setup

In Docker:  docker run -it --rm -v llmproxy_config:/root/.config/llmproxy llmproxy --setup
"""

import getpass
import json
import re
import sys
import traceback
from typing import Optional

import requests as _requests

from .config import (
    DEFAULT_SERVER_CONFIG,
    RESERVED_PROVIDER_NAMES,
    get_config_path,
    load_config,
    save_config,
)

# ---------------------------------------------------------------------------
# Provider templates
# ---------------------------------------------------------------------------

# Each entry is a dict with keys:
#   display           – human-readable label shown in the menu
#   key               – default provider name / model-ID prefix
#   base_url          – upstream base URL; may contain "{account_id}" or
#                       "{gateway_id}" as placeholders
#   account_id_required – (optional) True when the URL contains "{account_id}"
#   account_id_label  – (optional) prompt label for the account ID (default "Account ID")
#   account_id_hint   – (optional) one-line hint for finding the account ID
#   gateway_id_required – (optional) True when the URL contains "{gateway_id}"
#   gateway_id_label  – (optional) prompt label for the gateway ID (default "Gateway ID")
#   gateway_id_hint   – (optional) one-line hint for finding the gateway ID
#   key_required      – (optional) True when an API key is mandatory
#   key_hint          – (optional) one-line hint for obtaining the API key
PROVIDER_TEMPLATES: list[dict] = [
    {
        "display": "Nous Research (Hermes)",
        "key": "nous",
        "base_url": "https://inference-api.nousresearch.com/v1",
        "key_required": True,
        "key_hint": "Get your API key at nousresearch.com",
    },
    {
        "display": "Nvidia NIM",
        "key": "nvidia",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_required": True,
        "key_hint": "Get your API key at build.nvidia.com",
    },
    {
        "display": "Google Gemini (via OpenAI-compat endpoint)",
        "key": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_required": True,
        "key_hint": "Get your API key at aistudio.google.com/apikey",
    },
    {
        "display": "Cerebras",
        "key": "cerebras",
        "base_url": "https://api.cerebras.ai/v1",
        "key_required": True,
        "key_hint": "Get your API key at cloud.cerebras.ai",
    },
    {
        "display": "GitHub Models",
        "key": "github",
        "base_url": "https://models.inference.ai.azure.com",
        "key_required": True,
        "key_hint": "Get your token at github.com/settings/tokens",
    },
    {
        "display": "SambaNova Cloud",
        "key": "sambanova",
        "base_url": "https://api.sambanova.ai/v1",
        "key_required": True,
        "key_hint": "Get your API key at cloud.sambanova.ai",
    },
    {
        "display": "Mistral AI",
        "key": "mistral",
        "base_url": "https://api.mistral.ai/v1",
        "key_required": True,
        "key_hint": "Get your API key at console.mistral.ai/api-keys",
    },
    {
        "display": "Groq",
        "key": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_required": True,
        "key_hint": "Get your API key at console.groq.com/keys",
    },
    {
        "display": "Cloudflare Workers AI",
        "key": "cloudflare-workers",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "account_id_required": True,
        "account_id_label": "Cloudflare Account ID",
        "account_id_hint": "Find your account ID at dash.cloudflare.com (top-right corner)",
        "key_required": True,
        "key_hint": "Get your API token at dash.cloudflare.com → My Profile → API Tokens",
    },
    {
        "display": "Zhipu AI (BigModel)",
        "key": "zhipu",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_required": True,
        "key_hint": "Get your API key at open.bigmodel.cn/usercenter/apikeys",
    },
    {
        "display": "Z.AI",
        "key": "z-ai",
        "base_url": "https://api.z.ai/api/paas/v4",
        "key_required": True,
        "key_hint": "Get your API key at platform.z.ai",
    },
    {
        "display": "DeepSeek",
        "key": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "key_required": True,
        "key_hint": "Get your API key at platform.deepseek.com/api_keys",
    },
    {
        "display": "Cohere",
        "key": "cohere",
        "base_url": "https://api.cohere.com/compatibility/v1",
        "key_required": True,
        "key_hint": "Get your API key at dashboard.cohere.com/api-keys",
    },
    {
        "display": "OpenRouter",
        "key": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_required": True,
        "key_hint": "Get your API key at openrouter.ai/keys",
    },
    {
        "display": "Ollama Cloud",
        "key": "ollama-cloud",
        "base_url": "https://ollama.com/v1",
        "key_required": True,
        "key_hint": "Get your API key at ollama.com/settings/api-keys",
    },
    {
        "display": "Moonshot AI (Kimi)",
        "key": "moonshot",
        "base_url": "https://api.moonshot.ai/v1",
        "key_required": True,
        "key_hint": "Get your API key at platform.moonshot.ai/console/api-key",
    },
    {
        "display": "MiniMax",
        "key": "minimax",
        "base_url": "https://api.minimax.io/v1",
        "key_required": True,
        "key_hint": "Get your API key at platform.minimax.io",
    },
    {
        "display": "Hugging Face Inference",
        "key": "huggingface",
        "base_url": "https://router.huggingface.co/v1",
        "key_required": True,
        "key_hint": "Get your token at huggingface.co/settings/tokens",
    },
    {
        "display": "xAI (Grok)",
        "key": "xai",
        "base_url": "https://api.x.ai/v1",
        "key_required": True,
        "key_hint": "Get your API key at console.x.ai",
    },
    {
        "display": "Cloudflare AI Gateway",
        "key": "cloudflare-ai-gateway",
        "base_url": "https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/workers-ai/v1",
        "account_id_required": True,
        "account_id_label": "Cloudflare Account ID",
        "account_id_hint": "Find your account ID at dash.cloudflare.com (top-right corner)",
        "gateway_id_required": True,
        "gateway_id_label": "AI Gateway Name",
        "gateway_id_hint": "The name you gave your AI Gateway in the Cloudflare dashboard",
        "key_required": True,
        "key_hint": "Get your API token at dash.cloudflare.com → My Profile → API Tokens",
    },
    {
        "display": "Vercel AI Gateway",
        "key": "vercel",
        "base_url": "https://ai-gateway.vercel.sh/v1",
        "key_required": True,
        "key_hint": "Get your API key at vercel.com/account/tokens",
    },
    {
        "display": "Venice AI",
        "key": "venice",
        "base_url": "https://api.venice.ai/api/v1",
        "key_required": True,
        "key_hint": "Get your API key at venice.ai/settings/api",
    },
    {
        "display": "OpenCode Zen (free gateway)",
        "key": "opencode-zen",
        "base_url": "https://opencode.ai/zen/v1",
        "key_required": True,
        "key_hint": "Get your API key at opencode.ai",
    },
]

# ---------------------------------------------------------------------------
# Per-provider free-tier metadata (auto-populated after quick setup)
# ---------------------------------------------------------------------------
# Source: https://github.com/tashfeenahmed/freellmapi and provider docs.
# Each entry maps provider template key → config fragments to merge.
# Keys use "provider_key/model_id" format matching model_reasoning convention.
# Limits: rpm/rpd used for capacity-aware load balancing; tpm/tpd stored for
# reference but not yet enforced (would require response token counting).
PROVIDER_FREE_INFO: dict[str, dict] = {
    "google": {
        "known_free": [
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-flash-lite",
        ],
        "model_reasoning": {
            "google/gemini-2.5-pro": "deep",
            "google/gemini-2.5-flash": "standard",
            "google/gemini-2.5-flash-lite": "exploratory",
        },
        "free_limits": {
            "google/gemini-2.5-pro":        {"requests_per_minute": 5,  "requests_per_day": 100,  "tokens_per_minute": 250000, "tokens_per_day": None},
            "google/gemini-2.5-flash":      {"requests_per_minute": 10, "requests_per_day": 20,   "tokens_per_minute": 250000, "tokens_per_day": None},
            "google/gemini-2.5-flash-lite": {"requests_per_minute": 15, "requests_per_day": 1000, "tokens_per_minute": 250000, "tokens_per_day": None},
        },
    },
    "groq": {
        "known_free": [
            "groq/llama-3.3-70b-versatile",
            "groq/llama-4-scout-17b-16e-instruct",
            "groq/llama-3.1-8b-instant",
            "groq/mixtral-8x7b-32768",
            "groq/gemma2-9b-it",
        ],
        "model_reasoning": {
            "groq/llama-3.3-70b-versatile":        "standard",
            "groq/llama-4-scout-17b-16e-instruct": "standard",
            "groq/llama-3.1-8b-instant":           "exploratory",
            "groq/mixtral-8x7b-32768":             "standard",
            "groq/gemma2-9b-it":                   "exploratory",
        },
        "free_limits": {
            "groq/llama-3.3-70b-versatile":        {"requests_per_minute": 30, "requests_per_day": 1000,  "tokens_per_minute": 6000, "tokens_per_day": 500000},
            "groq/llama-4-scout-17b-16e-instruct": {"requests_per_minute": 30, "requests_per_day": 1000,  "tokens_per_minute": 6000, "tokens_per_day": 1000000},
            "groq/llama-3.1-8b-instant":           {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": None, "tokens_per_day": None},
            "groq/mixtral-8x7b-32768":             {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": None, "tokens_per_day": None},
            "groq/gemma2-9b-it":                   {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": None, "tokens_per_day": None},
        },
    },
    "cerebras": {
        "known_free": [
            "cerebras/qwen-3-coder-480b",
            "cerebras/llama-4-maverick-17b-128e-instruct",
            "cerebras/qwen3-235b",
            "cerebras/gpt-oss-120b",
        ],
        "model_reasoning": {
            "cerebras/qwen-3-coder-480b":                  "deep",
            "cerebras/llama-4-maverick-17b-128e-instruct": "standard",
            "cerebras/qwen3-235b":                         "deep",
            "cerebras/gpt-oss-120b":                       "deep",
        },
        "free_limits": {
            "cerebras/qwen-3-coder-480b":                  {"requests_per_minute": 30, "requests_per_day": None, "tokens_per_minute": 60000, "tokens_per_day": 1000000},
            "cerebras/llama-4-maverick-17b-128e-instruct": {"requests_per_minute": 30, "requests_per_day": None, "tokens_per_minute": 60000, "tokens_per_day": 1000000},
            "cerebras/qwen3-235b":                         {"requests_per_minute": 30, "requests_per_day": None, "tokens_per_minute": 60000, "tokens_per_day": 1000000},
            "cerebras/gpt-oss-120b":                       {"requests_per_minute": 30, "requests_per_day": None, "tokens_per_minute": 60000, "tokens_per_day": 1000000},
        },
    },
    "github": {
        "known_free": [
            "github/openai/gpt-5",
        ],
        "model_reasoning": {
            "github/openai/gpt-5": "deep",
        },
        "free_limits": {
            "github/openai/gpt-5": {"requests_per_minute": 10, "requests_per_day": 50, "tokens_per_minute": None, "tokens_per_day": None},
        },
    },
    "sambanova": {
        "known_free": [
            "sambanova/Meta-Llama-3.3-70B-Instruct",
        ],
        "model_reasoning": {
            "sambanova/Meta-Llama-3.3-70B-Instruct": "standard",
        },
        "free_limits": {
            "sambanova/Meta-Llama-3.3-70B-Instruct": {"requests_per_minute": 20, "requests_per_day": None, "tokens_per_minute": None, "tokens_per_day": 200000},
        },
    },
    "mistral": {
        "known_free": [
            "mistral/mistral-large-latest",
            "mistral/magistral-medium-latest",
            "mistral/codestral-latest",
        ],
        "model_reasoning": {
            "mistral/mistral-large-latest":    "standard",
            "mistral/magistral-medium-latest": "standard",
            "mistral/codestral-latest":        "exploratory",
        },
        "free_limits": {
            "mistral/mistral-large-latest":    {"requests_per_minute": 2, "requests_per_day": None, "tokens_per_minute": 500000, "tokens_per_day": None},
            "mistral/magistral-medium-latest": {"requests_per_minute": 2, "requests_per_day": None, "tokens_per_minute": 500000, "tokens_per_day": None},
            "mistral/codestral-latest":        {"requests_per_minute": 2, "requests_per_day": None, "tokens_per_minute": 500000, "tokens_per_day": None},
        },
    },
    "cohere": {
        "known_free": [
            "cohere/command-r-plus-08-2024",
        ],
        "model_reasoning": {
            "cohere/command-r-plus-08-2024": "standard",
        },
        "free_limits": {
            "cohere/command-r-plus-08-2024": {"requests_per_minute": 20, "requests_per_day": 33, "tokens_per_minute": None, "tokens_per_day": None},
        },
    },
    "cloudflare-workers": {
        "known_free": [
            "cloudflare-workers/@cf/meta/llama-3.1-70b-instruct",
            "cloudflare-workers/@cf/meta/llama-3.1-8b-instruct",
            "cloudflare-workers/@cf/mistral/mistral-7b-instruct-v0.1",
        ],
        "model_reasoning": {
            "cloudflare-workers/@cf/meta/llama-3.1-70b-instruct":   "standard",
            "cloudflare-workers/@cf/meta/llama-3.1-8b-instruct":    "exploratory",
            "cloudflare-workers/@cf/mistral/mistral-7b-instruct-v0.1": "exploratory",
        },
        "free_limits": {},
    },
    "zhipu": {
        "known_free": [
            "zhipu/glm-4.5-flash",
        ],
        "model_reasoning": {
            "zhipu/glm-4.5-flash": "standard",
        },
        "free_limits": {},
    },
    "z-ai": {
        "known_free": [
            "z-ai/glm-4.5-air",
            "z-ai/glm-4.5-flash",
        ],
        "model_reasoning": {
            "z-ai/glm-4.5-air":   "exploratory",
            "z-ai/glm-4.5-flash": "standard",
            "z-ai/glm-4.5":       "deep",
        },
        "free_limits": {
            "z-ai/glm-4.5-air":   {"requests_per_minute": 15, "requests_per_day": None, "tokens_per_minute": None, "tokens_per_day": 1000000},
            "z-ai/glm-4.5-flash": {"requests_per_minute": 15, "requests_per_day": None, "tokens_per_minute": None, "tokens_per_day": 1000000},
        },
    },
    "moonshot": {
        "known_free": [
            "moonshot/kimi-latest",
        ],
        "model_reasoning": {
            "moonshot/kimi-latest": "standard",
        },
        "free_limits": {
            "moonshot/kimi-latest": {"requests_per_minute": 60, "requests_per_day": None, "tokens_per_minute": None, "tokens_per_day": 500000},
        },
    },
    "minimax": {
        "known_free": [
            "minimax/MiniMax-M1",
        ],
        "model_reasoning": {
            "minimax/MiniMax-M1": "standard",
        },
        "free_limits": {
            "minimax/MiniMax-M1": {"requests_per_minute": 20, "requests_per_day": None, "tokens_per_minute": 1000000, "tokens_per_day": None},
        },
    },
    "nvidia": {
        "known_free": [
            "nvidia/meta/llama-3.1-70b-instruct",
        ],
        "model_reasoning": {
            "nvidia/meta/llama-3.1-70b-instruct": "standard",
        },
        "free_limits": {
            "nvidia/meta/llama-3.1-70b-instruct": {"requests_per_minute": 40, "requests_per_day": None, "tokens_per_minute": None, "tokens_per_day": None},
        },
    },
    "xai": {
        "known_free": [
            "xai/grok-3-mini",
            "xai/grok-3",
        ],
        "model_reasoning": {
            "xai/grok-3-mini": "standard",
            "xai/grok-3":      "deep",
        },
        "free_limits": {},
    },
    "openrouter": {
        "known_free": [],
        "model_reasoning": {
            "openrouter/deepseek/deepseek-v3.1:free":            "deep",
            "openrouter/moonshotai/kimi-k2:free":                "deep",
            "openrouter/qwen/qwen3-coder:free":                  "deep",
            "openrouter/z-ai/glm-4.5-air:free":                 "standard",
            "openrouter/mistralai/mistral-7b-instruct:free":     "exploratory",
            "openrouter/meta-llama/llama-3.2-3b-instruct:free":  "exploratory",
            "openrouter/meta-llama/llama-3.1-8b-instruct:free":  "exploratory",
            "openrouter/google/gemma-2-9b-it:free":              "exploratory",
            "openrouter/qwen/qwen-2.5-7b-instruct:free":         "exploratory",
        },
        "free_limits": {
            "openrouter/deepseek/deepseek-v3.1:free": {"requests_per_minute": 20, "requests_per_day": 200, "tokens_per_minute": None, "tokens_per_day": None},
            "openrouter/moonshotai/kimi-k2:free":     {"requests_per_minute": 20, "requests_per_day": 200, "tokens_per_minute": None, "tokens_per_day": None},
            "openrouter/qwen/qwen3-coder:free":       {"requests_per_minute": 20, "requests_per_day": 200, "tokens_per_minute": None, "tokens_per_day": None},
            "openrouter/z-ai/glm-4.5-air:free":      {"requests_per_minute": 20, "requests_per_day": 200, "tokens_per_minute": None, "tokens_per_day": None},
        },
    },
    "opencode-zen": {
        "known_free": [
            "opencode-zen/big-pickle",
            "opencode-zen/deepseek-v4-flash-free",
            "opencode-zen/minimax-m2.5-free",
            "opencode-zen/nemotron-3-super-free",
        ],
        "model_reasoning": {
            "opencode-zen/big-pickle":             "standard",
            "opencode-zen/deepseek-v4-flash-free": "standard",
            "opencode-zen/minimax-m2.5-free":      "standard",
            "opencode-zen/nemotron-3-super-free":  "deep",
        },
        "free_limits": {
            "opencode-zen/big-pickle":             {"requests_per_minute": None, "requests_per_day": None, "tokens_per_minute": None, "tokens_per_day": None},
            "opencode-zen/deepseek-v4-flash-free": {"requests_per_minute": None, "requests_per_day": None, "tokens_per_minute": None, "tokens_per_day": None},
            "opencode-zen/minimax-m2.5-free":      {"requests_per_minute": None, "requests_per_day": None, "tokens_per_minute": None, "tokens_per_day": None},
            "opencode-zen/nemotron-3-super-free":  {"requests_per_minute": None, "requests_per_day": None, "tokens_per_minute": None, "tokens_per_day": None},
        },
    },
}
