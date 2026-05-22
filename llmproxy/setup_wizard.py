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
