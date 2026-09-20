"""
llmproxy — OpenAI-compatible multi-provider LLM proxy.

Routes requests to upstream providers based on a provider-prefix model naming
convention: <provider_name>/<upstream_model_id>.
"""

__version__ = "1.0.0"
__author__ = "llmproxy"

# What llmproxy calls itself on every outbound request.
#
# It had no identity at all before: where a client sent no User-Agent, whatever
# HTTP library happened to be in use filled one in, so upstreams saw
# "python-requests/2.33.1" -- a library default leaking out rather than a
# decision. Worse, a client's own User-Agent was relayed verbatim, so a caller
# using urllib got a Cloudflare block page instead of an answer, and the choice
# of HTTP library silently decided whether an upstream would respond at all.
#
# Lives here rather than in server.py so flagship.py, setup_wizard.py,
# github_pr.py and the scrapers under scripts/ can all identify themselves the
# same way without importing the server.
USER_AGENT = f"llmproxy/{__version__}"
