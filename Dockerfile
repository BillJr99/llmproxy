# ============================================================
# llmproxy Dockerfile
# ============================================================
#
# Build:
#   docker build -t llmproxy .
#
# Run the server (config bind-mounted from host, runs as current user):
#   mkdir -p ~/.config/llmproxy
#   docker run -d \
#     -p 8080:8080 \
#     --user $(id -u):$(id -g) \
#     -v ~/.config/llmproxy:/config \
#     -e LLMPROXY_CONFIG=/config/config.json \
#     --name llmproxy \
#     llmproxy
#
# First-time setup (interactive — requires -it):
#   mkdir -p ~/.config/llmproxy
#   docker run -it --rm \
#     --user $(id -u):$(id -g) \
#     -v ~/.config/llmproxy:/config \
#     -e LLMPROXY_CONFIG=/config/config.json \
#     llmproxy --setup
#
# Re-run setup at any time without stopping the server:
#   docker run -it --rm \
#     --user $(id -u):$(id -g) \
#     -v ~/.config/llmproxy:/config \
#     -e LLMPROXY_CONFIG=/config/config.json \
#     llmproxy --setup
#
# After setup, restart the running container to pick up changes:
#   docker restart llmproxy
#
# Named-volume alternative (config stays inside Docker, not on the host
# filesystem — useful for CI or rootless environments):
#   docker run -d \
#     -p 8080:8080 \
#     -v llmproxy_config:/root/.config/llmproxy \
#     --name llmproxy \
#     llmproxy
#
# Pull from GHCR instead of building locally:
#   docker pull ghcr.io/billjr99/llmproxy:latest
# ============================================================

FROM python:3.12-slim AS base

# Keep Python output unbuffered so logs appear in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Install dependencies in a separate layer for cache efficiency ──────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Copy the application package ──────────────────────────────────────────
COPY llmproxy/ ./llmproxy/
COPY setup.py .
RUN pip install -e .

# ── Expose the default listen port ────────────────────────────────────────
EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────
# With no extra arguments:   starts the proxy server via gunicorn.
# With --setup:              launches the interactive wizard (needs -it).
# With --list-providers:     prints configured providers and exits.
# Any other llmproxy flags are passed through transparently.
#
# The recommended way to run is with --user $(id -u):$(id -g) and a bind
# mount of ~/.config/llmproxy to /config, plus LLMPROXY_CONFIG=/config/config.json.
# This ensures config files are owned by the host user and readable without
# sudo.  See the comments at the top of this file for full examples.
ENTRYPOINT ["python", "-m", "llmproxy"]
CMD []
