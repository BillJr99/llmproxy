# ============================================================
# llmproxy Dockerfile
# ============================================================
#
# Build:
#   docker build -t llmproxy .
#
# The image runs as a non-root user by default — no --user flag required.
# When bind-mounting a host config directory, pass --user $(id -u):$(id -g)
# so files created inside the container are owned by you on the host.
#
# Run the server (config bind-mounted from host):
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
# After setup, restart the running container to pick up host/port changes:
#   docker restart llmproxy
#
# Web admin UI: available on the same published port at /admin (e.g.
# http://localhost:8080/admin). The admin API serves loopback-only unless an
# admin token is set; since the container binds 0.0.0.0, pass a token to use it
# remotely:  -e LLMPROXY_ADMIN_TOKEN=choose-a-strong-token
# Provider api_key / base_url may use ${VAR} env references (e.g.
# "api_key": "${OPENAI_API_KEY}") resolved at request time, so pass secrets via
# -e rather than baking them into the bind-mounted config.json.
#
# Named-volume alternative (config stays inside Docker, not on the host
# filesystem — useful for CI or rootless environments). Mount over the
# default config location under the non-root user's home:
#   docker run -d \
#     -p 8080:8080 \
#     -v llmproxy_config:/home/llmproxy/.config/llmproxy \
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
COPY llmproxy/setup.py .
RUN pip install -e .

# Ship the free-models scraper too, so update_believed_free_on_startup can run
# it in-process. Optional at runtime — the server degrades gracefully if absent.
COPY scripts/ ./scripts/

# ── Run as a non-root user ────────────────────────────────────────────────
# Create an unprivileged user (uid 1000) in the root group (gid 0) and make
# the config + home directories group-writable. The image therefore runs as
# non-root by default (no --user required), and also works under an arbitrary
# `--user <uid>:0` (e.g. OpenShift, or `--user $(id -u):0`) because everything
# the process writes is group-root-writable. HOME is fixed so Path.home()
# (used for the default ~/.config/llmproxy config path) resolves even when the
# uid has no /etc/passwd entry.
ENV HOME=/home/llmproxy
RUN useradd --uid 1000 --gid 0 --create-home --home-dir "$HOME" llmproxy \
    && mkdir -p /config "$HOME/.config/llmproxy" \
    && chgrp -R 0 /config "$HOME" /app \
    && chmod -R g+rwX /config "$HOME" /app
USER 1000:0

# ── Expose the default listen port ────────────────────────────────────────
EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────
# With no extra arguments:   starts the proxy server via gunicorn.
# With --setup:              launches the interactive wizard (needs -it).
# With --list-providers:     prints configured providers and exits.
# Any other llmproxy flags are passed through transparently.
ENTRYPOINT ["python", "-m", "llmproxy"]
CMD []
