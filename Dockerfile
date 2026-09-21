# ============================================================
# llmproxy Dockerfile
# ============================================================
#
# Build:
#   docker build -t llmproxy .
#
# The image runs as a non-root user by default — no --user flag required.
# When bind-mounting a host config directory, pass --user $(id -u):0 so files
# created inside the container are owned by you on the host. Use group 0, not
# $(id -g): this image's writable directories are group-root (see the useradd
# stanza below), and a host GID of 1000 cannot write them. Group 0 grants no
# root privilege here, because the UID is still yours.
#
# Run the server (config bind-mounted from host, state in a named volume):
#   mkdir -p ~/.config/llmproxy
#   docker run -d \
#     -p 8080:8080 \
#     --user $(id -u):0 \
#     -v ~/.config/llmproxy:/config \
#     -v llmproxy_state:/state \
#     -e LLMPROXY_CONFIG=/config/config.json \
#     --name llmproxy \
#     llmproxy
#
# Dropping -v llmproxy_state:/state is fine for a quick trial, but everything
# llmproxy learns (routing metadata, flagship membership, refresh timestamps)
# then lives only in the container and is lost when it is removed. To keep the
# state beside config.json on the host instead, add
# -e LLMPROXY_STATE_DIR=/config and omit the state volume.
#
# If the host directory was created by Docker rather than by mkdir, it is owned
# by root and nothing in the container can write it:
#   sudo chown -R $(id -u):$(id -g) ~/.config/llmproxy
#
# If upstream provider domains fail to resolve (a common symptom after a Docker
# update that resets the daemon's DNS/iptables), pin public resolvers with
# --dns (the compose file already sets these for you):
#     docker run -d --dns 1.1.1.1 --dns 8.8.8.8  ...  llmproxy
#
# First-time setup (interactive — requires -it):
#   mkdir -p ~/.config/llmproxy
#   docker run -it --rm \
#     --user $(id -u):0 \
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

# ── Network debugging tools ────────────────────────────────────────────────
# The slim base ships without DNS/ping utilities, so `docker exec` debugging of
# upstream connectivity (e.g. a provider domain failing to resolve) is awkward.
# Add a minimal, well-known set: ping (iputils-ping), nslookup/dig (dnsutils),
# curl, and ip/ss (iproute2). ca-certificates keeps HTTPS verification working.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        dnsutils \
        iproute2 \
        iputils-ping \
    && rm -rf /var/lib/apt/lists/*

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
# Create an unprivileged user (uid 1000) in the root group (gid 0) and make the
# config, state, home and app directories group-writable. The image therefore
# runs as non-root by default (no --user required), and also works under an
# arbitrary `--user <uid>:0` (e.g. OpenShift, or `--user $(id -u):0`) because
# everything the process writes is group-root-writable.
#
# The GID matters as much as the UID. `--user $(id -u):$(id -g)` passes the
# host's group, normally 1000, which is in neither group 0 nor the owner of
# these directories — so only the "other" permission bits apply and the process
# cannot write them. Pass `:0` as the group instead; it grants no root
# privilege, since the UID is still yours, and it is the group these
# directories are shared with.
#
# HOME is fixed so Path.home() (used for the default ~/.config/llmproxy config
# path) resolves even when the uid has no /etc/passwd entry.
ENV HOME=/home/llmproxy
RUN useradd --uid 1000 --gid 0 --create-home --home-dir "$HOME" llmproxy \
    && mkdir -p /config /state "$HOME/.config/llmproxy" \
    && chgrp -R 0 /config /state "$HOME" /app \
    && chmod -R g+rwX /config /state "$HOME" /app
USER 1000:0

# ── Machine-written state ─────────────────────────────────────────────────
# routing_metadata.json, flagship_models.json, update_state.json,
# cost_probe_state.json and pr_state.json are rewritten on their own cadences
# and must be writable: each one carries the last-run timestamp that throttles
# its refresh, so a directory that cannot be written means every refresh is
# permanently due. Defaulting them to /state keeps them off the config mount,
# which is then free to be read-only. Unset LLMPROXY_STATE_DIR to put them back
# beside config.json.
ENV LLMPROXY_STATE_DIR=/state

# ── Expose the default listen port ────────────────────────────────────────
EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────
# With no extra arguments:   starts the proxy server via gunicorn.
# With --setup:              launches the interactive wizard (needs -it).
# With --list-providers:     prints configured providers and exits.
# Any other llmproxy flags are passed through transparently.
ENTRYPOINT ["python", "-m", "llmproxy"]
CMD []
