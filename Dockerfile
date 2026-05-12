# ============================================================
# llmproxy Dockerfile
# ============================================================
#
# Build:
#   docker build -t llmproxy .
#
# Run the server (config volume auto-created):
#   docker run -d \
#     -p 8080:8080 \
#     -v llmproxy_config:/root/.config/llmproxy \
#     --name llmproxy \
#     llmproxy
#
# First-time setup (interactive — requires -it):
#   docker run -it --rm \
#     -v llmproxy_config:/root/.config/llmproxy \
#     llmproxy --setup
#
# Re-run setup at any time without stopping the server:
#   docker run -it --rm \
#     -v llmproxy_config:/root/.config/llmproxy \
#     llmproxy --setup
#
# After setup, restart the running container to pick up changes:
#   docker restart llmproxy
#
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

# ── Declare the volume for persistent configuration ───────────────────────
# Config lives at /root/.config/llmproxy/config.json inside this volume.
# Mount it as a named Docker volume so it persists across container restarts
# and can be written to by both the server container and a one-off --setup run.
VOLUME ["/root/.config/llmproxy"]

# ── Expose the default listen port ────────────────────────────────────────
EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────
# With no extra arguments:   starts the proxy server via gunicorn.
# With --setup:              launches the interactive wizard (needs -it).
# With --list-providers:     prints configured providers and exits.
# Any other llmproxy flags are passed through transparently.
ENTRYPOINT ["python", "-m", "llmproxy"]
CMD []
